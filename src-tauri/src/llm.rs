//! Phase 2: the greeting. A hypothesis about a face goes in, a question comes out.
//!
//! The model never gets to assert how you feel. It gets to ask. If the key is
//! missing, the network is down, or the response is malformed, a canned greeting
//! takes over and the session continues -- a lost sentence is not worth a lost
//! journal entry.

use serde::Serialize;
use serde_json::{json, Value};
use std::time::Duration;

pub const BASE_URL: &str = "https://integrate.api.nvidia.com/v1";
/// Offered in the app, never applied behind your back.
pub const SUGGESTED_MODEL: &str = "deepseek-ai/deepseek-v4-flash-0731";

/// Latency budgets. A greeting nobody hears is worse than a canned one, so these
/// are part of the behaviour rather than a tuning detail.
///
/// Someone is sitting there waiting to be spoken to, and NIM's free tier queues
/// unpredictably (5-65s observed), so the two calls a person waits on fail fast.
/// Nobody is waiting on the extraction, so it gets the time to return real
/// metadata instead of a canned blank.
const GREET: Budget = Budget { timeout: 12, tries: 1 };
const REPLY: Budget = Budget { timeout: 15, tries: 1 };
const EXTRACT: Budget = Budget { timeout: 45, tries: 2 };
const SONG: Budget = Budget { timeout: 25, tries: 1 };

#[derive(Clone, Copy, Debug, PartialEq)]
struct Budget {
    timeout: u64,
    tries: u32,
}

/// A label is a model's word, not yours. These are the softened forms it is
/// allowed to offer back to you, and every one of them is easy to say "no" to.
///
/// Neutral is absent on purpose: "you look steady" is not a question.
pub fn looks_like(label: &str) -> &'static str {
    match label {
        "Anger" => "a bit wound up",
        "Contempt" => "a bit unimpressed",
        "Disgust" => "a bit put off",
        "Fear" => "a bit on edge",
        "Happiness" => "pretty bright",
        "Sadness" => "a little low",
        "Surprise" => "caught off guard",
        _ => "",
    }
}

const SYSTEM: &str = "You are Hyperjournal, a journal that greets one person by name when they sit down.

You are given a guess about their mood made by a camera. That guess is wrong roughly a third of the time and you must treat it as a guess, never as a fact.

Write exactly one spoken sentence, at most 28 words:
- Greet them by name and ask how their day has gone, right from the start of it. Not \"how are you now\" - you want the whole day.
- Any mood guess must be phrased as an open question they can disagree with.
- Never assert how they feel. \"You look a little low - how has your day been, right from the morning?\" is right. \"I can see you're sad\" is forbidden.
- If no mood guess is given, just ask openly about their day from the start.
- No emoji, no preamble, no quotation marks. Output the sentence only.";

const CONVERSE_SYSTEM: &str = "You are Hyperjournal, a journal having a short spoken conversation with one person about how their day is going.

Rules, all of them firm:
- One or two sentences. Never more.
- At most one question, and only if it opens something up.
- Never diagnose, never advise unless asked, never cheerlead.
- Wind down by the third exchange and let them go. A journal that will not stop talking is an alarm clock.
- Plain spoken English. No emoji, no lists, no quotation marks.";

const EXTRACT_SYSTEM: &str = "Read a short journal conversation and return JSON only, no prose, with exactly these keys:

  \"self_reported\": a short lowercase phrase in the PERSON'S OWN terms for how they said they feel, e.g. \"tired\", \"anxious about work\", \"good, just busy\". If they contradicted the camera's guess, follow the person. If they truly never said, use \"\".
  \"themes\": 1-3 short lowercase kebab-case tags, e.g. [\"sleep\",\"work-pressure\"].
  \"summary\": one sentence, under 20 words, in the third person, about what was going on.";

const SONG_SYSTEM: &str = "Pick one real, well-known song that suits how someone's day actually went.

Return JSON only, no prose, with exactly these keys:
  \"title\": the song title, exactly as released.
  \"artist\": the performing artist.
  \"why\": one short sentence, under 15 words, on why it fits. Speak to them, not about them.

Pick something that exists and is easy to find. Match the feeling rather than the words: a heavy day does not need a sad song, and a good day does not need a triumphant one. Never pick a song about self-harm or despair.";

pub const WINDING_DOWN: [&str; 3] = [
    "That makes sense. Anything else sitting with you?",
    "Got it. Is there more to that, or shall I write it down?",
    "Thanks for telling me. Go easy on yourself tonight, and I'll note this down.",
];

/// The name Hyperjournal speaks under, and the speaker label in the transcript.
pub const VOICE: &str = "Hyperjournal";

#[derive(Debug)]
pub enum Error {
    NoKey,
    NoModel,
    Network(String),
    Empty,
}

impl Error {
    /// The short name that lands in an entry's `source` field, so a canned
    /// sentence says why it was canned.
    fn kind(&self) -> &'static str {
        match self {
            Error::NoKey => "RuntimeError",
            Error::NoModel => "RuntimeError",
            Error::Network(_) => "URLError",
            Error::Empty => "ValueError",
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NoKey => write!(f, "no API key yet"),
            Error::NoModel => write!(f, "no model chosen yet"),
            Error::Network(e) => write!(f, "{e}"),
            Error::Empty => write!(f, "empty completion"),
        }
    }
}

fn canned_from(e: &Error) -> String {
    format!("canned ({})", e.kind())
}

pub struct Llm {
    http: reqwest::Client,
    key: Option<String>,
    model: Option<String>,
}

#[derive(Serialize, Debug, Default, PartialEq)]
pub struct Extracted {
    pub self_reported: String,
    pub themes: Vec<String>,
    pub summary: String,
}

impl Llm {
    pub fn new(key: Option<String>, model: Option<String>) -> Self {
        Self { http: reqwest::Client::new(), key, model }
    }

    pub fn has_key(&self) -> bool {
        self.key.as_ref().is_some_and(|k| !k.is_empty())
    }

    /// One call to a NIM chat endpoint. Errors on any failure; callers fall back.
    ///
    /// Retried because this endpoint intermittently queues past the timeout and
    /// intermittently returns a message whose `content` is null (the answer went
    /// into the model's reasoning channel instead). Both are transient, and one
    /// retry is cheaper than a canned sentence.
    async fn chat(
        &self,
        messages: Value,
        max_tokens: u32,
        temperature: f64,
        budget: Budget,
    ) -> Result<String, Error> {
        let key = self.key.as_deref().filter(|k| !k.is_empty()).ok_or(Error::NoKey)?;
        let model = self.model.as_deref().filter(|m| !m.is_empty()).ok_or(Error::NoModel)?;
        let body = json!({
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        });

        let mut last = Error::Empty;
        for _ in 0..budget.tries {
            let sent = self
                .http
                .post(format!("{BASE_URL}/chat/completions"))
                .bearer_auth(key)
                .timeout(Duration::from_secs(budget.timeout))
                .json(&body)
                .send()
                .await;
            match sent {
                Err(e) => last = Error::Network(e.to_string()),
                Ok(r) if !r.status().is_success() => {
                    last = Error::Network(format!("HTTP {}", r.status().as_u16()))
                }
                Ok(r) => match r.json::<Value>().await {
                    Err(e) => last = Error::Network(e.to_string()),
                    Ok(data) => {
                        // `content` is null when the model answered in its
                        // reasoning channel
                        let text = data["choices"][0]["message"]["content"]
                            .as_str()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if text.is_empty() {
                            last = Error::Empty;
                        } else {
                            return Ok(text);
                        }
                    }
                },
            }
        }
        Err(last)
    }

    pub async fn list_models(&self) -> Result<Vec<String>, Error> {
        let key = self.key.as_deref().filter(|k| !k.is_empty()).ok_or(Error::NoKey)?;
        let r = self
            .http
            .get(format!("{BASE_URL}/models"))
            .bearer_auth(key)
            .timeout(Duration::from_secs(EXTRACT.timeout))
            .send()
            .await
            .map_err(|e| Error::Network(e.to_string()))?;
        if !r.status().is_success() {
            return Err(Error::Network(format!("HTTP {}", r.status().as_u16())));
        }
        let data: Value = r.json().await.map_err(|e| Error::Network(e.to_string()))?;
        let mut out: Vec<String> = data["data"]
            .as_array()
            .map(|a| a.iter().filter_map(|m| m["id"].as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        out.sort();
        Ok(out)
    }

    /// Returns (sentence, source). `source` is "nim" or "canned (...)".
    ///
    /// A low-confidence guess, or a neutral one, is dropped entirely rather than
    /// softened -- "you look steady" is not a question worth answering.
    pub async fn greet(
        &self,
        name: &str,
        label: &str,
        confidence: f64,
        floor: f64,
        part_of_day: &str,
        recent: &[String],
    ) -> (String, String) {
        let mood = if confidence >= floor { looks_like(label) } else { "" };
        let fallback = canned(name, mood, part_of_day);

        let mut context = vec![
            format!("Person's name: {name}"),
            format!("Time of day: {part_of_day}"),
        ];
        context.push(if mood.is_empty() {
            "Camera's mood guess: none confident enough to use. Ask an open question.".into()
        } else {
            format!("Camera's mood guess: {label} (confidence {confidence:.2}), i.e. they may look {mood}")
        });
        if !recent.is_empty() {
            context.push(format!("Their last few entries: {}", recent.join("; ")));
        }

        let messages = json!([
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": context.join("\n")},
        ]);
        match self.chat(messages, 120, 0.7, GREET).await {
            Ok(text) => (text.trim_matches('"').to_string(), "nim".into()),
            Err(e) => (fallback, canned_from(&e)),
        }
    }

    /// Next thing Hyperjournal says. Returns (text, source).
    ///
    /// `winding_down` is decided by the caller from your own settings. It is not
    /// inferred from a turn count here, because how long you want to talk is not
    /// this function's business.
    pub async fn reply(
        &self,
        name: &str,
        turns: &[(String, String)],
        hypothesis: Option<(&str, f64)>,
        winding_down: bool,
    ) -> (String, String) {
        let exchanges = turns.iter().filter(|(who, _)| who != VOICE).count();
        let fallback =
            WINDING_DOWN[exchanges.saturating_sub(1).min(WINDING_DOWN.len() - 1)].to_string();

        let mut context = vec![format!("The person's name is {name}.")];
        if let Some((label, confidence)) = hypothesis {
            context.push(format!(
                "The camera guessed {label} at {confidence:.2} confidence. \
                 It is often wrong; if they said otherwise, they are right."
            ));
        }
        if winding_down {
            context.push("This is the last exchange. Acknowledge and let them go.".into());
        }

        let mut messages = vec![json!({
            "role": "system",
            "content": format!("{CONVERSE_SYSTEM}\n\n{}", context.join(" ")),
        })];
        for (who, text) in turns {
            messages.push(json!({
                "role": if who == VOICE { "assistant" } else { "user" },
                "content": text,
            }));
        }

        match self.chat(Value::Array(messages), 120, 0.7, REPLY).await {
            Ok(text) => (text.trim_matches('"').to_string(), "nim".into()),
            Err(e) => (fallback, canned_from(&e)),
        }
    }

    /// Structured metadata for the frontmatter. Returns (data, source).
    ///
    /// A malformed response falls back rather than writing a malformed entry.
    pub async fn extract(&self, turns: &[(String, String)]) -> (Extracted, String) {
        let said: Vec<&String> = turns.iter().filter(|(who, _)| who != VOICE).map(|(_, t)| t).collect();
        if said.is_empty() {
            return (Extracted::default(), "silent".into());
        }
        // honest degradation: their own first words stand in, and no invented summary
        let degraded = || Extracted {
            self_reported: in_their_words(said[0], 60),
            themes: Vec::new(),
            summary: String::new(),
        };

        let messages = json!([
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": transcript(turns)},
        ]);
        let raw = match self.chat(messages, 300, 0.2, EXTRACT).await {
            Ok(raw) => raw,
            Err(e) => return (degraded(), canned_from(&e)),
        };

        let Some(data) = first_json(&raw) else {
            return (degraded(), "canned (malformed)".into());
        };
        if !data["themes"].is_array() && !data["themes"].is_null() {
            return (degraded(), "canned (malformed)".into());
        }
        (
            Extracted {
                self_reported: clip(data["self_reported"].as_str().unwrap_or("").to_lowercase().trim(), 60),
                themes: data["themes"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .filter_map(|t| t.as_str())
                            .map(|t| clip(t.to_lowercase().trim(), 30))
                            .take(3)
                            .collect()
                    })
                    .unwrap_or_default(),
                summary: clip(data["summary"].as_str().unwrap_or("").trim(), 200),
            },
            "nim".into(),
        )
    }

    /// One song for the day, or None. Never blocks an entry from being written.
    ///
    /// Reads the conversation rather than the extracted summary, so it can run
    /// beside the extraction call instead of queueing behind it.
    pub async fn suggest_song(&self, turns: &[(String, String)]) -> Option<crate::journal::Song> {
        if !turns.iter().any(|(who, _)| who != VOICE) {
            return None;
        }
        let messages = json!([
            {"role": "system", "content": SONG_SYSTEM},
            {"role": "user", "content": transcript(turns)},
        ]);
        let raw = self.chat(messages, 200, 0.8, SONG).await.ok()?;
        let data = first_json(&raw)?;
        let title = data["title"].as_str().filter(|s| !s.is_empty())?;
        let artist = data["artist"].as_str().filter(|s| !s.is_empty())?;
        Some(crate::journal::Song {
            title: clip(title.trim(), 80),
            artist: clip(artist.trim(), 60),
            why: clip(data["why"].as_str().unwrap_or("").trim(), 120),
        })
    }
}

fn transcript(turns: &[(String, String)]) -> String {
    turns
        .iter()
        .map(|(who, text)| format!("{who}: {text}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Truncate on a character boundary, never mid-codepoint.
fn clip(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

pub fn canned(name: &str, mood: &str, part_of_day: &str) -> String {
    if mood.is_empty() {
        let mut head = part_of_day.chars();
        let capitalised = match head.next() {
            Some(c) => c.to_uppercase().collect::<String>() + head.as_str(),
            None => String::new(),
        };
        format!("{capitalised}, {name} — how has your day gone, right from the start?")
    } else {
        format!("You look {mood} this {part_of_day}, {name} — how has your day been, from the morning on?")
    }
}

/// Their own first words, cut at a sentence or a word -- never mid-word.
///
/// This is the fallback when the extraction call fails. It has to stay their
/// words rather than an invented phrase, but "very very tiring and it's fully
/// wasted, you" reads as a bug, because it is one.
pub fn in_their_words(said: &str, limit: usize) -> String {
    let mut text = said.split_whitespace().collect::<Vec<_>>().join(" ");
    for stop in [". ", "! ", "? "] {
        if let Some((head, _)) = text.split_once(stop) {
            if head.chars().count() <= limit {
                text = head.to_string();
                break;
            }
        }
    }
    if text.chars().count() > limit {
        text = clip(&text, limit);
        text = text.rsplit_once(' ').map_or(text.clone(), |(head, _)| head.to_string());
        // a dangling half-clause reads worse than a short one
        if let Some((head, _)) = text.rsplit_once(',') {
            if head.chars().count() >= 20 {
                text = head.to_string();
            }
        }
    }
    text.trim_matches([' ', ',', ';', ':', '-'])
        .trim_end_matches(['.', '!', '?'])
        .to_lowercase()
}

/// Models wrap JSON in prose and code fences no matter how firmly you ask.
pub fn first_json(text: &str) -> Option<Value> {
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end < start {
        return None;
    }
    serde_json::from_str(&text[start..=end]).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_greeting_asks_about_the_whole_day_not_just_this_minute() {
        let open = canned("Yash", "", "morning");
        assert!(open.starts_with("Morning, Yash"), "{open}");
        assert!(open.contains("from the start"), "{open}");

        let low = canned("Yash", looks_like("Sadness"), "evening");
        assert!(low.contains("low") && low.contains("day"), "{low}");
    }

    #[test]
    fn neutral_has_no_softened_form_so_it_drops() {
        assert_eq!(looks_like("Neutral"), "");
        assert_eq!(looks_like("Sadness"), "a little low");
        assert_eq!(looks_like("whatever"), "");
        // and a dropped guess must leave no mood word behind
        let text = canned("Yash", looks_like("Neutral"), "evening");
        assert!(!text.contains("steady"), "{text}");
        assert!(text.ends_with('?'), "{text}");
    }

    #[test]
    fn a_fallback_self_report_is_their_words_and_never_ends_mid_word() {
        let real = "Bro, today was very very tiring and it's fully wasted, you know, I did nothing.";
        let got = in_their_words(real, 60);
        assert_eq!(got, "bro, today was very very tiring and it's fully wasted");
        assert!(!got.ends_with(','), "{got}");

        assert_eq!(in_their_words("Just tired. Didn't sleep.", 60), "just tired");
        assert_eq!(in_their_words("", 60), "");
        assert!(in_their_words(&"word ".repeat(40), 60).chars().count() <= 60);
    }

    #[test]
    fn malformed_model_output_must_not_reach_the_frontmatter() {
        assert_eq!(first_json("here you go: {\"a\": 1} hope that helps"), Some(json!({"a": 1})));
        assert_eq!(first_json("no json here"), None);
        assert_eq!(first_json("{broken"), None);
        // a fenced block is the common case
        assert_eq!(first_json("```json\n{\"a\": 2}\n```"), Some(json!({"a": 2})));
    }

    #[test]
    fn the_latency_budget_is_part_of_the_behaviour() {
        // a person is waiting on these two
        assert_eq!(GREET, Budget { timeout: 12, tries: 1 });
        assert_eq!(REPLY, Budget { timeout: 15, tries: 1 });
        // nobody is waiting on this one, so it gets time to return real metadata
        assert!(EXTRACT.timeout >= 45 && EXTRACT.tries == 2);
    }

    #[tokio::test]
    async fn with_no_key_every_call_degrades_instead_of_failing() {
        let llm = Llm::new(None, None);
        assert!(!llm.has_key());

        let (text, source) = llm.greet("Yash", "Sadness", 0.21, 0.45, "evening", &[]).await;
        assert!(source.starts_with("canned"), "{source}");
        // below the floor the guess is dropped, and no mood word survives
        assert!(!text.contains("low") && !text.to_lowercase().contains("sad"), "{text}");
        assert!(text.ends_with('?'), "{text}");

        // conversation winds down on its own even with no network
        let turns = vec![
            (VOICE.to_string(), "How's it going?".to_string()),
            ("Yash".to_string(), "Tired.".to_string()),
        ];
        let (t, _) = llm.reply("Yash", &turns, None, false).await;
        assert!(!t.is_empty() && t.len() < 200, "{t}");

        let long: Vec<(String, String)> = ["a", "b", "c", "d", "e", "f"]
            .iter()
            .enumerate()
            .map(|(i, s)| {
                (if i % 2 == 0 { VOICE } else { "Yash" }.to_string(), s.to_string())
            })
            .collect();
        let (t3, _) = llm.reply("Yash", &long, None, false).await;
        assert_eq!(t3, WINDING_DOWN[WINDING_DOWN.len() - 1]);
    }

    #[tokio::test]
    async fn silence_extracts_to_nothing_and_nothing_is_invented() {
        let llm = Llm::new(None, None);
        let turns = vec![(VOICE.to_string(), "How are you?".to_string())];
        let (data, source) = llm.extract(&turns).await;
        assert_eq!(data, Extracted::default());
        assert_eq!(source, "silent");

        // a song is a nicety: no material to work from means no song, never a guess
        assert!(llm.suggest_song(&turns).await.is_none());
    }

    #[tokio::test]
    async fn a_failed_extraction_falls_back_to_their_own_words() {
        let llm = Llm::new(None, None);
        let turns = vec![
            (VOICE.to_string(), "How was it?".to_string()),
            ("Yash".to_string(), "Bro, today was very very tiring and it's fully wasted, you know.".to_string()),
        ];
        let (data, source) = llm.extract(&turns).await;
        assert!(source.starts_with("canned"), "{source}");
        assert_eq!(data.self_reported, "bro, today was very very tiring and it's fully wasted");
        assert!(data.themes.is_empty(), "no themes may be invented");
        assert!(data.summary.is_empty(), "no summary may be invented");
    }
}
