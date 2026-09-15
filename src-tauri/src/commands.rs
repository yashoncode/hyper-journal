//! What the frontend calls. One command per route the Python server had.
//!
//! The Python held its state in module globals behind a `threading.Lock`. Here
//! it is one `Mutex<App>`, with the same rule: the lock is never held across a
//! network call, because a slow model must not stop the camera.
//!
//! Expected failures come back as `{"error": "..."}` rather than a rejected
//! promise, which is the shape the frontend already checks for.

use crate::detect::{self, Detector};
use crate::journal::{self, Hypothesis, NewEntry};
use crate::llm::{self, Llm, VOICE};
use crate::paths;
use crate::settings::{self, Settings};
use crate::stt::{self, Speech};
use crate::vision;
use chrono::{Local, Timelike};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::Mutex;
use tauri::{AppHandle, State};


/// ~20 minutes of frames, then it stops accumulating.
const MAX_READINGS: usize = 4000;

#[derive(Default)]
pub struct Session {
    pub turns: Vec<(String, String)>,
    pub hypothesis: Option<Hypothesis>,
    pub readings: Vec<Vec<f32>>,
}

pub struct App {
    pub settings: Settings,
    pub session: Option<Session>,
    /// What this key can reach, filled on first ask.
    pub models: Vec<String>,
    /// The smoothed distribution, and the most recent hypothesis for `greet`.
    pub probs: Vec<f32>,
    pub baseline: Option<Vec<f32>>,
    /// A list while capturing a baseline, else None.
    pub collecting: Option<Vec<Vec<f32>>>,
    pub last: Option<(String, f64)>,
    /// Both ONNX models, built on the first frame rather than at startup, so
    /// opening the app does not wait two seconds for a camera nobody may use.
    pub detector: Option<Detector>,
    /// The speech model, fetched in the background at startup.
    pub speech: Speech,
}

impl App {
    pub fn new(settings: Settings, baseline: Option<Vec<f32>>) -> Self {
        Self {
            settings,
            session: None,
            models: Vec::new(),
            probs: vec![1.0 / vision::LABELS.len() as f32; vision::LABELS.len()],
            baseline,
            collecting: None,
            last: None,
            detector: None,
            speech: Speech::Fetching(0.0),
        }
    }

    fn llm(&self) -> Llm {
        Llm::new(self.settings.api_key(), self.settings.model())
    }
}

pub type Shared = Mutex<App>;

fn err(message: impl Into<String>) -> Value {
    json!({ "error": message.into() })
}

fn part_of_day() -> &'static str {
    match Local::now().hour() {
        h if h < 12 => "morning",
        h if h < 17 => "afternoon",
        _ => "evening",
    }
}

// --- settings ---------------------------------------------------------------

#[tauri::command]
pub fn config(state: State<'_, Shared>) -> Value {
    let app = state.lock().unwrap();
    json!({
        "settings": app.settings.as_json(),      // never includes the API key
        "key_set": app.settings.api_key().is_some(),
        "fields": settings::FIELDS.iter().copied().collect::<BTreeMap<_, _>>(),
        "voice": VOICE,
    })
}

#[tauri::command]
pub fn set_config(state: State<'_, Shared>, updates: BTreeMap<String, String>) -> Value {
    let allowed: BTreeMap<String, String> = updates
        .into_iter()
        .filter(|(k, _)| {
            k == settings::API_KEY || settings::FIELDS.iter().any(|(f, _)| f == k)
        })
        .collect();
    if allowed.is_empty() {
        return err("nothing to change");
    }
    let mut app = state.lock().unwrap();
    let saved: Vec<String> = allowed.keys().cloned().collect();
    match app.settings.save(allowed) {
        Ok(()) => json!({ "saved": saved }),
        Err(e) => err(format!("could not save settings: {e}")),
    }
}

#[tauri::command]
pub async fn models(state: State<'_, Shared>) -> Result<Value, ()> {
    let (cached, llm) = {
        let app = state.lock().unwrap();
        (app.models.clone(), app.llm())
    };
    if !cached.is_empty() {
        return Ok(json!(cached));
    }
    match llm.list_models().await {
        Ok(list) => {
            state.lock().unwrap().models = list.clone();
            Ok(json!(list))
        }
        Err(e) => Ok(err(format!("{e}"))),
    }
}

// --- the journal ------------------------------------------------------------

#[derive(Serialize)]
struct EntryRow {
    path: String,
    date: String,
    time: String,
    confirmed: bool,
    self_reported: Option<String>,
    summary: String,
    themes: Vec<String>,
    thought: String,
    song: Option<journal::Song>,
    body: String,
}

#[tauri::command]
pub fn entries(app: AppHandle) -> Value {
    let root = paths::journal_root(&app);
    let rows: Vec<EntryRow> = journal::entries(&root, Some(60))
        .into_iter()
        .map(|(path, post)| EntryRow {
            path: path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/"),
            date: post.meta.journal_date.clone(),
            time: post.meta.timestamp.chars().skip(11).take(5).collect(),
            confirmed: post.meta.confirmed,
            self_reported: post.meta.self_reported.clone(),
            summary: post.meta.summary.clone(),
            themes: post.meta.themes.clone(),
            thought: vision::thought_line(&post.meta.hypothesis, post.meta.confirmed),
            song: post.meta.song.clone(),
            body: post.content,
        })
        .collect();
    json!(rows)
}

#[tauri::command]
pub fn delete_entry(app: AppHandle, path: String) -> Value {
    let root = paths::journal_root(&app);
    let target = root.join(&path);

    // a path from the frontend is not to be trusted with a rename
    let (Ok(root), Ok(target)) = (root.canonicalize(), target.canonicalize()) else {
        return err("no such entry");
    };
    if target.extension().is_none_or(|e| e != "md")
        || !target.starts_with(&root)
        || !target.is_file()
    {
        return err("no such entry");
    }
    match journal::discard(&target, &root) {
        Ok(_) => json!({ "deleted": path }),
        Err(e) => err(format!("could not delete: {e}")),
    }
}

#[tauri::command]
pub fn delete_all(app: AppHandle) -> Value {
    let root = paths::journal_root(&app);
    let gone = journal::entry_paths(&root)
        .iter()
        .filter(|p| journal::discard(p, &root).is_ok())
        .count();
    json!({ "deleted": gone })
}

// --- the conversation -------------------------------------------------------

#[tauri::command]
pub async fn greet(app: AppHandle, state: State<'_, Shared>) -> Result<Value, ()> {
    let (llm, name, floor, hypothesis, last) = {
        let a = state.lock().unwrap();
        let calibrated = a.baseline.is_some();
        let shown = vision::calibrate(&a.probs, a.baseline.as_deref());
        // A face is not required. The page already tells people that with no
        // camera "everything else still works", and llm.greet asks an open
        // question when it is given no guess to soften.
        let block = a.last.as_ref().map(|_| vision::hypothesis_block(&shown, calibrated));
        (a.llm(), a.settings.name(), a.settings.floor(), block, a.last.clone())
    };

    let recent = journal::summaries(&paths::journal_root(&app), 3);
    let (label, confidence) = last.clone().unwrap_or_default();

    // the network call is slow and must not hold the lock
    let (text, source) = llm
        .greet(&name, &label, confidence, floor, part_of_day(), &recent)
        .await;

    let mut a = state.lock().unwrap();
    a.session = Some(Session {
        turns: vec![(VOICE.to_string(), text.clone())],
        hypothesis,
        readings: Vec::new(),
    });
    Ok(json!({
        "text": text,
        "source": source,
        "label": label,
        "confidence": (confidence * 1000.0).round() / 1000.0,
        "exchanges": 0,
    }))
}

#[tauri::command]
pub async fn say(state: State<'_, Shared>, text: String) -> Result<Value, ()> {
    let said = text.trim().to_string();
    let (llm, name, limit, turns, hypothesis) = {
        let mut a = state.lock().unwrap();
        if a.session.is_none() {
            return Ok(err("no session; greet first"));
        }
        if said.is_empty() {
            return Ok(err("nothing heard"));
        }
        let name = a.settings.name();
        let limit = a.settings.max_exchanges();
        let llm = a.llm();
        let session = a.session.as_mut().unwrap();
        session.turns.push((name.clone(), said));
        let hypothesis = session
            .hypothesis
            .as_ref()
            .map(|h| (h.label.clone(), h.confidence));
        (llm, name, limit, session.turns.clone(), hypothesis)
    };

    let exchanges = turns.iter().filter(|(who, _)| who != VOICE).count() as u32;
    // 0 or unset means talk for as long as you like
    if limit > 0 && exchanges >= limit {
        return Ok(json!({
            "text": "", "source": "done", "exchanges": exchanges, "done": true,
        }));
    }
    let winding_down = limit > 0 && exchanges + 1 >= limit;

    let (reply, source) = llm
        .reply(
            &name,
            &turns,
            hypothesis.as_ref().map(|(l, c)| (l.as_str(), *c)),
            winding_down,
        )
        .await;

    let mut a = state.lock().unwrap();
    if let Some(session) = a.session.as_mut() {
        session.turns.push((VOICE.to_string(), reply.clone()));
    }
    Ok(json!({
        "text": reply, "source": source, "exchanges": exchanges, "done": winding_down,
    }))
}

/// Nothing was written, so nothing is lost.
#[tauri::command]
pub fn discard(state: State<'_, Shared>) -> Value {
    state.lock().unwrap().session = None;
    json!({ "cleared": true })
}

#[tauri::command]
pub async fn save(app: AppHandle, state: State<'_, Shared>) -> Result<Value, ()> {
    let (llm, user, day_start_hour, turns, hypothesis) = {
        let a = state.lock().unwrap();
        let Some(session) = a.session.as_ref() else {
            return Ok(err("nothing to save"));
        };
        // the whole-conversation reading if the camera saw anything, else the
        // one it had at hello, which is all there was
        let calibrated = session.hypothesis.as_ref().is_some_and(|h| h.calibrated);
        let hypothesis = vision::reaction(&session.readings, calibrated)
            .or_else(|| session.hypothesis.clone())
            .unwrap_or_default();
        (
            a.llm(),
            a.settings.name().to_lowercase(),
            a.settings.day_start_hour(),
            session.turns.clone(),
            hypothesis,
        )
    };

    let spoke = turns.iter().any(|(who, _)| who != VOICE);
    // both of these are slow and neither needs the other's answer, so they wait
    // together rather than one behind the other
    let (extracted, song) = if spoke {
        let (a, b) = tokio::join!(llm.extract(&turns), llm.suggest_song(&turns));
        (a, b)
    } else {
        (llm.extract(&turns).await, None)
    };
    let (data, source) = extracted;

    // the person said nothing: the entry records the hypothesis and says so.
    // self_reported is left out entirely rather than backfilled from the guess.
    let self_reported = (spoke && !data.self_reported.is_empty())
        .then(|| journal::normalize(&data.self_reported));

    let mut entry = NewEntry::new(user, Local::now().into());
    entry.turns = turns;
    entry.hypothesis = hypothesis;
    entry.self_reported = self_reported.clone();
    entry.themes = data.themes.clone();
    entry.summary = data.summary.clone();
    entry.song = song.clone();
    entry.day_start_hour = day_start_hour;

    let root = paths::journal_root(&app);
    let path = match journal::write(&entry, &root) {
        Ok(p) => p,
        Err(e) => return Ok(err(format!("could not write the entry: {e}"))),
    };
    state.lock().unwrap().session = None;

    Ok(json!({
        "path": path.strip_prefix(&root).unwrap_or(&path).to_string_lossy().replace('\\', "/"),
        "confirmed": self_reported.is_some(),
        "self_reported": self_reported,
        "themes": data.themes,
        "summary": data.summary,
        "song": song,
        "source": source,
    }))
}

// --- the camera ---------------------------------------------------------------

/// One frame in, one hypothesis out.
///
/// The frame arrives as raw JPEG bytes over the IPC boundary rather than as a
/// JSON array of numbers, which at several frames a second is the difference
/// between a preview and a slideshow. Nothing is written to disk at either end:
/// the bytes live in one call and are dropped.
#[tauri::command]
pub fn analyze(app: AppHandle, state: State<'_, Shared>, request: tauri::ipc::Request<'_>) -> Value {
    let tauri::ipc::InvokeBody::Raw(jpeg) = request.body() else {
        return err("expected a frame");
    };
    let Ok(frame) = image::load_from_memory(jpeg) else {
        return err("bad frame");
    };
    let frame = frame.to_rgb8();

    let mut a = state.lock().unwrap();
    if a.detector.is_none() {
        match Detector::new() {
            Ok(d) => a.detector = Some(d),
            Err(e) => return err(format!("could not load the models: {e}")),
        }
    }
    let detector = a.detector.as_mut().unwrap();
    let Ok(faces) = detector.faces(&frame) else {
        return err("could not read the frame");
    };
    let Some(face) = detect::subject(&faces).copied() else {
        return json!({ "box": Value::Null, "faces": 0 });
    };
    let Some(cut) = detect::crop(&frame, &face) else {
        return json!({ "box": Value::Null, "faces": faces.len() });
    };
    let Ok(logits) = a.detector.as_mut().unwrap().emotion(&cut) else {
        return err("could not read the frame");
    };

    let raw = vision::softmax(&logits);
    a.probs = a
        .probs
        .iter()
        .zip(&raw)
        .map(|(p, r)| vision::EMA * p + (1.0 - vision::EMA) * r)
        .collect();

    let mut remaining = 0usize;
    if let Some(collecting) = a.collecting.as_mut() {
        collecting.push(raw.clone());
        remaining = vision::BASELINE_FRAMES.saturating_sub(collecting.len());
        if remaining == 0 {
            let frames = a.collecting.take().unwrap();
            let mean: Vec<f32> = (0..vision::LABELS.len())
                .map(|i| frames.iter().map(|f| f[i]).sum::<f32>() / frames.len() as f32)
                .collect();
            save_baseline(&app, &mean);
            a.baseline = Some(mean);
        }
    }

    let shown = vision::calibrate(&a.probs, a.baseline.as_deref());
    let top = vision::argmax(&shown);
    let label = vision::LABELS[top];
    let confidence = shown[top] as f64;
    a.last = Some((label.to_string(), confidence));

    // while you are talking, keep watching. One reading at hello is a thin
    // record of a conversation; the shape of it over five minutes is not.
    if let Some(session) = a.session.as_mut() {
        if session.readings.len() < MAX_READINGS {
            session.readings.push(shown.clone());
        }
    }

    let sure = confidence >= a.settings.floor();
    let round4 = |v: f32| (v as f64 * 10000.0).round() / 10000.0;
    json!({
        // a sentence, not a number: nobody journalling wants to read "sad 0.42",
        // and a number invites belief the model has not earned
        "phrase": if sure { llm::looks_like(label) } else { "" },
        "box": [face.x, face.y, face.w, face.h],
        "frame": [frame.width(), frame.height()],
        "faces": faces.len(),
        "probs": vision::LABELS.iter().zip(&shown).map(|(k, v)| (*k, round4(*v))).collect::<BTreeMap<_, _>>(),
        "raw": vision::LABELS.iter().zip(&raw).map(|(k, v)| (*k, round4(*v))).collect::<BTreeMap<_, _>>(),
        "label": label,
        "confidence": round4(shown[top]),
        "sure": sure,
        "calibrated": a.baseline.is_some(),
        "remaining": remaining,
    })
}

/// Audio in, text out, and the samples are dropped.
///
/// The body is four bytes of sample rate followed by the raw f32 samples the
/// page took off the audio graph. Nothing is written to disk at either end.
#[tauri::command]
pub fn listen(state: State<'_, Shared>, request: tauri::ipc::Request<'_>) -> Value {
    let tauri::ipc::InvokeBody::Raw(body) = request.body() else {
        return err("expected audio");
    };
    if body.len() < 4 {
        return json!({ "text": "" });
    }
    let rate = u32::from_le_bytes([body[0], body[1], body[2], body[3]]);
    let samples: Vec<f32> = body[4..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let a = state.lock().unwrap();
    let Speech::Ready(speech) = &a.speech else {
        return a.speech.status();
    };
    let pcm = stt::resample(&samples, rate);
    json!({ "text": speech.transcribe(&pcm) })
}

fn save_baseline(app: &AppHandle, baseline: &[f32]) {
    let path = paths::calibration_path(app);
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let body = json!({ "baseline": baseline, "labels": vision::LABELS });
    let _ = std::fs::write(path, serde_json::to_vec_pretty(&body).unwrap_or_default());
}

// --- calibration ------------------------------------------------------------

#[tauri::command]
pub fn start_baseline(state: State<'_, Shared>) -> Value {
    state.lock().unwrap().collecting = Some(Vec::new());
    json!({ "ok": true, "remaining": vision::BASELINE_FRAMES })
}

#[tauri::command]
pub fn clear_baseline(app: AppHandle, state: State<'_, Shared>) -> Value {
    let mut a = state.lock().unwrap();
    a.baseline = None;
    a.collecting = None;
    let _ = std::fs::remove_file(paths::calibration_path(&app));
    json!({ "ok": true })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_day_is_split_the_way_the_greeting_says_it_is() {
        // the boundaries the Python used, so a greeting never says "morning" at 5pm
        assert_eq!(part_of_day_at(0), "morning");
        assert_eq!(part_of_day_at(11), "morning");
        assert_eq!(part_of_day_at(12), "afternoon");
        assert_eq!(part_of_day_at(16), "afternoon");
        assert_eq!(part_of_day_at(17), "evening");
        assert_eq!(part_of_day_at(23), "evening");
    }

    fn part_of_day_at(h: u32) -> &'static str {
        match h {
            h if h < 12 => "morning",
            h if h < 17 => "afternoon",
            _ => "evening",
        }
    }

    #[test]
    fn an_error_is_shaped_the_way_the_frontend_checks_for_it() {
        let e = err("nothing to save");
        assert_eq!(e["error"], json!("nothing to save"));
    }

    #[test]
    fn only_known_settings_may_be_written() {
        // the filter that set_config applies, checked without needing a State
        let updates: BTreeMap<String, String> = BTreeMap::from([
            ("HYPERJOURNAL_NAME".to_string(), "Yashwanth".to_string()),
            ("NVIDIA_API_KEY".to_string(), "nvapi-x".to_string()),
            ("PATH".to_string(), "/evil".to_string()),
            ("DAY_START_HOUR".to_string(), "6".to_string()),
        ]);
        let allowed: Vec<String> = updates
            .into_iter()
            .filter(|(k, _)| {
                k == settings::API_KEY || settings::FIELDS.iter().any(|(f, _)| f == k)
            })
            .map(|(k, _)| k)
            .collect();
        assert_eq!(
            allowed,
            vec!["DAY_START_HOUR", "HYPERJOURNAL_NAME", "NVIDIA_API_KEY"]
        );
        assert!(!allowed.contains(&"PATH".to_string()), "arbitrary keys must not be writable");
    }
}
