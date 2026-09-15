//! Phase 3: speech in. Audio arrives, text leaves, the buffer is dropped.
//!
//! Nothing is written to disk. The transcript goes into the journal because that
//! is the point of the journal; the recording does not, because it is not.
//!
//! The Python fed `faster-whisper` a container and let ffmpeg decode it. Here the
//! page captures Float32 samples directly through the Web Audio API and sends
//! them at 16 kHz mono, which is what whisper.cpp wants, so there is no codec on
//! this side at all.

use std::io::Write;
use std::path::{Path, PathBuf};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub const MODEL_FILE: &str = "ggml-base.en.bin";
pub const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
/// Roughly 148 MB. Used only to show a sensible percentage before the server
/// tells us the real length.
pub const MODEL_BYTES: u64 = 147_951_465;

pub const SAMPLE_RATE: u32 = 16_000;

/// Shorter than this is a cough or a mis-click.
const MIN_SECONDS: f32 = 0.6;
/// The model's own confidence in what it heard.
const MIN_LOGPROB: f32 = -1.0;
/// Above this, whisper itself believes the segment was not speech.
const MAX_NO_SPEECH: f32 = 0.6;

/// Whisper does not return nothing for silence -- it returns its training data.
///
/// These are the artifacts it reaches for, and every one of them would otherwise
/// become a sentence you never said, in a journal you trust.
const HALLUCINATIONS: [&str; 18] = [
    "thank you",
    "thanks for watching",
    "thank you for watching",
    "bye",
    "you",
    "thanks",
    "please subscribe",
    "subtitles by the amara.org community",
    "subs by www.zeoranger.co.uk",
    "www.mooji.org",
    "amara.org",
    "okay",
    "transcription by castingwords",
    "i'm sorry",
    ".",
    "...",
    "",
    " ",
];

pub fn is_hallucination(text: &str) -> bool {
    let bare = text.trim().to_lowercase();
    let bare = bare.trim_matches(['.', '!', '?', ',']).trim();
    HALLUCINATIONS.contains(&bare)
}

pub struct Stt {
    ctx: WhisperContext,
}

impl Stt {
    pub fn load(model: &Path) -> Result<Self, String> {
        let path = model.to_string_lossy().to_string();
        WhisperContext::new_with_params(&path, WhisperContextParameters::default())
            .map(|ctx| Self { ctx })
            .map_err(|e| format!("could not load the speech model: {e}"))
    }

    /// 16 kHz mono samples to text, or "" if it was silence.
    ///
    /// Returns "" rather than an error: a session where the microphone heard
    /// nothing is a real session that gets recorded as unconfirmed, not a
    /// failure.
    pub fn transcribe(&self, pcm: &[f32]) -> String {
        if (pcm.len() as f32 / SAMPLE_RATE as f32) < MIN_SECONDS {
            return String::new();
        }
        let Ok(mut state) = self.ctx.create_state() else {
            return String::new();
        };

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) as i32);
        params.set_language(Some("en"));
        // stops one hallucination seeding the next
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        if state.full(params, pcm).is_err() {
            return String::new();
        }

        let mut kept: Vec<String> = Vec::new();
        for segment in state.as_iter() {
            // whisper's own judgement that this was not speech at all
            if segment.no_speech_probability() > MAX_NO_SPEECH {
                continue;
            }
            // the mean log probability across the segment's tokens, which is what
            // faster-whisper called avg_logprob
            let probs: Vec<f32> = (0..segment.n_tokens())
                .filter_map(|i| segment.get_token(i))
                .map(|t| t.token_probability().max(1e-9).ln())
                .collect();
            if probs.is_empty() {
                continue;
            }
            let avg_logprob = probs.iter().sum::<f32>() / probs.len() as f32;
            if avg_logprob <= MIN_LOGPROB {
                continue;
            }
            if let Ok(text) = segment.to_str_lossy() {
                let text = text.trim();
                if !text.is_empty() {
                    kept.push(text.to_string());
                }
            }
        }

        let text = kept.join(" ").trim().to_string();
        if is_hallucination(&text) {
            return String::new();
        }
        text
    }
}

/// Fetch the weights, reporting progress as a fraction between 0 and 1.
///
/// Downloads beside the target and renames, so an interrupted download cannot
/// leave a half a model behind that then fails to load forever.
pub async fn download_model(
    target: &Path,
    mut progress: impl FnMut(f64),
) -> Result<PathBuf, String> {
    if target.is_file() {
        return Ok(target.to_path_buf());
    }
    if let Some(dir) = target.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{e}"))?;
    }

    let response = reqwest::get(MODEL_URL)
        .await
        .map_err(|e| format!("could not reach the speech model: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("could not fetch the speech model: HTTP {}", response.status()));
    }
    let total = response.content_length().unwrap_or(MODEL_BYTES);

    let tmp = target.with_extension("part");
    let mut file = std::fs::File::create(&tmp).map_err(|e| format!("{e}"))?;
    let mut written: u64 = 0;
    let mut stream = response;

    while let Some(chunk) = stream
        .chunk()
        .await
        .map_err(|e| format!("the download stopped: {e}"))?
    {
        file.write_all(&chunk).map_err(|e| format!("{e}"))?;
        written += chunk.len() as u64;
        progress((written as f64 / total as f64).min(1.0));
    }
    file.flush().map_err(|e| format!("{e}"))?;
    drop(file);

    std::fs::rename(&tmp, target).map_err(|e| format!("{e}"))?;
    Ok(target.to_path_buf())
}

/// Downsample interleaved mono samples from the browser's rate to 16 kHz.
///
/// Linear interpolation. The page could resample itself, but the browser's
/// `AudioContext` rate varies by device and doing it here means one
/// implementation rather than one per platform.
pub fn resample(input: &[f32], from: u32) -> Vec<f32> {
    if from == SAMPLE_RATE || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / SAMPLE_RATE as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let left = pos.floor() as usize;
            let right = (left + 1).min(input.len() - 1);
            let t = (pos - left as f64) as f32;
            input[left] * (1.0 - t) + input[right] * t
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_known_artifacts_are_caught_whatever_the_punctuation() {
        assert!(is_hallucination("Thank you."));
        assert!(is_hallucination("  thanks for watching!  "));
        assert!(is_hallucination("..."));
        assert!(is_hallucination(""));
        assert!(is_hallucination("Okay"));
        // and a real sentence is not
        assert!(!is_hallucination("Today was hard."));
        assert!(!is_hallucination("thank you for listening to me tonight"));
    }

    #[test]
    fn resampling_lands_on_the_rate_whisper_wants() {
        let input: Vec<f32> = (0..48_000).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = resample(&input, 48_000);
        // one second in, one second out
        assert!((out.len() as i64 - 16_000).abs() <= 1, "{}", out.len());
        assert!(out.iter().all(|v| v.is_finite()));

        // already at the right rate, nothing is touched
        let same = resample(&input, SAMPLE_RATE);
        assert_eq!(same.len(), input.len());

        assert!(resample(&[], 44_100).is_empty());
    }

    #[test]
    fn resampling_preserves_the_shape_of_a_tone() {
        // a 100 Hz tone at 48 kHz, resampled, must still be a 100 Hz tone
        let from = 48_000u32;
        let input: Vec<f32> = (0..from)
            .map(|i| (2.0 * std::f32::consts::PI * 100.0 * i as f32 / from as f32).sin())
            .collect();
        let out = resample(&input, from);
        // count zero crossings: 100 Hz over one second is ~200 of them
        let crossings = out.windows(2).filter(|w| w[0].signum() != w[1].signum()).count();
        assert!((199..=201).contains(&crossings), "got {crossings} crossings");
    }
}

/// Where the speech model has got to.
///
/// It is 148 MB, so it is fetched in the background at startup rather than the
/// first time somebody holds the button down, and the microphone says so until
/// it is ready.
pub enum Speech {
    Fetching(f64),
    Ready(Box<Stt>),
    Failed(String),
}

impl Speech {
    /// What to tell the page when it asks to transcribe and this is not ready.
    pub fn status(&self) -> serde_json::Value {
        match self {
            Speech::Fetching(progress) => {
                serde_json::json!({ "text": "", "status": "fetching", "progress": progress })
            }
            Speech::Failed(e) => serde_json::json!({ "text": "", "error": e }),
            Speech::Ready(_) => serde_json::json!({ "text": "" }),
        }
    }
}
