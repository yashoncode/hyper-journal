//! Hyperjournal's voice, for platforms whose webview has none.
//!
//! On the desktop the page uses `speechSynthesis`, which is already there and
//! already works. Android's System WebView does not implement it, so the page
//! falls back to this, which reaches `android.speech.tts.TextToSpeech` through
//! the `tts` crate.
//!
//! The engine is kept alive between utterances. Building one per sentence means
//! paying Android's asynchronous engine initialisation every time, and losing
//! the first words while it completes.

use serde_json::{json, Value};
use std::sync::Mutex;
use tauri::State;

#[derive(Default)]
pub struct Voice(pub Mutex<Option<tts::Tts>>);

#[tauri::command]
pub fn speak(voice: State<'_, Voice>, text: String) -> Value {
    let mut held = voice.0.lock().unwrap();
    if held.is_none() {
        match tts::Tts::default() {
            Ok(engine) => *held = Some(engine),
            Err(e) => return json!({ "error": format!("no voice on this device: {e}") }),
        }
    }
    // interrupt anything still being said: the previous sentence is stale the
    // moment there is a new one
    match held.as_mut().unwrap().speak(text, true) {
        Ok(_) => json!({ "ok": true }),
        Err(e) => json!({ "error": format!("{e}") }),
    }
}
