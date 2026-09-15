//! Hyperjournal: a local journal that listens, watches, and remembers.

pub mod commands;
pub mod detect;
pub mod journal;
pub mod llm;
pub mod paths;
pub mod settings;
pub mod speak;
pub mod stt;
pub mod vision;

use commands::App;
use settings::Settings;
use std::sync::Mutex;
use tauri::Manager;

/// The resting-face baseline from a previous run, if there is one.
///
/// A baseline that cannot be read is not an error worth stopping for: the app
/// simply reads faces with no allowance until the person calibrates again.
fn load_baseline(path: &std::path::Path) -> Option<Vec<f32>> {
    let text = std::fs::read_to_string(path).ok()?;
    let data: serde_json::Value = serde_json::from_str(&text).ok()?;
    let values: Vec<f32> = data["baseline"]
        .as_array()?
        .iter()
        .filter_map(|v| v.as_f64().map(|f| f as f32))
        .collect();
    (values.len() == vision::LABELS.len()).then_some(values)
}

/// Record how the speech model is getting on.
///
/// The guard is bound rather than matched inline: a `MutexGuard` borrowed from
/// the `State` temporary would otherwise be dropped after the thing it borrows.
fn set_speech(handle: &tauri::AppHandle, speech: stt::Speech) {
    let state = handle.state::<Mutex<App>>();
    let locked = state.lock();
    if let Ok(mut app) = locked {
        app.speech = speech;
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }

            let handle = app.handle();
            let settings = Settings::load(
                &paths::settings_path(handle),
                paths::legacy_env(handle).as_deref(),
            );
            // the journal has to exist before anything tries to list it
            let _ = std::fs::create_dir_all(paths::journal_root(handle));
            let _ = std::fs::create_dir_all(paths::data_dir(handle));
            let baseline = load_baseline(&paths::calibration_path(handle));

            app.manage(Mutex::new(App::new(settings, baseline)));
            app.manage(speak::Voice::default());

            // The speech model is 148 MB, so it is fetched once in the
            // background rather than the first time somebody holds the button
            // and then waits two minutes wondering what broke.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let target = paths::whisper_model_path(&handle, stt::MODEL_FILE);
                let progress = {
                    let handle = handle.clone();
                    move |fraction: f64| set_speech(&handle, stt::Speech::Fetching(fraction))
                };
                let ready = match stt::download_model(&target, progress).await {
                    Ok(path) => tauri::async_runtime::spawn_blocking(move || stt::Stt::load(&path))
                        .await
                        .unwrap_or_else(|e| Err(format!("{e}"))),
                    Err(e) => Err(e),
                };
                set_speech(
                    &handle,
                    match ready {
                        Ok(engine) => stt::Speech::Ready(Box::new(engine)),
                        Err(e) => stt::Speech::Failed(e),
                    },
                );
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::config,
            commands::set_config,
            commands::models,
            commands::entries,
            commands::delete_entry,
            commands::delete_all,
            commands::greet,
            commands::say,
            commands::discard,
            commands::save,
            commands::start_baseline,
            commands::clear_baseline,
            commands::analyze,
            commands::listen,
            speak::speak,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
