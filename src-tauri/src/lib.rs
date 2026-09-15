//! Hyperjournal: a local journal that listens, watches, and remembers.

pub mod commands;
pub mod journal;
pub mod llm;
pub mod paths;
pub mod settings;
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
