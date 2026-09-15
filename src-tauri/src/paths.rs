//! Where things live, per platform.
//!
//! The Python kept everything beside the script, which stops working the moment
//! the app is installed somewhere read-only and means nothing on Android. Three
//! directories replace it:
//!
//! - the journal itself, in Documents, where a person can find it without this
//!   app and where it survives an uninstall;
//! - settings, in the platform config directory;
//! - the Whisper model and the calibration baseline, in the app data directory,
//!   because both are caches that can be rebuilt.

use std::path::PathBuf;
use tauri::{AppHandle, Manager};

/// The journal. `Documents/Hyperjournal`, holding `YYYY/MM/*.md` and `trashed/`.
///
/// Falls back to the app data directory only if the platform cannot name a
/// documents directory at all, which would otherwise leave nowhere to write.
pub fn journal_root(app: &AppHandle) -> PathBuf {
    match app.path().document_dir() {
        Ok(dir) => shared(dir).join("Hyperjournal"),
        Err(_) => data_dir(app).join("journal"),
    }
}

/// On Android, the *shared* Documents folder rather than this app's own.
///
/// Tauri reports `getExternalFilesDir(DIRECTORY_DOCUMENTS)`, which is
/// `.../Android/data/<package>/files/Documents`: private, invisible to most
/// file managers since Android 11, and deleted when the app is uninstalled. A
/// journal that is erased by uninstalling the journal is not a journal, so the
/// path is walked back to the storage root and the shared folder used instead.
///
/// This needs All-files access, which `MainActivity` asks for. Without it the
/// writes fail and the app stays on the private path, which still works.
#[cfg(target_os = "android")]
fn shared(scoped: PathBuf) -> PathBuf {
    let text = scoped.to_string_lossy().to_string();
    match text.split_once("/Android/data/") {
        Some((storage_root, _)) if !storage_root.is_empty() => {
            let candidate = PathBuf::from(storage_root).join("Documents");
            // only prefer it if it is actually reachable
            if std::fs::create_dir_all(&candidate).is_ok() {
                candidate
            } else {
                scoped
            }
        }
        _ => scoped,
    }
}

#[cfg(not(target_os = "android"))]
fn shared(dir: PathBuf) -> PathBuf {
    dir
}

/// App data: the Whisper model and the calibration baseline.
pub fn data_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| std::env::temp_dir().join("hyperjournal"))
}

pub fn settings_path(app: &AppHandle) -> PathBuf {
    app.path()
        .app_config_dir()
        .unwrap_or_else(|_| data_dir(app))
        .join("settings.json")
}

/// Where the camera's resting-face baseline is remembered.
pub fn calibration_path(app: &AppHandle) -> PathBuf {
    data_dir(app).join("calib.json")
}

/// The GGML weights whisper.cpp loads, downloaded on first use.
pub fn whisper_model_path(app: &AppHandle, file: &str) -> PathBuf {
    data_dir(app).join(file)
}

/// A `.env` left over from the Python, if one is still lying around.
///
/// Only consulted when there are no settings yet, so it runs once and then
/// never again. The candidates cover running from a dev build (where the
/// working directory is `src-tauri`) and running beside a checkout.
pub fn legacy_env(app: &AppHandle) -> Option<PathBuf> {
    let mut candidates = vec![PathBuf::from(".env"), PathBuf::from("../.env")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(".env"));
        }
    }
    if let Ok(home) = app.path().home_dir() {
        candidates.push(home.join("hyperjournal").join(".env"));
    }
    candidates.into_iter().find(|p| p.is_file())
}
