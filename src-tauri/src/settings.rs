//! Everything a person might want to change. Edited in the app, stored as JSON
//! in the platform config directory.
//!
//! The Python kept these in a `.env` beside the script. That cannot work once
//! the app is installed to Program Files, and means nothing at all on Android,
//! so the file moved. The keys did not: they are still the `.env` names, and an
//! existing `.env` is imported once on first run so nobody has to retype an API
//! key.
//!
//! Values are stored as strings and cast on read, exactly as the `.env` was.
//! A missing or unparseable value reads back as `None`, so callers decide what
//! an unset setting means rather than inheriting a guess.

use serde::Serialize;
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Below this the camera's guess is not worth saying out loud.
pub const CONFIDENCE_FLOOR: f64 = 0.45;

pub const NAME: &str = "HYPERJOURNAL_NAME";
pub const MODEL: &str = "NVIDIA_MODEL";
pub const FLOOR: &str = "CONFIDENCE_FLOOR";
pub const DAY_START: &str = "DAY_START_HOUR";
pub const MAX_EXCHANGES: &str = "MAX_EXCHANGES";
pub const API_KEY: &str = "NVIDIA_API_KEY";

/// The settable keys and what they are, in the order the settings page shows
/// them. The API key is deliberately absent: it is written in and never read
/// back out.
pub const FIELDS: [(&str, &str); 5] = [
    (NAME, "what it calls you out loud"),
    (MODEL, "which NIM model writes the sentences"),
    (FLOOR, "below this the camera's guess is not said out loud"),
    (DAY_START, "hour the journal day rolls over"),
    (MAX_EXCHANGES, "replies before it winds down; 0 or blank means no limit"),
];

#[derive(Debug, Default, Clone)]
pub struct Settings {
    path: PathBuf,
    values: BTreeMap<String, String>,
}

/// Three lines of `.env` parsing, so a dotenv crate stays uninstalled.
fn parse_env(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| {
            (
                k.trim().to_string(),
                v.trim().trim_matches('"').trim_matches('\'').to_string(),
            )
        })
        .collect()
}

impl Settings {
    /// Load from `path`. When there is nothing there yet, import `legacy_env`
    /// if it exists, so the Python's `.env` carries over on first run.
    pub fn load(path: &Path, legacy_env: Option<&Path>) -> Self {
        let values = fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str::<BTreeMap<String, String>>(&t).ok())
            .or_else(|| {
                let env = legacy_env?;
                let imported = parse_env(&fs::read_to_string(env).ok()?);
                (!imported.is_empty()).then_some(imported)
            })
            .unwrap_or_default();
        Self { path: path.to_path_buf(), values }
    }

    /// Apply changes and write them out. A blank value clears the setting, so
    /// emptying a box in the app means "unset", not "set to empty string".
    pub fn save(&mut self, updates: BTreeMap<String, String>) -> io::Result<()> {
        for (key, value) in updates {
            if value.trim().is_empty() {
                self.values.remove(&key);
            } else {
                self.values.insert(key, value.trim().to_string());
            }
        }
        if let Some(dir) = self.path.parent() {
            fs::create_dir_all(dir)?;
        }
        let text = serde_json::to_string_pretty(&self.values)?;
        let tmp = self.path.with_extension("json.tmp");
        fs::write(&tmp, text)?;
        fs::rename(&tmp, &self.path)
    }

    fn raw(&self, key: &str) -> Option<&str> {
        // an environment variable still wins, so a shell can override without
        // editing the file
        let _ = key;
        self.values.get(key).map(String::as_str).filter(|v| !v.is_empty())
    }

    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| self.raw(key).map(str::to_string))
    }

    /// What it calls you out loud. There is a fallback here because a greeting
    /// has to address somebody.
    pub fn name(&self) -> String {
        self.get(NAME).unwrap_or_else(|| "there".into())
    }

    /// The chosen model, or `None`. There is deliberately no default: picking
    /// one silently is how you end up talking to a model nobody chose.
    pub fn model(&self) -> Option<String> {
        self.get(MODEL)
    }

    pub fn api_key(&self) -> Option<String> {
        self.get(API_KEY)
    }

    /// The confidence floor the person chose, or the shipped one.
    pub fn floor(&self) -> f64 {
        self.get(FLOOR).and_then(|v| v.parse().ok()).unwrap_or(CONFIDENCE_FLOOR)
    }

    pub fn day_start_hour(&self) -> i64 {
        self.get(DAY_START)
            .and_then(|v| v.parse().ok())
            .unwrap_or(crate::journal::DAY_START_HOUR)
    }

    /// Replies before it winds down. 0 or unset means talk for as long as you like.
    pub fn max_exchanges(&self) -> u32 {
        self.get(MAX_EXCHANGES).and_then(|v| v.parse().ok()).unwrap_or(0)
    }

    /// What the settings page shows. Cast, so the page gets a number where it
    /// expects a number and `null` where nothing has been chosen. Never
    /// includes the API key.
    pub fn as_json(&self) -> serde_json::Value {
        use serde_json::Value;
        let text = |k: &str| self.get(k).map_or(Value::Null, Value::String);
        let number = |k: &str| {
            self.get(k)
                .and_then(|v| serde_json::Number::from_f64(v.parse::<f64>().ok()?))
                .map_or(Value::Null, Value::Number)
        };
        serde_json::json!({
            NAME: text(NAME),
            MODEL: text(MODEL),
            FLOOR: number(FLOOR),
            DAY_START: number(DAY_START),
            MAX_EXCHANGES: number(MAX_EXCHANGES),
        })
    }
}

/// What the settings page is told. `key_set` rather than the key itself: it
/// goes in and never comes back out.
#[derive(Serialize)]
pub struct Config {
    pub settings: serde_json::Value,
    pub key_set: bool,
    pub fields: BTreeMap<String, String>,
    pub voice: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("hj-settings-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn unset_values_stay_unset_rather_than_guessing() {
        let dir = tmp("empty");
        let s = Settings::load(&dir.join("settings.json"), None);
        assert_eq!(s.model(), None, "no model may be picked silently");
        assert_eq!(s.api_key(), None);
        assert_eq!(s.name(), "there");
        // the shipped defaults stand in only where a caller cannot proceed without one
        assert_eq!(s.floor(), CONFIDENCE_FLOOR);
        assert_eq!(s.day_start_hour(), 4);
        assert_eq!(s.max_exchanges(), 0, "0 means talk for as long as you like");
    }

    #[test]
    fn a_blank_value_clears_a_setting() {
        let dir = tmp("clear");
        let path = dir.join("settings.json");
        let mut s = Settings::load(&path, None);
        s.save(BTreeMap::from([(NAME.into(), "Yashwanth".into())])).unwrap();
        assert_eq!(s.name(), "Yashwanth");

        s.save(BTreeMap::from([(NAME.into(), "   ".into())])).unwrap();
        assert_eq!(s.name(), "there");
        assert!(!fs::read_to_string(&path).unwrap().contains("Yashwanth"));
    }

    #[test]
    fn settings_survive_a_reload() {
        let dir = tmp("reload");
        let path = dir.join("settings.json");
        let mut s = Settings::load(&path, None);
        s.save(BTreeMap::from([
            (NAME.into(), "Yashwanth".into()),
            (FLOOR.into(), "0.9".into()),
            (DAY_START.into(), "6".into()),
            (API_KEY.into(), "nvapi-secret".into()),
        ]))
        .unwrap();

        let again = Settings::load(&path, None);
        assert_eq!(again.name(), "Yashwanth");
        assert_eq!(again.floor(), 0.9);
        assert_eq!(again.day_start_hour(), 6);
        assert_eq!(again.api_key().as_deref(), Some("nvapi-secret"));
    }

    #[test]
    fn the_api_key_never_reaches_the_page() {
        let dir = tmp("secret");
        let path = dir.join("settings.json");
        let mut s = Settings::load(&path, None);
        s.save(BTreeMap::from([(API_KEY.into(), "nvapi-secret".into())])).unwrap();

        let shown = serde_json::to_string(&s.as_json()).unwrap();
        assert!(!shown.contains("nvapi-secret"), "{shown}");
        assert!(!shown.contains(API_KEY), "{shown}");
    }

    #[test]
    fn an_existing_dotenv_is_imported_once() {
        let dir = tmp("migrate");
        let env = dir.join(".env");
        fs::write(
            &env,
            "# Written by Hyperjournal. Change these in the app.\n\
             NVIDIA_API_KEY=nvapi-from-env\n\
             HYPERJOURNAL_NAME=Yashwanth\n\
             CONFIDENCE_FLOOR=0.45\n\
             \n\
             MAX_EXCHANGES=\n",
        )
        .unwrap();

        let s = Settings::load(&dir.join("settings.json"), Some(&env));
        assert_eq!(s.api_key().as_deref(), Some("nvapi-from-env"));
        assert_eq!(s.name(), "Yashwanth");
        assert_eq!(s.floor(), 0.45);
        assert_eq!(s.max_exchanges(), 0, "an empty value in .env is not a value");
    }

    #[test]
    fn the_page_gets_numbers_as_numbers_and_nulls_where_nothing_is_chosen() {
        let dir = tmp("json");
        let path = dir.join("settings.json");
        let mut s = Settings::load(&path, None);
        s.save(BTreeMap::from([
            (NAME.into(), "Yashwanth".into()),
            (FLOOR.into(), "0.45".into()),
        ]))
        .unwrap();

        let j = s.as_json();
        assert_eq!(j[NAME], serde_json::json!("Yashwanth"));
        assert_eq!(j[FLOOR], serde_json::json!(0.45));
        assert!(j[MODEL].is_null(), "an unchosen model must read as null, not empty string");
        assert!(j[DAY_START].is_null());
    }

    #[test]
    fn a_junk_value_reads_as_unset_rather_than_zero() {
        let dir = tmp("junk");
        let path = dir.join("settings.json");
        let mut s = Settings::load(&path, None);
        s.save(BTreeMap::from([(FLOOR.into(), "not a number".into())])).unwrap();
        assert_eq!(s.floor(), CONFIDENCE_FLOOR, "junk must fall back, not become 0.0");
    }
}
