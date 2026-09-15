//! Phase 3: the output. One Markdown file per entry, readable in a year with
//! no software at all.
//!
//! This is the part that must outlive the camera, the models and the API, so it
//! has no dependency on any of them and is the best-tested thing here.

use chrono::{DateTime, Duration, FixedOffset, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// 01:30 on the 12th still belongs to the 11th.
pub const DAY_START_HOUR: i64 = 4;

/// Not a dot-directory: you should be able to find it.
pub const TRASH: &str = "trashed";

/// What the camera made of a face. Field order here is alphabetical on purpose:
/// PyYAML sorted its keys, and entries written by the Python must stay
/// indistinguishable from entries written by this.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub struct Hypothesis {
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub began: Option<String>,
    pub calibrated: bool,
    pub confidence: f64,
    pub distribution: BTreeMap<String, f64>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ended: Option<String>,
    pub label: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub samples: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct Song {
    pub artist: String,
    pub title: String,
    pub why: String,
}

/// The frontmatter. Alphabetical, as above.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Meta {
    pub confirmed: bool,
    pub hypothesis: Hypothesis,
    pub journal_date: String,
    /// What the person said about themselves. Absent entirely when they said
    /// nothing -- never backfilled from the hypothesis, because the absence is
    /// the honest record.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub self_reported: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub song: Option<Song>,
    pub summary: String,
    pub themes: Vec<String>,
    pub timestamp: String,
    pub user: String,
}

#[derive(Debug, Clone)]
pub struct Post {
    pub meta: Meta,
    pub content: String,
}

/// The day the person is still living, which is not the calendar date.
///
/// Deliberately not `when.date()`. Someone writing at 01:30 is finishing
/// yesterday, and an entry filed under tomorrow is a subtly wrong history.
pub fn journal_date(when: DateTime<FixedOffset>, day_start_hour: i64) -> NaiveDate {
    (when - Duration::hours(day_start_hour)).date_naive()
}

/// The verbatim transcript. Not a summary -- the summary has its own field.
pub fn render_body(turns: &[(String, String)]) -> String {
    turns
        .iter()
        .map(|(speaker, text)| format!("**{speaker}:** {text}").trim().to_string())
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// Everything one entry needs. A struct rather than nine positional arguments,
/// because `write(user, turns, hyp, None, vec![], "", None, ...)` at the call
/// site tells you nothing about which `None` is which.
#[derive(Debug, Clone)]
pub struct NewEntry {
    pub user: String,
    pub turns: Vec<(String, String)>,
    pub hypothesis: Hypothesis,
    pub self_reported: Option<String>,
    pub themes: Vec<String>,
    pub summary: String,
    pub song: Option<Song>,
    pub when: DateTime<FixedOffset>,
    pub day_start_hour: i64,
}

impl NewEntry {
    pub fn new(user: impl Into<String>, when: DateTime<FixedOffset>) -> Self {
        Self {
            user: user.into(),
            turns: Vec::new(),
            hypothesis: Hypothesis::default(),
            self_reported: None,
            themes: Vec::new(),
            summary: String::new(),
            song: None,
            when,
            day_start_hour: DAY_START_HOUR,
        }
    }
}

/// `journal_date` and `timestamp` are strings, and must stay strings.
///
/// This emitter follows YAML 1.2, where a bare `2026-09-11` is just text. Most
/// things that will ever read these files -- PyYAML included, which wrote the
/// existing entries -- follow YAML 1.1, where the same bare scalar resolves to
/// a date. Quoting the two fields that carry an entry's identity keeps them
/// unambiguous, and keeps new entries looking like the ones already on disk.
fn quote_dates(yaml: &str) -> String {
    yaml.lines()
        .map(|line| {
            match line.split_once(": ") {
                Some((key, value))
                    if matches!(key, "journal_date" | "timestamp")
                        && !value.starts_with('\'')
                        && !value.starts_with('"') =>
                {
                    format!("{key}: '{}'", value.replace('\'', "''"))
                }
                _ => line.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn serialize(post: &Meta, content: &str) -> Result<String, serde_yaml_ng::Error> {
    let yaml = quote_dates(&serde_yaml_ng::to_string(post)?);
    Ok(format!("---\n{yaml}---\n\n{content}\n"))
}

/// Split a file into its frontmatter and its body.
///
/// Tolerant of CRLF because a file that has been through a Windows editor is
/// still your journal entry.
fn deserialize(raw: &str) -> io::Result<Post> {
    let text = raw.strip_prefix('\u{feff}').unwrap_or(raw).replace("\r\n", "\n");
    let body = text
        .strip_prefix("---\n")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no frontmatter"))?;
    let (yaml, content) = body
        .split_once("\n---\n")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "unterminated frontmatter"))?;
    let meta: Meta = serde_yaml_ng::from_str(yaml)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(Post {
        meta,
        content: content.trim_start_matches('\n').trim_end().to_string(),
    })
}

/// Write one entry atomically. Returns the path.
pub fn write(entry: &NewEntry, root: &Path) -> io::Result<PathBuf> {
    let confirmed = entry.self_reported.is_some();
    let meta = Meta {
        confirmed,
        hypothesis: entry.hypothesis.clone(),
        journal_date: journal_date(entry.when, entry.day_start_hour).to_string(),
        self_reported: entry.self_reported.clone(),
        song: entry.song.clone(),
        summary: entry.summary.clone(),
        themes: entry.themes.clone(),
        timestamp: entry.when.format("%Y-%m-%dT%H:%M:%S%:z").to_string(),
        user: entry.user.clone(),
    };

    let stem = entry.when.format("%Y-%m-%dT%H%M").to_string();
    let dir = root
        .join(entry.when.format("%Y").to_string())
        .join(entry.when.format("%m").to_string());
    fs::create_dir_all(&dir)?;

    // filenames are minute-granular, so two sessions in one minute collide.
    // Rare, but an overwrite here is a destroyed memory, so suffix instead.
    let mut out = dir.join(format!("{stem}.md"));
    let mut n = 2;
    while out.exists() {
        out = dir.join(format!("{stem}-{n}.md"));
        n += 1;
    }

    let text = serialize(&meta, &render_body(&entry.turns))
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // temp file in the same directory, then rename: a crash mid-write cannot
    // leave half an entry behind
    let tmp = dir.join(format!(".{stem}.{}.tmp", std::process::id()));
    fs::write(&tmp, text.as_bytes())?;
    if let Err(e) = fs::rename(&tmp, &out) {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    Ok(out)
}

/// Move an entry out of the journal instead of unlinking it.
///
/// Deleting is a click. Regretting it is a week later. The file keeps its name
/// under `trashed/`, out of every listing, and is still there when you want it
/// back.
pub fn discard(path: &Path, root: &Path) -> io::Result<PathBuf> {
    let dir = root.join(TRASH);
    fs::create_dir_all(&dir)?;
    let name = path
        .file_name()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "no filename"))?;
    let stem = path.file_stem().unwrap_or(name).to_string_lossy().to_string();

    let mut dest = dir.join(name);
    let mut n = 2;
    while dest.exists() {
        dest = dir.join(format!("{stem}-{n}.md"));
        n += 1;
    }
    fs::rename(path, &dest)?;
    Ok(dest)
}

pub fn read(path: &Path) -> io::Result<Post> {
    deserialize(&fs::read_to_string(path)?)
}

/// Sort key. A `-2` collision suffix is later than the file it followed, but
/// sorts before it as plain text ('-' < '.'), so pull the suffix out first.
fn order(path: &Path) -> (String, u32) {
    let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
    match stem.rsplit_once('-') {
        Some((head, tail)) if head.contains('T') && !tail.is_empty() => match tail.parse::<u32>() {
            Ok(n) => (head.to_string(), n),
            Err(_) => (stem, 1),
        },
        _ => (stem, 1),
    }
}

fn subdirs(dir: &Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect(),
        Err(_) => Vec::new(),
    };
    out.sort();
    out
}

/// Every entry path, newest first. The filesystem is the index; there is no
/// second copy to drift.
///
/// `trashed/` sits one level up from `YYYY/MM/`, so it falls out of this walk
/// without needing to be excluded.
pub fn entry_paths(root: &Path) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for year in subdirs(root) {
        for month in subdirs(&year) {
            if let Ok(rd) = fs::read_dir(&month) {
                found.extend(
                    rd.flatten()
                        .map(|e| e.path())
                        .filter(|p| p.extension().is_some_and(|x| x == "md")),
                );
            }
        }
    }
    found.sort_by(|a, b| order(b).cmp(&order(a)));
    found
}

pub fn entries(root: &Path, limit: Option<usize>) -> Vec<(PathBuf, Post)> {
    let paths = entry_paths(root);
    let paths = match limit {
        Some(n) => &paths[..n.min(paths.len())],
        None => &paths[..],
    };
    paths
        .iter()
        .filter_map(|p| read(p).ok().map(|post| (p.clone(), post)))
        .collect()
}

/// One-liners from the last few entries, for greeting context.
pub fn summaries(root: &Path, n: usize) -> Vec<String> {
    entries(root, Some(n))
        .into_iter()
        .filter(|(_, post)| !post.meta.summary.trim().is_empty())
        .map(|(_, post)| format!("{}: {}", post.meta.journal_date, post.meta.summary.trim()))
        .collect()
}

/// A short lowercase phrase, so 'Pretty Tired!' and 'pretty tired' agree.
pub fn normalize(phrase: &str) -> String {
    let trimmed = phrase.trim().trim_matches(['.', '!', '?']).to_lowercase();
    trimmed.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(60).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ist() -> FixedOffset {
        FixedOffset::east_opt(5 * 3600 + 30 * 60).unwrap()
    }

    fn at(day: u32, hour: u32, min: u32) -> DateTime<FixedOffset> {
        ist().with_ymd_and_hms(2026, 9, day, hour, min, 22).unwrap()
    }

    fn hyp() -> Hypothesis {
        Hypothesis {
            label: "Sadness".into(),
            confidence: 0.42,
            calibrated: true,
            distribution: BTreeMap::from([("Sadness".into(), 0.42), ("Neutral".into(), 0.31)]),
            ..Default::default()
        }
    }

    fn turns() -> Vec<(String, String)> {
        vec![
            ("Hyperjournal".into(), "You look a little low — how are you doing?".into()),
            ("Yashwanth".into(), "Just tired.\nDidn't sleep — café till 2am. ☕".into()),
        ]
    }

    #[test]
    fn day_boundary_is_not_the_calendar_date() {
        // 03:59 and 04:01 are different days; 23:59 is its own
        assert_eq!(journal_date(at(12, 3, 59), DAY_START_HOUR), NaiveDate::from_ymd_opt(2026, 9, 11).unwrap());
        assert_eq!(journal_date(at(12, 4, 1), DAY_START_HOUR), NaiveDate::from_ymd_opt(2026, 9, 12).unwrap());
        assert_eq!(journal_date(at(11, 23, 59), DAY_START_HOUR), NaiveDate::from_ymd_opt(2026, 9, 11).unwrap());
        assert_eq!(journal_date(at(12, 1, 30), DAY_START_HOUR), NaiveDate::from_ymd_opt(2026, 9, 11).unwrap());
    }

    #[test]
    fn round_trip_keeps_unicode_and_multi_line_turns() {
        let dir = tempdir();
        let root = dir.as_path();
        let mut e = NewEntry::new("yashwanth", at(11, 8, 14));
        e.turns = turns();
        e.hypothesis = hyp();
        e.self_reported = Some("tired".into());
        e.themes = vec!["sleep".into(), "work-pressure".into()];
        e.summary = "Slept badly.".into();

        let p = write(&e, root).unwrap();
        assert_eq!(p.file_name().unwrap(), "2026-09-11T0814.md");
        assert_eq!(p.parent().unwrap(), root.join("2026").join("09"));

        let got = read(&p).unwrap();
        assert_eq!(got.meta.self_reported.as_deref(), Some("tired"));
        assert!(got.meta.confirmed);
        assert_eq!(got.meta.journal_date, "2026-09-11");
        assert_eq!(got.meta.hypothesis, hyp());
        assert_eq!(got.meta.themes, vec!["sleep", "work-pressure"]);
        assert!(got.content.contains('☕') && got.content.contains("café till 2am"));
        assert_eq!(got.content, render_body(&turns()));
        assert!(got.meta.timestamp.contains("+05:30"), "{}", got.meta.timestamp);
    }

    #[test]
    fn silence_records_no_self_report_at_all() {
        let dir = tempdir();
        let root = dir.as_path();
        let mut e = NewEntry::new("yashwanth", at(11, 21, 14));
        e.turns = vec![("Hyperjournal".into(), "Evening — how are you?".into())];
        e.hypothesis = hyp();

        let q = write(&e, root).unwrap();
        let silent = read(&q).unwrap();
        assert!(!silent.meta.confirmed);
        assert_eq!(silent.meta.self_reported, None);
        // the key must be absent from the file, not present and empty
        let raw = fs::read_to_string(&q).unwrap();
        assert!(!raw.contains("self_reported"), "{raw}");
    }

    #[test]
    fn same_minute_twice_must_not_overwrite() {
        let dir = tempdir();
        let root = dir.as_path();
        let mut first = NewEntry::new("yashwanth", at(11, 21, 14));
        first.turns = vec![("Hyperjournal".into(), "Evening — how are you?".into())];
        first.hypothesis = hyp();
        let q = write(&first, root).unwrap();

        let mut second = NewEntry::new("yashwanth", at(11, 21, 14));
        second.turns = vec![("Hyperjournal".into(), "again".into())];
        second.hypothesis = hyp();
        second.self_reported = Some("fine".into());
        let r = write(&second, root).unwrap();

        assert_eq!(r.file_name().unwrap(), "2026-09-11T2114-2.md");
        assert!(!read(&q).unwrap().meta.confirmed, "first entry was clobbered");
    }

    #[test]
    fn listing_is_newest_first_and_leaves_no_debris() {
        let dir = tempdir();
        let root = dir.as_path();

        let mut a = NewEntry::new("yashwanth", at(11, 8, 14));
        a.turns = turns();
        a.hypothesis = hyp();
        a.self_reported = Some("tired".into());
        a.summary = "Slept badly.".into();
        let p = write(&a, root).unwrap();

        let mut b = NewEntry::new("yashwanth", at(11, 21, 14));
        b.turns = vec![("Hyperjournal".into(), "evening".into())];
        b.hypothesis = hyp();
        write(&b, root).unwrap();

        let mut c = NewEntry::new("yashwanth", at(11, 21, 14));
        c.turns = vec![("Hyperjournal".into(), "again".into())];
        c.hypothesis = hyp();
        c.self_reported = Some("fine".into());
        let r = write(&c, root).unwrap();

        assert_eq!(entries(root, None).len(), 3);
        assert_eq!(entries(root, None)[0].0, r, "newest first");
        assert_eq!(summaries(root, 3), vec!["2026-09-11: Slept badly."]);

        let debris: Vec<_> = entry_paths(root)
            .iter()
            .filter(|p| p.to_string_lossy().ends_with(".tmp"))
            .cloned()
            .collect();
        assert!(debris.is_empty());

        // deleting moves the file aside; it does not destroy it
        let before = entries(root, None).len();
        let gone = discard(&p, root).unwrap();
        assert!(gone.exists() && !p.exists());
        assert_eq!(entries(root, None).len(), before - 1, "trashed entries must leave the listing");
        assert_eq!(
            read(&gone).unwrap().meta.self_reported.as_deref(),
            Some("tired"),
            "the file itself is untouched"
        );
    }

    #[test]
    fn the_day_boundary_is_a_setting_so_writing_must_honour_it() {
        let dir = tempdir();
        let root = dir.as_path();
        let mut e = NewEntry::new("yashwanth", at(12, 2, 14));
        e.turns = vec![("Hyperjournal".into(), "late".into())];
        e.hypothesis = hyp();
        e.self_reported = Some("up late".into());
        e.day_start_hour = 6;
        let late = write(&e, root).unwrap();
        assert_eq!(read(&late).unwrap().meta.journal_date, "2026-09-11");
    }

    #[test]
    fn normalize_agrees_on_case_and_spacing() {
        assert_eq!(normalize("  Pretty   Tired! "), "pretty tired");
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("a".repeat(80).as_str()).len(), 60);
    }

    /// An entry written by the Python must still read. This is the exact
    /// frontmatter python-frontmatter produced, byte for byte.
    #[test]
    fn reads_an_entry_written_by_the_python() {
        let raw = "---\nconfirmed: true\nhypothesis:\n  calibrated: true\n  confidence: 0.892\n  \
                   distribution:\n    Anger: 0.101\n    Neutral: 0.892\n  label: Neutral\n\
                   journal_date: '2026-09-11'\nself_reported: bro, today was fully wasted\n\
                   summary: ''\nthemes: []\ntimestamp: '2026-09-11T20:36:40+05:30'\nuser: yash\n\
                   ---\n\n**Hyperjournal:** Evening.\n\n**Yashwanth:** Tired.\n";
        let post = deserialize(raw).unwrap();
        assert_eq!(post.meta.user, "yash");
        assert_eq!(post.meta.journal_date, "2026-09-11");
        assert_eq!(post.meta.hypothesis.label, "Neutral");
        assert_eq!(post.meta.hypothesis.confidence, 0.892);
        assert_eq!(post.meta.hypothesis.distribution["Anger"], 0.101);
        assert_eq!(post.meta.summary, "");
        assert!(post.meta.themes.is_empty());
        assert!(post.content.starts_with("**Hyperjournal:** Evening."));
        assert!(post.content.ends_with("**Yashwanth:** Tired."));
    }

    /// Whatever this writes, it must be able to read, and the frontmatter must
    /// come out in the same alphabetical order the Python used.
    #[test]
    fn written_frontmatter_stays_alphabetical() {
        let dir = tempdir();
        let mut e = NewEntry::new("yash", at(11, 8, 14));
        e.turns = turns();
        e.hypothesis = hyp();
        e.self_reported = Some("tired".into());
        e.song = Some(Song {
            title: "Nightswimming".into(),
            artist: "R.E.M.".into(),
            why: "It sits with you.".into(),
        });
        let p = write(&e, dir.as_path()).unwrap();
        let raw = fs::read_to_string(&p).unwrap();

        let keys: Vec<&str> = raw
            .lines()
            .skip(1) // the opening ---
            .take_while(|l| *l != "---") // stop at the closing one, before the body
            .filter(|l| !l.starts_with(' ') && l.contains(':'))
            .map(|l| l.split(':').next().unwrap())
            .collect();
        assert!(keys.contains(&"song"), "song must reach the file: {keys:?}");
        // the two identity fields stay quoted, so no YAML 1.1 reader turns them
        // into dates behind our back
        assert!(raw.contains("journal_date: '2026-09-11'"), "{raw}");
        assert!(raw.contains("timestamp: '2026-09-11T08:14:22+05:30'"), "{raw}");
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "frontmatter keys must be alphabetical: {keys:?}");
        assert!(raw.starts_with("---\n"));
        assert!(read(&p).is_ok());
    }

    // --- a temp dir, without pulling in a crate for six lines ----------------

    struct TempDir(PathBuf);

    impl TempDir {
        fn as_path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir() -> TempDir {
        use std::sync::atomic::{AtomicU32, Ordering};
        static N: AtomicU32 = AtomicU32::new(0);
        let p = std::env::temp_dir().join(format!(
            "hyperjournal-test-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}
