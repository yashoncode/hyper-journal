"""Phase 3: the output. One Markdown file per entry, readable in a year with
no software at all.

This is the part that must outlive the camera, the models and the API, so it
has no dependency on any of them and is the best-tested thing here.
"""
import datetime as dt
import os
import pathlib
import re
import tempfile

import frontmatter

HERE = pathlib.Path(__file__).parent
JOURNAL = HERE / "journal"
DAY_START_HOUR = 4      # 01:30 on the 12th still belongs to the 11th


def journal_date(when: dt.datetime, day_start_hour: int = DAY_START_HOUR) -> dt.date:
    """The day the person is still living, which is not the calendar date.

    Deliberately not `when.date()`. Someone writing at 01:30 is finishing
    yesterday, and an entry filed under tomorrow is a subtly wrong history.
    """
    return (when - dt.timedelta(hours=day_start_hour)).date()


def path_for(when: dt.datetime) -> pathlib.Path:
    return JOURNAL / f"{when:%Y}" / f"{when:%m}" / f"{when:%Y-%m-%dT%H%M}.md"


def render_body(turns) -> str:
    """The verbatim transcript. Not a summary -- the summary has its own field."""
    return "\n\n".join(f"**{speaker}:** {text}".strip() for speaker, text in turns)


def write(user, turns, hypothesis, self_reported=None, themes=(), summary="",
          when=None, root=None, day_start_hour=None, song=None) -> pathlib.Path:
    """Write one entry atomically. Returns the path.

    `self_reported` is what the person said about themselves; `hypothesis` is
    what the camera thought. When they disagree, the person is right. When the
    person said nothing, `self_reported` is absent entirely -- not backfilled
    from the hypothesis, because the absence is the honest record.
    """
    when = when or dt.datetime.now().astimezone()
    assert when.tzinfo is not None, "naive datetimes lose the hour that was actually lived"
    confirmed = self_reported is not None

    meta = {
        "timestamp": when.isoformat(timespec="seconds"),
        "journal_date": journal_date(when, day_start_hour or DAY_START_HOUR).isoformat(),
        "user": user,
        "confirmed": confirmed,
        "themes": list(themes),
        "summary": summary,
        "hypothesis": hypothesis,
    }
    if confirmed:
        meta["self_reported"] = self_reported
    if song:
        meta["song"] = song

    post = frontmatter.Post(render_body(turns), **meta)
    out = (root or JOURNAL) / f"{when:%Y}" / f"{when:%m}" / f"{when:%Y-%m-%dT%H%M}.md"
    out.parent.mkdir(parents=True, exist_ok=True)

    # filenames are minute-granular, so two sessions in one minute collide.
    # Rare, but an overwrite here is a destroyed memory, so suffix instead.
    n = 2
    while out.exists():
        out = out.with_name(f"{when:%Y-%m-%dT%H%M}-{n}.md")
        n += 1

    # temp file in the same directory, then rename: a crash mid-write cannot
    # leave half an entry behind
    fd, tmp = tempfile.mkstemp(dir=out.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(frontmatter.dumps(post).encode("utf-8"))
        os.replace(tmp, out)
    except BaseException:
        pathlib.Path(tmp).unlink(missing_ok=True)
        raise
    return out


TRASH = "trashed"        # not a dot-directory: you should be able to find it


def discard(path, root=None) -> pathlib.Path:
    """Move an entry out of the journal instead of unlinking it.

    Deleting is a click. Regretting it is a week later. The file keeps its name
    under journal/trashed/, out of every listing, and is still there when you
    want it back.
    """
    path = pathlib.Path(path)
    dest = (pathlib.Path(root or JOURNAL) / TRASH) / path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 2
    while dest.exists():
        dest = dest.with_name(f"{path.stem}-{n}.md")
        n += 1
    os.replace(path, dest)
    return dest


def read(path) -> frontmatter.Post:
    return frontmatter.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def _order(path):
    """Sort key. A `-2` collision suffix is later than the file it followed, but
    sorts before it as plain text ('-' < '.'), so pull the suffix out first."""
    head, sep, tail = path.stem.rpartition("-")
    return (head, int(tail)) if sep and tail.isdigit() and "T" in head else (path.stem, 1)


def entries(root=None, limit=None):
    """Newest first. The filesystem is the index; there is no second copy to drift."""
    root = pathlib.Path(root or JOURNAL)
    if not root.exists():
        return []
    found = sorted(root.glob("*/*/*.md"), key=_order, reverse=True)
    return [(p, read(p)) for p in (found[:limit] if limit else found)]


def summaries(n=3, root=None):
    """One-liners from the last few entries, for greeting context."""
    out = []
    for _, post in entries(root, limit=n):
        s = (post.get("summary") or "").strip()
        if s:
            out.append(f"{post.get('journal_date')}: {s}")
    return out


def normalize(phrase: str) -> str:
    """A short lowercase phrase, so 'Pretty Tired!' and 'pretty tired' agree."""
    return re.sub(r"\s+", " ", (phrase or "").strip().strip(".!?").lower())[:60]


def _selftest():
    import tempfile as tf
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))

    # day boundary: 03:59 and 04:01 are different days; 23:59 is its own
    assert journal_date(dt.datetime(2026, 9, 12, 3, 59, tzinfo=tz)) == dt.date(2026, 9, 11)
    assert journal_date(dt.datetime(2026, 9, 12, 4, 1, tzinfo=tz)) == dt.date(2026, 9, 12)
    assert journal_date(dt.datetime(2026, 9, 11, 23, 59, tzinfo=tz)) == dt.date(2026, 9, 11)
    assert journal_date(dt.datetime(2026, 9, 12, 1, 30, tzinfo=tz)) == dt.date(2026, 9, 11)

    with tf.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        when = dt.datetime(2026, 9, 11, 8, 14, 22, tzinfo=tz)
        hyp = {"label": "Sadness", "confidence": 0.42, "calibrated": True,
               "distribution": {"Sadness": 0.42, "Neutral": 0.31}}
        turns = [("Hyperjournal", "You look a little low — how are you doing?"),
                 ("Yashwanth", "Just tired.\nDidn't sleep — café till 2am. ☕")]

        # round trip, including unicode and a multi-line turn
        p = write("yashwanth", turns, hyp, self_reported="tired",
                  themes=["sleep", "work-pressure"], summary="Slept badly.", when=when, root=root)
        assert p.name == "2026-09-11T0814.md" and p.parent == root / "2026" / "09", p
        got = read(p)
        assert got["self_reported"] == "tired"
        assert got["confirmed"] is True
        assert got["journal_date"] == "2026-09-11"
        assert got["hypothesis"] == hyp
        assert got["themes"] == ["sleep", "work-pressure"]
        assert "☕" in got.content and "café till 2am" in got.content
        assert got.content == render_body(turns)
        assert "+05:30" in got["timestamp"], got["timestamp"]

        # silence: confirmed false, and no self_reported key at all
        q = write("yashwanth", [("Hyperjournal", "Evening — how are you?")], hyp,
                  when=when.replace(hour=21), root=root)
        silent = read(q)
        assert silent["confirmed"] is False
        assert "self_reported" not in silent.keys(), silent.keys()

        # same minute twice must not overwrite: the first entry is a memory
        r = write("yashwanth", [("Hyperjournal", "again")], hyp, self_reported="fine",
                  when=when.replace(hour=21), root=root)
        assert r.name == "2026-09-11T2114-2.md", r.name
        assert read(q)["confirmed"] is False, "first entry was clobbered"

        assert len(entries(root)) == 3
        assert entries(root)[0][0] == r                       # newest first
        assert summaries(root=root) == ["2026-09-11: Slept badly."]
        assert not list(root.rglob("*.tmp"))                  # no debris

        # deleting moves the file aside; it does not destroy it
        n_before = len(entries(root))
        gone = discard(p, root=root)
        assert gone.exists() and not p.exists(), gone
        assert len(entries(root)) == n_before - 1, "trashed entries must leave the listing"
        assert read(gone)["self_reported"] == "tired", "the file itself is untouched"

        # the boundary is a setting, so writing must honour it
        late = write("yashwanth", [("Hyperjournal", "late")], hyp, self_reported="up late",
                     when=when.replace(day=12, hour=2), root=root, day_start_hour=6)
        assert read(late)["journal_date"] == "2026-09-11", read(late)["journal_date"]

    assert normalize("  Pretty   Tired! ") == "pretty tired"
    assert normalize(None) == ""
    print("ok")


if __name__ == "__main__":
    _selftest()
