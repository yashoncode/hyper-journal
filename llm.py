"""Phase 2: the greeting. A hypothesis about a face goes in, a question comes out.

The model never gets to assert how you feel. It gets to ask. If the key is
missing, the network is down, or the response is malformed, a canned greeting
takes over and the session continues -- a lost sentence is not worth a lost
journal entry.

Key lookup order: $NVIDIA_API_KEY, then a `.env` file next to this one.
"""
import json
import os
import pathlib
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).parent
ENV = HERE / ".env"
BASE_URL = "https://integrate.api.nvidia.com/v1"
SUGGESTED_MODEL = "deepseek-ai/deepseek-v4-flash-0731"   # offered in the app, never applied behind your back
TIMEOUT = 45          # NIM's free tier queues; a slow reply still beats a canned one

# Below this the guess is not worth saying out loud; ask an open question instead.
from emotion_cam import CONFIDENCE_FLOOR as FLOOR_FALLBACK


def floor():
    """The confidence floor the person chose, or the shipped one."""
    v = settings().get("CONFIDENCE_FLOOR")
    return FLOOR_FALLBACK if v is None else v

# A label is a model's word, not yours. These are the softened forms it is
# allowed to offer back to you, and every one of them is easy to say "no" to.
SOFTENED = {
    "Anger": "a bit wound up",
    "Contempt": "a bit unimpressed",
    "Disgust": "a bit put off",
    "Fear": "a bit on edge",
    "Happiness": "pretty bright",
    "Sadness": "a little low",
    "Surprise": "caught off guard",
}

SYSTEM = """You are Hyperjournal, a journal that greets one person by name when they sit down.

You are given a guess about their mood made by a camera. That guess is wrong roughly a third of the time and you must treat it as a guess, never as a fact.

Write exactly one spoken sentence, at most 28 words:
- Greet them by name and ask how their day has gone, right from the start of it. Not "how are you now" - you want the whole day.
- Any mood guess must be phrased as an open question they can disagree with.
- Never assert how they feel. "You look a little low - how has your day been, right from the morning?" is right. "I can see you're sad" is forbidden.
- If no mood guess is given, just ask openly about their day from the start.
- No emoji, no preamble, no quotation marks. Output the sentence only."""


def in_their_words(said, limit=60):
    """Their own first words, cut at a sentence or a word -- never mid-word.

    This is the fallback when the extraction call fails. It has to stay their
    words rather than an invented phrase, but "very very tiring and it's fully
    wasted, you" reads as a bug, because it is one.
    """
    text = " ".join((said or "").split())
    for stop in (". ", "! ", "? "):
        head = text.split(stop)[0]
        if head != text and len(head) <= limit:
            text = head
            break
    if len(text) > limit:
        text = text[:limit].rsplit(" ", 1)[0]
        head = text.rsplit(",", 1)[0]          # a dangling half-clause reads worse than a short one
        if "," in text and len(head) >= 20:
            text = head
    return text.strip(" ,;:-").rstrip(".!?").lower()


def looks_like(label):
    """The softened phrase for a label, or "" when there is nothing worth saying.
    Neutral has no phrase on purpose: "you look steady" is not a question."""
    return SOFTENED.get(label or "", "")


def read_env(path=ENV):
    """Three lines of .env parsing, so python-dotenv stays uninstalled."""
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def api_key():
    return os.environ.get("NVIDIA_API_KEY") or read_env().get("NVIDIA_API_KEY") or ""


def model_name():
    """The chosen model, or "". There is deliberately no default: picking one
    silently is how you end up talking to a model nobody chose."""
    return os.environ.get("NVIDIA_MODEL") or read_env().get("NVIDIA_MODEL") or ""


def chat(messages, model=None, max_tokens=120, temperature=0.7, timeout=TIMEOUT, tries=2):
    """One call to a NIM chat endpoint. Raises on any failure; callers fall back.

    Retried once, because this endpoint intermittently queues past the timeout
    and intermittently returns a message whose `content` is null (the answer
    went into the model's reasoning channel instead). Both are transient, and
    one retry is cheaper than a canned sentence.
    """
    key = api_key()
    if not key:
        raise RuntimeError("no API key yet")
    chosen = model or model_name()
    if not chosen:
        raise RuntimeError("no model chosen yet")
    body = json.dumps({
        "model": chosen,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    last = None
    for _ in range(tries):
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            # `content` is null when the model answered in its reasoning channel
            text = (data["choices"][0]["message"].get("content") or "").strip()
            if not text:
                raise ValueError("empty completion")
            return text
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                ValueError, KeyError, IndexError, TimeoutError) as e:
            last = e
    raise last


def list_models():
    key = api_key()
    req = urllib.request.Request(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return sorted(m["id"] for m in json.loads(r.read())["data"])


def canned(name, mood, part_of_day):
    if mood is None:
        return f"{part_of_day.capitalize()}, {name} — how has your day gone, right from the start?"
    return f"You look {mood} this {part_of_day}, {name} — how has your day been, from the morning on?"


def greet(name, label, confidence, part_of_day="morning", recent=()):
    """Return (sentence, source). `source` is 'nim' or 'canned'.

    A low-confidence guess, or a neutral one, is dropped entirely rather than
    softened -- 'you look steady' is not a question worth answering.
    """
    mood = SOFTENED.get(label) if confidence >= floor() else None
    fallback = canned(name, mood, part_of_day)

    context = [f"Person's name: {name}", f"Time of day: {part_of_day}"]
    context.append(f"Camera's mood guess: {label} (confidence {confidence:.2f}), i.e. they may look {mood}"
                   if mood else "Camera's mood guess: none confident enough to use. Ask an open question.")
    if recent:
        context.append("Their last few entries: " + "; ".join(recent))

    try:
        # someone is sitting there waiting to be spoken to. NIM's free tier queues
        # unpredictably (5-65s observed), so fail fast to the canned sentence
        # rather than leave them staring at a page that says "thinking".
        text = chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": "\n".join(context)}], timeout=12, tries=1)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, RuntimeError,
            ValueError, KeyError, TimeoutError) as e:
        return fallback, f"canned ({type(e).__name__})"
    return text.strip('"'), "nim"


def _selftest():
    # the greeting asks about the whole day, not just this minute
    open_line = canned("Yash", None, "morning")
    assert open_line.startswith("Morning, Yash") and "from the start" in open_line, open_line
    assert "low" in canned("Yash", SOFTENED["Sadness"], "evening")
    assert "day" in canned("Yash", SOFTENED["Sadness"], "evening")

    # below the floor: the guess is dropped, and no mood word survives
    text, src = greet("Yash", "Sadness", 0.21)
    assert src.startswith("canned") or src == "nim"
    if src.startswith("canned"):
        assert "low" not in text and "sad" not in text.lower(), text
        assert text.endswith("?"), text

    # neutral is above the floor but has no softened form, so it also drops
    assert SOFTENED.get("Neutral") is None
    text, _ = greet("Yash", "Neutral", 0.92)
    assert "steady" not in text

    assert read_env(pathlib.Path("does-not-exist")) == {}

    # conversation winds down on its own even with no network
    t, src = reply("Yash", [("Hyperjournal", "How's it going?"), ("Yash", "Tired.")])
    assert t and len(t) < 200, t
    t3, src3 = reply("Yash", [("Hyperjournal", "a"), ("Yash", "b"), ("Hyperjournal", "c"), ("Yash", "d"),
                              ("Hyperjournal", "e"), ("Yash", "f")])
    assert t3 == WINDING_DOWN[-1] or src3 == "nim", (t3, src3)

    # silence extracts to nothing, and nothing is invented
    data, src = extract([("Hyperjournal", "How are you?")])
    assert data == {"self_reported": "", "themes": [], "summary": ""} and src == "silent"

    # malformed model output must not reach the frontmatter
    assert _first_json("here you go: {\"a\": 1} hope that helps") == {"a": 1}
    assert _first_json("no json here") is None
    assert _first_json("{broken") is None

    # a song is a nicety: no material to work from means no song, never a guess
    assert suggest_song([("Hyperjournal", "how was your day?")]) is None

    # Latency budget. A greeting nobody hears is worse than a canned one, so
    # these ceilings are part of the behaviour, not a tuning detail.
    import unittest.mock as mock
    seen = {}
    def spy(name):
        def f(messages, **kw):
            seen[name] = (kw.get("timeout", TIMEOUT), kw.get("tries", 2))
            raise RuntimeError("stubbed")
        return f
    with mock.patch(__name__ + ".chat", spy("greet")):
        greet("Yash", "Sadness", 0.61)
    with mock.patch(__name__ + ".chat", spy("reply")):
        reply("Yash", [("Hyperjournal", "hi"), ("Yash", "tired")])
    with mock.patch(__name__ + ".chat", spy("extract")):
        extract([("Hyperjournal", "hi"), ("Yash", "tired")])
    assert seen["greet"] == (12, 1), seen["greet"]      # a person is waiting
    assert seen["reply"] == (15, 1), seen["reply"]      # a person is waiting
    assert seen["extract"][0] >= 45, seen["extract"]    # nobody is waiting

    # a fallback self-report is still their words, but it must not end mid-word
    real_case = "Bro, today was very very tiring and it's fully wasted, you know, I did nothing."
    got = in_their_words(real_case)
    assert got == "bro, today was very very tiring and it's fully wasted", got
    assert not got.endswith(","), got
    assert in_their_words("Just tired. Didn't sleep.") == "just tired"
    assert in_their_words("") == ""
    assert len(in_their_words("word " * 40)) <= 60

    # the configured floor must actually reach the decision it names
    real = read_env()
    with mock.patch(__name__ + ".read_env", lambda *a, **k: {**real, "CONFIDENCE_FLOOR": "0.9"}):
        assert floor() == 0.9
        with mock.patch(__name__ + ".chat", spy("greet")):
            text, _ = greet("Yash", "Sadness", 0.8)          # 0.8 is below a 0.9 floor
        assert "low" not in text and text.endswith("?"), text

    print("ok  (key present)" if api_key() else "ok  (no key; canned path only)")




# --- conversation -----------------------------------------------------------

CONVERSE_SYSTEM = """You are Hyperjournal, a journal having a short spoken conversation with one person \
about how their day is going.

Rules, all of them firm:
- One or two sentences. Never more.
- At most one question, and only if it opens something up.
- Never diagnose, never advise unless asked, never cheerlead.
- Wind down by the third exchange and let them go. A journal that will not stop \
talking is an alarm clock.
- Plain spoken English. No emoji, no lists, no quotation marks."""

EXTRACT_SYSTEM = """Read a short journal conversation and return JSON only, no prose, \
with exactly these keys:

  "self_reported": a short lowercase phrase in the PERSON'S OWN terms for how they said \
they feel, e.g. "tired", "anxious about work", "good, just busy". If they contradicted the \
camera's guess, follow the person. If they truly never said, use "".
  "themes": 1-3 short lowercase kebab-case tags, e.g. ["sleep","work-pressure"].
  "summary": one sentence, under 20 words, in the third person, about what was going on."""

WINDING_DOWN = [
    "That makes sense. Anything else sitting with you?",
    "Got it. Is there more to that, or shall I write it down?",
    "Thanks for telling me. Go easy on yourself tonight, and I'll note this down.",
]


def reply(name, turns, hypothesis=None, winding_down=False):
    """Next thing Hyperjournal says. `turns` is [(speaker, text), ...]. Returns (text, source).

    `winding_down` is decided by the caller from your own settings. It is not
    inferred from a turn count here, because how long you want to talk is not
    this function's business.
    """
    exchanges = sum(1 for who, _ in turns if who != "Hyperjournal")
    fallback = WINDING_DOWN[min(exchanges - 1, len(WINDING_DOWN) - 1)] if exchanges else WINDING_DOWN[0]

    context = [f"The person's name is {name}."]
    if hypothesis:
        context.append(f"The camera guessed {hypothesis[0]} at {hypothesis[1]:.2f} confidence. "
                       "It is often wrong; if they said otherwise, they are right.")
    if winding_down:
        context.append("This is the last exchange. Acknowledge and let them go.")

    messages = [{"role": "system", "content": CONVERSE_SYSTEM + "\n\n" + " ".join(context)}]
    for who, text in turns:
        messages.append({"role": "assistant" if who == "Hyperjournal" else "user", "content": text})

    try:
        return chat(messages, max_tokens=120, timeout=15, tries=1).strip('"'), "nim"
    except Exception as e:
        return fallback, f"canned ({type(e).__name__})"


def extract(turns):
    """Structured metadata for the frontmatter. Returns (dict, source).

    A malformed response falls back rather than writing a malformed entry.
    """
    said = [t for who, t in turns if who != "Hyperjournal"]
    blank = {"self_reported": "", "themes": [], "summary": ""}
    if not said:
        return blank, "silent"

    transcript = "\n".join(f"{who}: {text}" for who, text in turns)
    try:
        # nobody is waiting on this one: the conversation is over, so spend
        # the time to get real metadata instead of a canned blank
        raw = chat([{"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": transcript}],
                   max_tokens=300, temperature=0.2, timeout=45, tries=2)
    except Exception as e:
        # honest degradation: their own first words stand in, and no invented summary
        return {"self_reported": in_their_words(said[0]), "themes": [], "summary": ""}, f"canned ({type(e).__name__})"

    data = _first_json(raw)
    if not isinstance(data, dict) or not isinstance(data.get("themes", []), list):
        return {"self_reported": in_their_words(said[0]), "themes": [], "summary": ""}, "canned (malformed)"
    return {
        "self_reported": str(data.get("self_reported", ""))[:60].lower().strip(),
        "themes": [str(t)[:30].lower().strip() for t in data.get("themes", [])][:3],
        "summary": str(data.get("summary", "")).strip()[:200],
    }, "nim"


SONG_SYSTEM = """Pick one real, well-known song that suits how someone's day actually went.

Return JSON only, no prose, with exactly these keys:
  "title": the song title, exactly as released.
  "artist": the performing artist.
  "why": one short sentence, under 15 words, on why it fits. Speak to them, not about them.

Pick something that exists and is easy to find. Match the feeling rather than the words: a heavy day does not need a sad song, and a good day does not need a triumphant one. Never pick a song about self-harm or despair."""


def suggest_song(turns):
    """One song for the day, or None. Never blocks an entry from being written.

    Reads the conversation rather than the extracted summary, so it can run
    beside the extraction call instead of queueing behind it.
    """
    said = [t for who, t in turns if who != "Hyperjournal"]
    if not said:
        return None
    transcript = "\n".join(f"{who}: {text}" for who, text in turns)
    try:
        raw = chat([{"role": "system", "content": SONG_SYSTEM},
                    {"role": "user", "content": transcript}],
                   max_tokens=200, temperature=0.8, timeout=25, tries=1)
    except Exception:
        return None
    data = _first_json(raw)
    if not isinstance(data, dict) or not data.get("title") or not data.get("artist"):
        return None
    return {"title": str(data["title"])[:80].strip(),
            "artist": str(data["artist"])[:60].strip(),
            "why": str(data.get("why", ""))[:120].strip()}


def _first_json(text):
    """Models wrap JSON in prose and code fences no matter how firmly you ask."""
    import re
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None




# --- settings -----------------------------------------------------------------
# Everything a person might want to change lives in .env and is edited in the web
# app. Nothing here is hardcoded in the modules that use it.

SETTINGS = {
    # key in .env        cast        what it is
    "HYPERJOURNAL_NAME": (str,       "what it calls you out loud"),
    "NVIDIA_MODEL":      (str,       "which NIM model writes the sentences"),
    "CONFIDENCE_FLOOR":  (float,     "below this the camera's guess is not said out loud"),
    "DAY_START_HOUR":    (int,       "hour the journal day rolls over"),
    "MAX_EXCHANGES":     (int,       "replies before it winds down; 0 or blank means no limit"),
}


def settings():
    """Current settings, cast. Missing or unparseable values come back as None,
    so callers decide what an unset setting means rather than inheriting a guess."""
    env = read_env()
    out = {}
    for key, (cast, _) in SETTINGS.items():
        raw = (os.environ.get(key) or env.get(key) or "").strip()
        try:
            out[key] = cast(raw) if raw else None
        except ValueError:
            out[key] = None
    return out


def write_env(updates, path=ENV):
    """Rewrite .env in place, keeping comments and key order. Blank value clears.

    The API key is written here and never read back out to anyone: `settings()`
    does not carry it and no route returns it.
    """
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    remaining = dict(updates)
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    tmp = path.with_suffix(".env.tmp")
    tmp.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    import sys
    if "--models" in sys.argv:
        print("\n".join(list_models()))
    elif "--greet" in sys.argv:
        print(greet("Yashwanth", "Sadness", 0.61, "evening"))
    else:
        _selftest()
