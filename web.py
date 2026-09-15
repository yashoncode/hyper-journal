"""Phase 1, in a browser. http://localhost:8000

The browser owns the camera and posts JPEG frames here; this process runs the
face detector and the emotion model and posts back numbers. No frame is
written to disk at either end -- the bytes live in one request and are dropped.

    python web.py [port]
"""
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

import journal
import llm
import stt
from emotion_cam import BASELINE_FRAMES, CALIB, EMA, LABELS, YUNET, calibrate, softmax

HERE = pathlib.Path(__file__).parent
def cfg(key, fallback):
    """A setting, or the fallback when the person has not chosen one yet.
    Read per request, so changing it in the app takes effect without a restart."""
    v = llm.settings().get(key)
    return fallback if v is None or v == "" else v
lock = threading.Lock()          # the detector is stateful; one frame at a time

detector = recognizer = None
probs = np.full(len(LABELS), 1 / len(LABELS), dtype=np.float32)
baseline = None
collecting = None                # a list while capturing a baseline, else None
last = None                      # the most recent hypothesis, for /greet
session = None                   # the conversation in progress, or None

MODELS = []                      # what this key can reach, filled on first ask
MAX_READINGS = 4000              # ~20 minutes of frames, then it stops accumulating
VOICE = "Hyperjournal"


def load():
    global detector, recognizer, baseline
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8, 0.3, 5000)
    recognizer = HSEmotionRecognizer()
    assert recognizer.idx_to_class == dict(enumerate(LABELS)), recognizer.idx_to_class
    recognizer.predict_emotions(np.zeros((224, 224, 3), np.uint8))   # first call is ~2s
    if CALIB.exists():
        baseline = np.array(json.loads(CALIB.read_text())["baseline"], dtype=np.float32)


def thought_line(post):
    """What the camera made of you across the conversation, in a sentence.
    No label, no percentage -- the guess stays on the record without pretending
    to be a measurement."""
    h = post.get("hypothesis") or {}
    phrase = llm.looks_like(h.get("label"))
    watched = h.get("samples")
    began, ended = llm.looks_like(h.get("began")), llm.looks_like(h.get("ended"))

    if began and ended and began != ended:
        arc = f"While you talked it saw you go from {began} to {ended}."
    elif phrase:
        arc = (f"It watched you through the conversation and mostly saw you {phrase}."
               if watched else f"It thought you seemed {phrase}.")
    else:
        arc = "It could not make much of your face, which is common and fine."

    if post.get("confirmed"):
        return arc + " You said what you said, and that is what is written."
    return arc + " You never answered, so nothing was recorded as yours."


def part_of_day(hour=None):
    h = datetime.now().hour if hour is None else hour
    return "morning" if h < 12 else "afternoon" if h < 17 else "evening"


def analyze(jpeg: bytes) -> dict:
    """One frame in, one hypothesis out. Called under `lock`."""
    global probs, baseline, collecting, last
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "bad frame"}
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None or not len(faces):
        return {"box": None, "faces": 0}

    # largest face is the subject; whoever is behind them is not
    box = np.clip(max(faces, key=lambda f: f[2] * f[3])[:4].astype(int), 0, [w, h, w, h])
    x, y, bw, bh = box
    crop = frame[y:y + bh, x:x + bw]
    if not crop.size:
        return {"box": None, "faces": len(faces)}

    _, logits = recognizer.predict_emotions(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    raw = softmax(np.asarray(logits, dtype=np.float32))
    probs = EMA * probs + (1 - EMA) * raw

    remaining = 0
    if collecting is not None:
        collecting.append(raw)
        remaining = BASELINE_FRAMES - len(collecting)
        if remaining <= 0:
            baseline = np.mean(collecting, axis=0)
            CALIB.write_text(json.dumps({"baseline": baseline.tolist(), "labels": LABELS}, indent=2))
            collecting = None

    shown = calibrate(probs, baseline)
    top = int(shown.argmax())
    last = (LABELS[top], float(shown[top]))

    # while you are talking, keep watching. One reading at hello is a thin
    # record of a conversation; the shape of it over five minutes is not.
    if session is not None and len(session["readings"]) < MAX_READINGS:
        session["readings"].append(shown.copy())
    sure = bool(shown[top] >= llm.floor())
    return {
        # a sentence, not a number: nobody journalling wants to read "sad 0.42",
        # and a number invites belief the model has not earned
        "phrase": llm.looks_like(LABELS[top]) if sure else "",
        "box": [int(v) for v in box],
        "frame": [w, h],
        "faces": len(faces),
        "probs": {k: round(float(v), 4) for k, v in zip(LABELS, shown)},
        "raw": {k: round(float(v), 4) for k, v in zip(LABELS, raw)},
        "label": LABELS[top],
        "confidence": round(float(shown[top]), 4),
        "sure": sure,
        "calibrated": baseline is not None,
        "remaining": max(0, remaining),
    }


def hypothesis_block():
    """What the camera thought, frozen at the moment the session started."""
    shown = calibrate(probs, baseline)
    top = int(shown.argmax())
    return {
        "label": LABELS[top],
        "confidence": round(float(shown[top]), 3),
        "calibrated": baseline is not None,
        "distribution": {k: round(float(v), 3) for k, v in zip(LABELS, shown)},
    }


def reaction(readings, calibrated):
    """What your face did across the whole conversation, averaged.

    Not the moment it said hello. The mean is the honest summary of a noisy
    signal, and the first and last thirds say whether anything shifted.
    """
    if not readings:
        return None
    stack = np.array(readings, dtype=np.float32)
    mean = stack.mean(axis=0)
    third = max(1, len(stack) // 3)
    began = LABELS[int(stack[:third].mean(axis=0).argmax())]
    ended = LABELS[int(stack[-third:].mean(axis=0).argmax())]
    top = int(mean.argmax())
    return {
        "label": LABELS[top],
        "confidence": round(float(mean[top]), 3),
        "calibrated": calibrated,
        "distribution": {k: round(float(v), 3) for k, v in zip(LABELS, mean)},
        "samples": len(stack),
        "began": began,
        "ended": ended,
    }


def save_session():
    """End the conversation and write the entry. Returns what was written."""
    global session
    if session is None:
        return {"error": "nothing to save"}, 409

    turns = session["turns"]
    spoke = [t for who, t in turns if who != VOICE]
    # the whole-conversation reading if the camera saw anything, else the one
    # it had at hello, which is all there was
    watched = reaction(session["readings"], session["hypothesis"].get("calibrated", False))
    hypothesis = watched or session["hypothesis"]
    # both of these are slow and neither needs the other's answer, so they wait
    # together rather than one behind the other
    with ThreadPoolExecutor(2) as pool:
        pending_song = pool.submit(llm.suggest_song, turns) if spoke else None
        data, src = llm.extract(turns)
        song = pending_song.result() if pending_song else None

    # the person said nothing: the entry records the hypothesis and says so.
    # self_reported is left out entirely rather than backfilled from the guess.
    self_reported = journal.normalize(data["self_reported"]) if spoke and data["self_reported"] else None
    path = journal.write(
        user=cfg("HYPERJOURNAL_NAME", "there").lower(), turns=turns,
        hypothesis=hypothesis, day_start_hour=cfg("DAY_START_HOUR", journal.DAY_START_HOUR),
        self_reported=self_reported, themes=data["themes"], summary=data["summary"], song=song,
    )
    session = None
    return {"path": str(path.relative_to(HERE)), "confirmed": self_reported is not None,
            "self_reported": self_reported, "themes": data["themes"],
            "summary": data["summary"], "song": song, "source": src}, 200


class Handler(BaseHTTPRequestHandler):
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/config":
            return self.send_json({
                "settings": llm.settings(),          # never includes the API key
                "key_set": bool(llm.api_key()),
                "fields": {k: d for k, (_, d) in llm.SETTINGS.items()},
                "voice": VOICE,
            })
        if self.path == "/models":
            global MODELS
            if not MODELS:
                try:
                    MODELS = llm.list_models()
                except Exception as e:
                    return self.send_json({"error": f"{type(e).__name__}: {e}"}, 502)
            return self.send_json(MODELS)
        if self.path == "/entries":
            return self.send_json([
                {"path": str(p.relative_to(HERE)), "date": post.get("journal_date"),
                 "time": str(post.get("timestamp", ""))[11:16],
                 "confirmed": bool(post.get("confirmed")),
                 "self_reported": post.get("self_reported"),
                 "summary": post.get("summary"), "themes": post.get("themes") or [],
                 "thought": thought_line(post), "song": post.get("song"),
                 "body": post.content}
                for p, post in journal.entries(limit=60)])
        served = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/index.html": ("index.html", "text/html; charset=utf-8"),
                  "/settings": ("settings.html", "text/html; charset=utf-8"),
                  "/app.css": ("app.css", "text/css; charset=utf-8")}
        if self.path not in served:
            return self.send_error(404)
        name, ctype = served[self.path]
        body = (HERE / name).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        global baseline, collecting
        if self.path == "/analyze":
            n = int(self.headers.get("Content-Length", 0))
            if n > 4_000_000:
                return self.send_error(413)
            data = self.rfile.read(n)
            with lock:
                return self.send_json(analyze(data))
        if self.path == "/greet":
            global session
            with lock:
                hypothesis = last
                block = hypothesis_block() if hypothesis else None
            if hypothesis is None:
                return self.send_json({"error": "no face seen yet"}, 409)
            label, confidence = hypothesis
            # the network call is slow and must not hold the frame lock
            text, source = llm.greet(cfg("HYPERJOURNAL_NAME", "there"), label, confidence,
                                     part_of_day(), recent=journal.summaries(3))
            session = {"turns": [(VOICE, text)], "hypothesis": block, "readings": []}
            return self.send_json({"text": text, "source": source, "label": label,
                                   "confidence": round(confidence, 3), "exchanges": 0})

        if self.path == "/listen":
            n = int(self.headers.get("Content-Length", 0))
            if n > 25_000_000:
                return self.send_error(413)
            audio = self.rfile.read(n)
            return self.send_json({"text": stt.transcribe(audio)})

        if self.path == "/say":
            n = int(self.headers.get("Content-Length", 0))
            said = json.loads(self.rfile.read(n) or b"{}").get("text", "").strip()
            if session is None:
                return self.send_json({"error": "no session; greet first"}, 409)
            if not said:
                return self.send_json({"error": "nothing heard"}, 400)
            name = cfg("HYPERJOURNAL_NAME", "there")
            session["turns"].append((name, said))
            exchanges = sum(1 for who, _ in session["turns"] if who != VOICE)
            # 0 or unset means talk for as long as you like
            limit = cfg("MAX_EXCHANGES", 0) or 0
            if limit and exchanges >= limit:
                return self.send_json({"text": "", "source": "done", "exchanges": exchanges,
                                       "done": True})
            hyp = session["hypothesis"]
            text, source = llm.reply(name, session["turns"], (hyp["label"], hyp["confidence"]),
                                     winding_down=bool(limit) and exchanges + 1 >= limit)
            session["turns"].append((VOICE, text))
            return self.send_json({"text": text, "source": source, "exchanges": exchanges,
                                   "done": bool(limit) and exchanges + 1 >= limit})

        if self.path == "/delete":
            n = int(self.headers.get("Content-Length", 0))
            rel = json.loads(self.rfile.read(n) or b"{}").get("path", "")
            target = (HERE / rel).resolve()
            root = journal.JOURNAL.resolve()
            # a path from the browser is not to be trusted with unlink()
            if target.suffix != ".md" or not target.is_relative_to(root) or not target.is_file():
                return self.send_json({"error": "no such entry"}, 400)
            journal.discard(target)
            return self.send_json({"deleted": rel})

        if self.path == "/config":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            updates = {k: str(v).strip() for k, v in body.items()
                       if k in llm.SETTINGS or k == "NVIDIA_API_KEY"}
            if not updates:
                return self.send_json({"error": "nothing to change"}, 400)
            llm.write_env(updates)                 # the key goes in and never comes back out
            return self.send_json({"saved": sorted(updates)})

        if self.path == "/discard":
            session = None                         # nothing was written, so nothing is lost
            return self.send_json({"cleared": True})

        if self.path == "/save":
            body, code = save_session()
            return self.send_json(body, code)
        if self.path == "/baseline":
            with lock:
                collecting = []
            return self.send_json({"ok": True, "remaining": BASELINE_FRAMES})
        if self.path == "/baseline/clear":
            with lock:
                baseline, collecting = None, None
                CALIB.unlink(missing_ok=True)
            return self.send_json({"ok": True})
        self.send_error(404)

    def do_DELETE(self):
        if self.path != "/entries/all":
            return self.send_error(404)
        gone = 0
        for path in journal.JOURNAL.glob("*/*/*.md"):
            journal.discard(path)
            gone += 1
        return self.send_json({"deleted": gone})

    def log_message(self, *a):
        pass                      # one line per frame is not a log, it is a flood


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("loading models...")
    load()
    # 127.0.0.1, not localhost: we bind IPv4 only (so nothing is exposed on the
    # LAN) and on Windows "localhost" resolves to ::1 first, costing 2s a request.
    # Browsers treat 127.0.0.1 as a secure origin, so getUserMedia still works.
    print(f"greeting: {llm.settings()['NVIDIA_MODEL'] or 'no model chosen'}"
          if llm.api_key() else "greeting: canned (no API key yet)")
    print(f"http://127.0.0.1:{port}   (ctrl-c to stop)")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass
