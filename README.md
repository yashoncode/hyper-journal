# Hyperjournal

Webcam → face → emotion *hypothesis*, in a browser on your own machine.
Nothing is recorded.

The label on screen is a guess from a model that is wrong a lot (see below).
Phase 1 exists to find out how wrong it is on *your* face, in *your* room,
before any of it reaches a journal entry.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt        # Linux/mac: .venv/bin/python

mkdir -p models
curl -L -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# the emotion model normally self-downloads, but the package ships a broken
# import (`urllib` without `urllib.request`); fetch it by hand into the cache
# it looks in:
mkdir -p ~/.hsemotion
curl -L -o ~/.hsemotion/enet_b0_8_best_vgaf.onnx \
  https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx
```

## Run

```bash
.venv/Scripts/python web.py             # optional: web.py 8080
```

Then open **http://127.0.0.1:8000** — not `http://localhost:8000`. The server
binds IPv4 only, so nothing is reachable from the rest of your network, and on
Windows `localhost` resolves to `::1` first, which costs 2 seconds per request.
Browsers treat `127.0.0.1` as a secure origin, so the camera prompt still works.

The browser owns the camera and posts JPEG frames to the local process, which
runs the two models and posts numbers back. ~30 ms round trip on CPU.

A no-browser version with the same logic is still there if you want it:

```bash
.venv/Scripts/python emotion_cam.py              # b = baseline, c = clear, q = quit
.venv/Scripts/python emotion_cam.py --selftest   # calibration arithmetic, no camera
```

## Calibrate, then judge it

Hit **Capture baseline** and sit with an ordinary, unremarkable face. The 30
samples it averages become `calib.json`, and at runtime that average is
subtracted from every reading before the top label is picked.

This matters more than the model choice. A resting face reads as *sad* or
*angry* on every model trained on this kind of data, and it is not the same
wrong for every person. The baseline is what your nothing-in-particular looks
like to the model; what is left after subtracting it is the part that is
actually about your mood. Re-run it when the light, your hair, or the season
changes.

If almost nothing survives the subtraction (`RESIDUAL_FLOOR`, 2% of the mass),
the reading falls back to the uncalibrated one rather than normalizing leftover
rounding error into a mood you are not having.

## Stack

- **YuNet** (ships with OpenCV, 230 KB) — face presence and location, ~5 ms/frame.
- **HSEmotion** `enet_b0_8_best_vgaf` (ONNX, 16 MB) — 8-class AffectNet, ~22 ms.
  Chosen over DeepFace: no TensorFlow, ~30× smaller install, and trained on
  AffectNet rather than FER2013, which is a materially harder and more
  realistic dataset.
- **`http.server`** from the standard library. One user, one machine, no framework.

Labels: `Anger Contempt Disgust Fear Happiness Neutral Sadness Surprise`.
Fear, Disgust and Contempt are rare in the training data and are close to noise
in practice — treat a confident reading of those with suspicion.

## What the number means

Around 60–65% top-1 on posed, well-lit faces; worse with glasses, beards, side
lighting, an off-axis head, and worse again for faces outside the dataset's
demographic centre of mass. It is a **prompt**, not a measurement. Later phases
phrase it as a question you can say "no" to, and record your answer — not the
model's — as the truth.

## Privacy

No frame is written to disk, ever — not for debugging, not for caching. Frames
exist for the length of one HTTP request and are dropped. The server listens on
127.0.0.1 only; nothing leaves the machine. `calib.json` holds eight numbers and
no biometric identity.

## HTTP surface

| route | what |
|---|---|
| `GET /` | the page |
| `POST /analyze` | body is a JPEG; returns box, calibrated + raw distributions, top label, confidence |
| `POST /baseline` | start capturing 30 resting-face samples |
| `POST /baseline/clear` | forget the baseline, delete `calib.json` |

---

# Phase 2 — the greeting

The hypothesis from phase 1 becomes one spoken sentence, phrased so you can say
no to it. The model asks; it never asserts.

## Your API key

Paste it into `.env` (gitignored, never committed, never logged):

```
NVIDIA_API_KEY=nvapi-...
```

Get one at <https://build.nvidia.com> → pick any model → **Get API Key**.
`$NVIDIA_API_KEY` in the environment wins over the file if both are set.

```bash
.venv/Scripts/python llm.py --models    # what your key can reach
.venv/Scripts/python llm.py --greet     # one live greeting
.venv/Scripts/python llm.py             # self-check, no key needed
```

Default model is `meta/llama-3.3-70b-instruct`; set `NVIDIA_MODEL` in `.env` to
change it. NVIDIA NIM speaks the OpenAI chat-completions dialect, so this is a
plain HTTPS POST — no SDK, no new dependency.

## Without a key, it still works

No key, no network, a timeout, a malformed response — every one of those falls
back to a canned sentence and the session carries on. A lost sentence is not
worth a lost journal entry. Canned output for reference:

```
Sadness   0.61   You look a little low this evening, Yashwanth — how's it going?
Sadness   0.21   Evening, Yashwanth — how are you doing today?
Happiness 0.78   You look pretty bright this evening, Yashwanth — how's it going?
Neutral   0.92   Evening, Yashwanth — how are you doing today?
Anger     0.55   You look a bit wound up this evening, Yashwanth — how's it going?
```

Two rules visible there, and both are deliberate:

- **Below `CONFIDENCE_FLOOR` (0.45) the guess is dropped**, not softened. A
  25%-confident "sad" is noise, and dressing it up as "you seem a little low"
  puts a mood in your head that the camera did not actually find.
- **Neutral is dropped too**, even at 0.92. "You look steady" is not a question
  worth answering, and neutral is where this model class parks everything it
  cannot read.

## What leaves the machine

With a key set, each greeting sends: your name, the mood label, its confidence,
and the time of day. **No image, no video, no audio, ever** — the frames never
leave the `/analyze` request they arrived in. With the key blank, nothing leaves
at all.

| route | what |
|---|---|
| `POST /greet` | greeting for the latest hypothesis; `409` if no face has been seen yet |

---

# Phase 3 — the conversation and the journal

Hyperjournal greets you out loud, you hold a button and answer, it asks one or two
follow-ups, and then it writes a Markdown file you can read in a year with no
software at all. The files are the product; everything else is the input method.

## Using it

1. `.venv\Scripts\python web.py` → open <http://127.0.0.1:8000>
2. **Capture baseline** once, if you have not (phase 1).
3. **Greet me** — it speaks, using your own machine's voice.
4. **Hold to talk** (or hold the spacebar) while you answer. Release when done.
5. Up to four exchanges, then **Save entry**.

The journal appears underneath the camera, newest first. Unconfirmed entries —
the ones where you never answered — are drawn dashed and faded, and carry no
`self_reported` field at all.

## What an entry looks like

```markdown
---
confirmed: true
hypothesis:
  calibrated: true
  confidence: 0.214
  distribution: {Anger: 0.103, Disgust: 0.214, Fear: 0.214, Sadness: 0.21, ...}
  label: Fear
journal_date: '2026-09-11'
self_reported: knackered
summary: Yashwanth is tired from poor sleep and anxious about a client demo tomorrow.
themes:
- sleep
- work-pressure
timestamp: '2026-09-11T20:00:59+05:30'
user: yashwanth
---

**Hyperjournal:** Yashwanth, you seem a bit low this evening — what's been going on?

**Yashwanth:** Honestly, just knackered. I didn't sleep much, and I've got a client demo tomorrow at 2.

**Hyperjournal:** That makes sense. Anything else sitting with you?
```

The camera said **Fear**. You said **knackered**. `self_reported` is the field
that counts, and the hypothesis is kept beside it, with its full distribution,
so that a year from now you can see how wrong the camera was rather than having
to trust it. When they disagree, you are right.

`journal_date` is **not** `timestamp.date()`. The day rolls over at 04:00, so an
entry at 01:30 files under the previous day — the day you are still living.
`journal.py --selftest` pins that, along with the round trip and the same-minute
collision case.

## The voices

- **Out:** the browser's own `speechSynthesis` — offline, native, no model to
  download, and it uses whatever good voices Windows already has.
- **In:** `faster-whisper` `base.en`, on your CPU, in this process. Audio is
  transcribed and the buffer is dropped. It is never written to disk and never
  sent anywhere.

Push-to-talk rather than always-listening voice detection, deliberately: it means
the microphone is open only while you hold the button, which removes the whole
class of bugs where the speakers feed the microphone and the app interviews
itself. There is still a 300 ms guard after speech ends before the button
re-enables.

Whisper does not return silence for silence — it returns its training data
("Thank you.", "Subtitles by the Amara.org community"). Three guards stop that
reaching your journal: Silero VAD, a 0.6 s minimum, and a denylist. `stt.py`
asserts that three seconds of digital silence and a pure tone both transcribe to
nothing.

## Choosing the model

`meta/llama-3.3-70b-instruct` — the obvious default — is **not** served on this
key, and would have 404'd into a canned greeting forever without saying so. What
survived a sweep of fifteen candidates:

| model | greeting | verdict |
|---|---|---|
| `deepseek-ai/deepseek-v4-flash-0731` | 2.4 s, phrased as a question | **chosen** |
| `openai/gpt-oss-20b` | 17.5 s, "you seem a little sad" | too slow, too assertive |
| `nvidia/nemotron-3.5-lightning-30b-a3b` | leaked its reasoning trace | no |
| the other twelve | 404 or timeout | not served on this key |

Run `python llm.py --models` if your key ever changes.

**NIM's free tier queues unpredictably — 5 to 65 seconds for the same prompt.**
So the timeouts differ by who is waiting:

| call | timeout | why |
|---|---|---|
| greeting | 12 s, 1 try | someone is sitting there waiting to be spoken to |
| conversation turn | 15 s, 1 try | same |
| extraction | 45 s, 2 tries | the conversation is over; nobody is waiting |

Every one of them falls back rather than failing. Run with the network off and
you still get a greeting, still have a conversation, and still get an entry —
with an empty `summary` and your own first words as `self_reported`, which is
honest rather than invented.

## What leaves the machine

With a key set: the conversation text and the mood label. **No audio, no video,
no images, ever.** With `NVIDIA_API_KEY` blank, nothing leaves at all and every
feature still works except the wording of the sentences.

`journal/`, `calib.json`, `.env` and `models/` are all gitignored.

## Routes

| route | what |
|---|---|
| `POST /greet` | opens a session; greeting for the current hypothesis |
| `POST /listen` | body is audio; returns transcript or `""` for silence |
| `POST /say` | your turn in; Hyperjournal's reply out; `done` after 4 exchanges |
| `POST /save` | extract, write the Markdown file, end the session |
| `GET /entries` | the journal as JSON, newest first |

## Tests

```bash
.venv/Scripts/python journal.py       # round trip, day boundary, collisions, unicode
.venv/Scripts/python stt.py           # silence and tone produce no text
.venv/Scripts/python llm.py           # fallbacks, malformed JSON, wind-down
.venv/Scripts/python emotion_cam.py --selftest   # calibration arithmetic
```

All four pass with no camera and no microphone. `llm.py` passes with no network.

---

# Settings live in the app

There is nothing to hand-edit. Open <http://127.0.0.1:8000>, click **Settings**,
and everything is there:

| setting | what it does |
|---|---|
| Your name | what it calls you out loud, and the `user` field in every entry |
| Model | a live dropdown of every model **your** key can actually reach |
| NVIDIA API key | stored on this machine; blank keeps the current one |
| Say the guess above | the confidence floor — below it, it asks an open question instead |
| Day starts at | the hour the journal day rolls over |
| Replies before winding down | how many times it answers before it lets you go |

Saving rewrites `.env` in place, keeping the comments and leaving the key alone
unless you typed a new one. Changes apply on the next request — no restart.

**Nothing is chosen for you.** There is no default model: an unset model means
it says so and falls back to fixed sentences, rather than quietly talking to
something you never picked. `meta/llama-3.3-70b-instruct` returning 404 into a
silent canned greeting is exactly the failure that rule exists to prevent.

The key is written and never read back. `GET /config` returns `key_set: true`
and no key; no route, log line, or error message ever contains it.

## Demo entries

Three sample entries are on disk so the ledger is not empty on first look. They
are fabricated. Delete them before you start your own:

```bash
rm -rf journal
```

## Verified in a real browser

Playwright drives Chromium against the running server, so the page is checked
rather than assumed: the ledger renders, transcripts render as dialogue, the
settings panel loads 80 live models, saving preserves the key, the greeting
resolves in 6.7 s, and a full greet → hold-to-talk → transcribe → reply loop
completes with a fake microphone.

Two defects that found:

- `greet()` was still on a 45 s × 2 timeout because an earlier patch had
  silently missed its anchor, so the page sat on "thinking" for 90 s.
  `llm.py --selftest` now asserts the latency budget of all three calls.
- `speechSynthesis` never fired `onend` in a browser with no voices, leaving
  **Hold to talk** disabled forever. There is a watchdog now.

---

# The app, as it stands

Two pages. `/` is Today, `/settings` is everything you can change. Nothing
configurable is hardcoded and nothing is chosen for you.

## What the camera does now

It watches for the whole conversation, not just the moment it says hello. Every
frame while you are talking is kept in memory, and on save the mean of them
becomes the entry's record, along with what your face was doing in the first
third and the last third. So an entry can say *"While you talked it saw you go
from a little low to pretty bright"* — which is a real thing to know a year
later, and a single snapshot never was.

On screen it is only ever a sentence. No labels, no percentages, no probability
bars: a number invites belief the model has not earned. A wording has to hold
for five seconds before another can replace it, because a line that rewrites
itself thirty times a second is unreadable and looks unsure of itself.

## Talking

The greeting asks how the day has gone *from the start of it*, not how you are
this second. It replies for as long as you keep talking — **How long it talks**
in Settings is blank by default, which means no limit — and responds to what you
actually said: warmth when the day sounds hard, gladness when it sounds good.
Never a pep talk, never advice you didn't ask for.

**Clear** drops a conversation you haven't saved. It asks twice once you've
actually spoken.

## A song for the day

When an entry is saved, it picks one real song that suits how the day went and
links a search for it. It reads the conversation rather than the summary, so it
runs *beside* the extraction call rather than queueing behind it, and a save
waits for the slower of the two instead of both. If it fails, the entry is
written without a song.

## Deleting is not destroying

**Delete** on an entry, and **Clear all entries** in Settings, move files into
`journal/trashed/`. They leave the app and the listings; they are still on disk.
This app's whole job is not losing what you said, and an unrecoverable button in
it was a mistake.
