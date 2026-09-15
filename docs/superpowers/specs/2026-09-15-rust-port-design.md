# Hyperjournal: Python to Rust/Tauri v2

Replace the Python implementation with Rust so the project ships a Windows
installer and an Android APK, with no interpreter and no separate server
process. Behaviour is held constant: the Rust port must produce the same
journal files, the same greetings and the same emotion hypotheses as the
Python it replaces.

## Why this is a rewrite and not a wrapper

The Python app is a local HTTP server. The browser owns the camera and the
microphone and posts frames and audio to `127.0.0.1:8000`; the server runs
two ONNX models and Whisper, calls NVIDIA NIM, and writes Markdown. Shipping
that as an app means shipping CPython, OpenCV, ONNX Runtime and CTranslate2
to every machine, and none of it runs on Android. Each subsystem is ported
to a Rust module instead.

## Scope decisions

| Decision | Choice |
|---|---|
| Android capability | Full parity: camera emotion, on-device Whisper, speech out |
| Journal location | `Documents/Hyperjournal/` on both platforms |
| Whisper model | Downloaded on first run, not bundled |
| Python removal | After the Rust passes ported selftests, not before |

`Documents` over app-private storage because `journal.py` exists to write
files that "outlive the camera, the models and the API". Files that vanish on
uninstall do not. The cost is Android storage permissions.

The Whisper GGML model is 148 MB. Bundling it puts the APK over the Play
Store's 150 MB base limit, so it is fetched on first launch into the app data
dir with a progress bar.

## Module map

    src-tauri/src/
      lib.rs        builder, managed state, command registration
      paths.rs      journal root and config dir, per platform
      journal.rs    <- journal.py
      settings.rs   <- llm.py, settings half
      llm.rs        <- llm.py, NIM half
      vision.rs     <- emotion_cam.py
      stt.rs        <- stt.py
      speak.rs      new: text to speech
      commands.rs   <- web.py's routes

One crate, not a workspace. A `core` crate split would buy a Tauri-free test
build; `#[cfg(test)]` in one crate buys the same thing without the structure.

## Frontend to backend

Tauri `invoke` commands, not an embedded HTTP server. Keeping the server would
leave `src/index.html` untouched, but on Android it costs a cleartext-traffic
exemption, CORS headers and port lifecycle management. Fifteen mechanical
`fetch` to `invoke` edits is the smaller problem, on the platform where
problems are expensive.

State that was module-level globals behind a `threading.Lock` in `web.py`
becomes one `tauri::State<Mutex<App>>`.

## Four design choices inside the port

### Audio: raw PCM, not MediaRecorder

The frontend sends WebM/Opus today and Python decodes it through `av`, which
is ffmpeg. Rust has no dependable Opus-in-WebM decoder; `symphonia` does not
cover that pairing. Instead an AudioWorklet captures Float32 samples,
downsamples to 16 kHz mono, and sends raw floats, which is exactly what
whisper.cpp consumes. Roughly forty lines of JavaScript removes a codec
dependency from the Rust side entirely.

### Speech out: the `tts` crate on both platforms

`speechSynthesis` works in WebView2 but Android's System WebView does not
implement it. The `tts` crate wraps WinRT `SpeechSynthesizer` on Windows and
`android.speech.tts.TextToSpeech` over JNI on Android, giving one code path
and no custom Tauri plugin.

This also deletes the watchdog block at `src/index.html:107-129`, which exists
only to work around browsers dropping `onend` and machines with no voices
installed. The 300 ms guard interval before the microphone re-opens stays: it
stops the transcriber recording Hyperjournal's own sentence as the user's
reply.

### Face detection: hand-written YuNet decode

`cv2.FaceDetectorYN` hides the postprocessing. `ort` returns raw tensors, so
the port generates priors for strides 8, 16 and 32, decodes boxes, scores them
as `sqrt(cls * obj)` and runs NMS: about 120 lines.

The alternative is the `opencv` crate, which means building OpenCV for
Android. That is a worse trade than 120 lines. Verification is differential:
the same frames through `cv2.FaceDetectorYN` and through the Rust decoder,
comparing boxes.

### Settings: JSON in the config dir, migrated once

`.env` beside the executable cannot work once the app is installed to Program
Files, and has no meaning on Android. Settings move to JSON in the platform
config dir. On first run an existing `.env` is read and imported, so the API
key and preferences carry over. The key stays plaintext in a private directory,
which is what the Python did.

## Assets

| File | Size | Delivery |
|---|---|---|
| `face_detection_yunet_2023mar.onnx` | 232 KB | bundled |
| `enet_b0_8_best_vgaf.onnx` | 16 MB | bundled |
| `ggml-base.en.bin` | 148 MB | downloaded on first run |

## Behaviour that must survive the port

These are the parts with selftests in the Python, and the tests port with them:

- The journal day rolls over at 04:00, so 01:30 on the 12th belongs to the
  11th (`journal.py:22`).
- Two entries in one minute must not overwrite; the second gets a `-2` suffix
  (`journal.py:70`).
- Writes are atomic through a temp file and rename, leaving no debris.
- Deleting moves the file to `journal/trashed/` and never unlinks it.
- When the person said nothing, `self_reported` is absent from the frontmatter
  entirely rather than backfilled from the camera's guess.
- Baseline subtraction falls back to the raw distribution when almost nothing
  survives, rather than inventing a mood from rounding error
  (`emotion_cam.py:44`).
- A mood guess below the confidence floor is dropped, not softened.
- Every LLM call falls back to a canned sentence rather than failing the
  session, and the greeting's timeout is 12 s because someone is sitting there
  waiting.
- Whisper's silence hallucinations are filtered against a known list.

## Phases

Each phase ends with something runnable.

| # | Work | Ends with |
|---|---|---|
| 0 | git init, vendor ONNX models, `paths.rs` | safety net |
| 1 | `journal.rs` and its tests | file format proven |
| 2 | `settings.rs`, `llm.rs` and their tests | greeting and extraction work |
| 3 | `commands.rs`, frontend `fetch` to `invoke` | desktop app runs, typed input |
| 4 | `vision.rs`: ort, YuNet decode, hsemotion, calibration | desktop camera works |
| 5 | `stt.rs`: whisper-rs, model download, PCM capture | desktop voice works |
| 6 | `speak.rs`: tts crate | desktop parity |
| 7 | NSIS bundle | installer ships |
| 8 | Android: ort and whisper.cpp for arm64, permissions, WebView patch | APK ships |
| 9 | Delete Python, rewrite README | conversion complete |

Python stays in the tree until phase 9 as the reference to diff against.

## Known risks

All three are in phase 8 and none is a redesign:

- `ort` may have no prebuilt ONNX Runtime for `aarch64-linux-android`, forcing
  a source build.
- `whisper-rs` drives cmake, which must pick up the NDK toolchain file.
- Tauri's generated Android WebView does not grant camera or microphone
  permission by default; `WebChromeClient.onPermissionRequest` needs handling
  and the runtime permission needs requesting.
