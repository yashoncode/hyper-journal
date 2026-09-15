# Hyperjournal

A journal that greets you by name, guesses at your mood from the camera, and
then asks rather than tells. What you say back is what gets written down.

One Markdown file per entry, readable in a year with no software at all.

Windows app and Android app. One binary each — no Python, no server, no
interpreter, nothing listening on a port.

```
Documents/Hyperjournal/2026/09/2026-09-11T2036.md
```

## Install

- **Windows** — `Hyperjournal_0.1.0_x64-setup.exe`
- **Android** — `app-arm64-release.apk`, sideloaded. On first launch it opens the
  All-files-access screen; without it the journal falls back to app-private
  storage and disappears when you uninstall. arm64 only, which is every
  Android phone since about 2017.

On first run it fetches the speech model, ~148 MB, once. Everything else is
already in the binary.

## Where your journal lives

| | |
|---|---|
| Windows | `Documents\Hyperjournal\YYYY\MM\*.md` |
| Android | `/storage/emulated/0/Documents/Hyperjournal/YYYY/MM/*.md` |

Deleted entries move to `Hyperjournal/trashed/`. They are not destroyed.

Settings live in the platform config directory as `settings.json`. A `.env`
left over from the Python version is imported once, so an existing API key
carries over.

## Calibrate, then judge it

Under Settings, hit **Capture baseline** and sit with an ordinary,
unremarkable face. The 30 samples it averages are subtracted from every later
reading before the top label is picked.

This matters more than the model choice. A resting face reads as *sad* or
*angry* on every model trained on this kind of data, and it is not the same
wrong for every person. The baseline is what your nothing-in-particular looks
like to the model; what survives subtracting it is the part that is actually
about your mood. Re-run it when the light, your hair, or the season changes.

If almost nothing survives (2% of the mass), the reading falls back to the
uncalibrated one rather than normalizing leftover rounding error into a mood
you are not having.

## What an entry looks like

```markdown
---
confirmed: true
hypothesis:
  began: Sadness
  calibrated: true
  confidence: 0.214
  distribution:
    Anger: 0.103
    Fear: 0.214
    Sadness: 0.21
  ended: Neutral
  label: Fear
  samples: 812
journal_date: '2026-09-11'
self_reported: knackered
song:
  artist: R.E.M.
  title: Nightswimming
  why: It sits with you rather than trying to fix anything.
summary: Yashwanth is tired from poor sleep and anxious about a client demo tomorrow.
themes:
- sleep
- work-pressure
timestamp: '2026-09-11T20:00:59+05:30'
user: yashwanth
---

**Hyperjournal:** Yashwanth, you seem a bit low this evening — what's been going on?

**Yashwanth:** Honestly, just knackered. I didn't sleep much, and I've got a client demo tomorrow at 2.
```

The camera said **Fear**. You said **knackered**. `self_reported` is the field
that counts, and the hypothesis is kept beside it, with its whole distribution,
so a year from now you can see how wrong the camera was rather than having to
trust it. When they disagree, you are right.

If you said nothing, `self_reported` is **absent** — not empty, absent. The
silence is the honest record, and nothing is backfilled from the guess.

`journal_date` is not `timestamp.date()`. The day rolls over at 04:00, so an
entry at 01:30 files under the previous day: the day you are still living.

## The voices

- **Out** — `speechSynthesis` on the desktop, which is already there and uses
  whatever good voices Windows has. Android's WebView has no Web Speech API, so
  there it goes through `android.speech.tts.TextToSpeech` instead.
- **In** — whisper.cpp `base.en`, on your own CPU, in this process. The page
  takes raw samples off the audio graph and hands them over at 16 kHz. The
  buffer is dropped; it is never written to disk and never sent anywhere.

Push-to-talk rather than always-listening, deliberately: the microphone is open
only while you hold the button, which removes the whole class of bugs where the
speakers feed the microphone and the app interviews itself. There is a 300 ms
guard after speech ends before the button re-enables.

Whisper does not return silence for silence — it returns its training data
("Thank you.", "Subtitles by the Amara.org community"). Three guards stop that
reaching your journal: a 0.6 s minimum, whisper's own `no_speech_probability`
and mean token log-probability, and a denylist.

## What leaves the machine

The conversation text, to whichever NIM model you chose, to write the greeting
and pull out the themes. That is all.

No frame is ever written to disk — not for debugging, not for caching. Frames
exist for one call and are dropped. Audio likewise. The calibration baseline is
eight numbers and no biometric identity.

Without an API key the app still works: it falls back to fixed sentences and
records what you say exactly as it would otherwise.

## Stack

- **YuNet** (232 KB) — where the face is, ~5 ms/frame.
- **HSEmotion** `enet_b0_8_best_vgaf` (16 MB) — 8-class AffectNet, ~22 ms.
- **ONNX Runtime** via `ort`, with NNAPI on Android.
- **whisper.cpp** via `whisper-rs`.
- **Tauri v2** — the system webview, and Rust underneath. No node, no bundler;
  the frontend is three static files.

Labels: `Anger Contempt Disgust Fear Happiness Neutral Sadness Surprise`. Fear,
Disgust and Contempt are rare in the training data and close to noise in
practice — treat a confident reading of those with suspicion.

Around 60–65% top-1 on posed, well-lit faces; worse with glasses, beards, side
lighting, an off-axis head, and worse again for faces outside the dataset's
demographic centre of mass. It is a **prompt**, not a measurement.

## The command surface

The page calls Rust directly; there is no HTTP.

| command | what |
|---|---|
| `analyze` | raw JPEG bytes in; box, distributions, top label, confidence out |
| `greet` / `say` / `save` / `discard` | the conversation, and writing it down |
| `listen` | 4 bytes of sample rate then raw f32 samples; text out |
| `speak` | platform text-to-speech, where the webview has none |
| `entries` / `delete_entry` / `delete_all` | the journal |
| `config` / `set_config` / `models` | settings; the API key goes in and never comes back out |
| `start_baseline` / `clear_baseline` | calibration |

## Building it

Needs Rust, and for the speech model [LLVM][llvm] (bindgen) and [CMake][cmake]
(whisper.cpp). Paths to both are pinned in `src-tauri/.cargo/config.toml` —
change them if yours live elsewhere.

```bash
cd src-tauri
cargo test                  # 40 tests, no camera and no network needed
cargo tauri build           # -> target/release/bundle/nsis/*.exe
```

### Android

Also needs the Android SDK and NDK r27, plus [Ninja][ninja].

```bash
export ANDROID_HOME=~/AppData/Local/Android/Sdk
export NDK_HOME="$ANDROID_HOME/ndk/27.3.13750724"
cargo tauri android build --target aarch64
```

That will get as far as copying the built library and then stop: Tauri
*symlinks* it into `jniLibs`, and Windows refuses without developer mode. Copy
it yourself and let gradle finish:

```bash
cp target/aarch64-linux-android/release/libhyperjournal_lib.so \
   gen/android/app/src/main/jniLibs/arm64-v8a/
cd gen/android
./gradlew assembleArm64Release -x rustBuildArm64Release
# -> app/build/outputs/apk/arm64/release/app-arm64-release.apk
```

Signing comes from `gen/android/keystore.properties`, which is gitignored along
with the `.jks` it points at. **Keep that keystore.** Android will refuse to
install an update signed by a different key.

Two upstream bugs are worked around at build time, both because `whisper-rs-sys`
asks `cfg!(target_os = "windows")` in a build script, where that describes the
host rather than the target: `src-tauri/android-toolchain.cmake` pins the ABI
and strips MSVC's `/utf-8`, and `src-tauri/android-stubs/` explains the empty
`libadvapi32.a`.

[llvm]: https://github.com/llvm/llvm-project/releases
[cmake]: https://cmake.org/download/
[ninja]: https://github.com/ninja-build/ninja/releases

## Tests

```bash
cd src-tauri && cargo test
```

40 of them, all offline. They pin the things that would be quietly wrong
otherwise: the 04:00 day boundary, two entries in the same minute not
overwriting each other, `self_reported` being absent rather than empty when you
said nothing, frontmatter staying byte-compatible with what the Python wrote,
baseline subtraction falling back instead of inventing a mood, every LLM call
degrading to a canned sentence rather than failing the session, and the latency
budget the greeting is allowed.

The face detector was checked against `cv2.FaceDetectorYN` directly: same image,
same boxes, same scores to four decimals.
