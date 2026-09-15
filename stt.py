"""Phase 3: speech in. Audio arrives, text leaves, the buffer is dropped.

Nothing is written to disk. The transcript goes into the journal because that
is the point of the journal; the recording does not, because it is not.
"""
import io
import os

MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base.en")
MIN_SECONDS = 0.6          # shorter than this is a cough or a mis-click
MIN_LOGPROB = -1.0         # the model's own confidence in what it heard

# Whisper does not return nothing for silence -- it returns its training data.
# These are the artifacts it reaches for, and every one of them would otherwise
# become a sentence you never said, in a journal you trust.
HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "bye",
    "you", "thanks", "please subscribe", "subtitles by the amara.org community",
    "subs by www.zeoranger.co.uk", "www.mooji.org", "amara.org", "okay",
    "transcription by castingwords", "i'm sorry", ".", "...",
}

_model = None


def load():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe(audio: bytes) -> str:
    """Audio bytes (any container ffmpeg reads) to text, or '' if it was silence.

    Returns '' rather than raising: a session where the microphone heard nothing
    is a real session that gets recorded as unconfirmed, not an error.
    """
    if not audio:
        return ""
    try:
        segments, info = load().transcribe(
            io.BytesIO(audio),
            beam_size=1,
            vad_filter=True,               # gate on speech, never on amplitude
            condition_on_previous_text=False,   # stops one hallucination seeding the next
        )
        segments = list(segments)
    except Exception:
        return ""

    if info.duration < MIN_SECONDS or not segments:
        return ""

    kept = [s.text.strip() for s in segments if s.avg_logprob > MIN_LOGPROB]
    text = " ".join(t for t in kept if t).strip()
    if not text or text.lower().strip(".!?,") in HALLUCINATIONS:
        return ""
    return text


def _selftest():
    import struct
    import wave

    def wav(seconds, sample=lambda i: 0):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
            w.writeframes(b"".join(struct.pack("<h", sample(i)) for i in range(int(16000 * seconds))))
        return buf.getvalue()

    assert transcribe(b"") == ""
    assert transcribe(b"not audio at all") == ""
    assert transcribe(wav(0.2)) == "", "too short must be dropped"

    silence = transcribe(wav(3.0))
    assert silence == "", f"silence hallucinated: {silence!r}"

    import math
    tone = transcribe(wav(3.0, lambda i: int(8000 * math.sin(i * 0.05))))
    assert tone == "", f"tone hallucinated: {tone!r}"

    assert "thank you" in HALLUCINATIONS
    print("ok  (silence and tone both produced no text)")


if __name__ == "__main__":
    _selftest()
