"""Phase 1: webcam -> face -> emotion hypothesis, on screen only.

Nothing is written to disk except the calibration baseline (calib.json).
No frames, no images, ever.

Keys:  b = capture neutral baseline   c = clear baseline   q = quit
"""
import json
import pathlib
import sys
import time

import cv2
import numpy as np
from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

HERE = pathlib.Path(__file__).parent
YUNET = HERE / "models" / "face_detection_yunet_2023mar.onnx"
CALIB = HERE / "calib.json"

LABELS = ["Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
ANALYZE_EVERY = 5        # frames; emotion net is ~30ms, detector is ~5ms
EMA = 0.6                # smoothing of the probability distribution
BASELINE_FRAMES = 30
CONFIDENCE_FLOOR = 0.45  # below this the label is not worth saying out loud
RESIDUAL_FLOOR = 0.02    # mass surviving baseline subtraction, below which nothing is happening


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def calibrate(probs, baseline):
    """Subtract the user's resting-face distribution, renormalize.

    A neutral face reads as 'sad' or 'angry' on every model trained on this
    data; the baseline is what that person's nothing-in-particular looks like.
    """
    if baseline is None:
        return probs
    d = np.clip(probs - baseline, 0, None)
    s = d.sum()
    # almost nothing survived the subtraction: this face is at rest, and
    # normalizing the leftover noise would invent a mood out of rounding error
    return d / s if s > RESIDUAL_FLOOR else probs


def draw(frame, box, probs, label, conf, calibrated, collecting):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
    head = f"{label} {conf:.2f}" if conf >= CONFIDENCE_FLOOR else f"unsure ({label} {conf:.2f})"
    cv2.putText(frame, head, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)

    for i, (name, p) in enumerate(zip(LABELS, probs)):
        ty = 24 + i * 22
        cv2.rectangle(frame, (10, ty - 12), (10 + int(p * 200), ty + 4), (200, 160, 60), -1)
        cv2.putText(frame, f"{name} {p:.2f}", (218, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    status = f"baseline: {'on' if calibrated else 'off (press b)'}"
    if collecting:
        status = f"hold a resting face... {collecting}"
    cv2.putText(frame, status, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    if not YUNET.exists():
        sys.exit(f"missing {YUNET} -- see README")

    print("loading models...")
    detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.8, 0.3, 5000)
    recognizer = HSEmotionRecognizer()
    assert recognizer.idx_to_class == dict(enumerate(LABELS)), recognizer.idx_to_class
    recognizer.predict_emotions(np.zeros((224, 224, 3), np.uint8))   # first call is ~2s

    baseline = None
    if CALIB.exists():
        baseline = np.array(json.loads(CALIB.read_text())["baseline"], dtype=np.float32)

    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        sys.exit("no camera")

    probs = np.full(len(LABELS), 1 / len(LABELS), dtype=np.float32)
    collecting = None        # a list while capturing a baseline, else None
    n, t0 = 0, time.time()
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            box = None
            if faces is not None and len(faces):
                # largest face is the subject; anyone behind them is not
                box = max(faces, key=lambda f: f[2] * f[3])[:4].astype(int)
                box = np.clip(box, 0, [w, h, w, h])

            if box is not None and n % ANALYZE_EVERY == 0:
                x, y, bw, bh = box
                crop = frame[y:y + bh, x:x + bw]
                if crop.size:
                    _, logits = recognizer.predict_emotions(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    raw = softmax(np.asarray(logits, dtype=np.float32))
                    probs = EMA * probs + (1 - EMA) * raw
                    if collecting is not None:
                        collecting.append(raw)

            shown = calibrate(probs, baseline)
            if box is not None:
                top = int(shown.argmax())
                draw(frame, box, shown, LABELS[top], float(shown[top]),
                     baseline is not None, BASELINE_FRAMES - len(collecting) if collecting is not None else 0)

            n += 1
            if n % 60 == 0:
                print(f"{60 / (time.time() - t0):.1f} fps", end="\r")
                t0 = time.time()

            cv2.imshow("mirror - phase 1", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("b"):
                collecting = []
            if key == ord("c"):
                baseline, collecting = None, None
                CALIB.unlink(missing_ok=True)
                print("baseline cleared")

            if collecting is not None and len(collecting) >= BASELINE_FRAMES:
                baseline = np.mean(collecting, axis=0)
                CALIB.write_text(json.dumps({"baseline": baseline.tolist(), "labels": LABELS}, indent=2))
                collecting = None
                print("baseline saved:", dict(zip(LABELS, baseline.round(3))))
    finally:
        cam.release()
        cv2.destroyAllWindows()


def _selftest():
    base = np.array([.3, .05, .05, .05, .1, .3, .1, .05], dtype=np.float32)
    raw = np.array([.3, .05, .05, .05, .4, .1, .0, .05], dtype=np.float32)
    out = calibrate(raw, base)
    assert abs(out.sum() - 1.0) < 1e-6, out.sum()
    assert LABELS[out.argmax()] == "Happiness", out          # was 'Anger'/'Neutral' tied before
    assert (out >= 0).all()
    assert calibrate(raw, None) is raw
    assert calibrate(base, base) is base                      # resting face falls back to raw
    assert calibrate(base + np.float32(1e-4), base) is not None
    print("ok")


if __name__ == "__main__":
    _selftest() if "--selftest" in sys.argv else main()
