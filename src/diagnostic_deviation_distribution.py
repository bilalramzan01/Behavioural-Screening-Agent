"""
Standalone diagnostic - inspect the deviation distribution for IMG_5033
using the eyebrow-cropped pupil detection, to check whether the
remaining flagged frames are borderline (threshold recalibration would
fix them) or genuine outliers (a different problem remains).

Run directly, no notebook/cell dependency:

    python diagnostic_deviation_distribution.py
"""

import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw_videos", "IMG_5033.MOV")

SAMPLE_RATE_PER_SECOND = 2
EYEBROW_CROP_FRAC = 0.30
CURRENT_THRESHOLD = 0.08

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def locate_pupil_center(eye_gray):
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    min_val = int(blurred.min())
    _, thresh = cv2.threshold(blurred, min_val + 30, 255, cv2.THRESH_BINARY_INV)
    M = cv2.moments(thresh)
    if M["m00"] == 0:
        h, w = eye_gray.shape
        return w / 2, h / 2
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy


def locate_pupil_center_cropped(eye_gray, crop_frac=EYEBROW_CROP_FRAC):
    h, w = eye_gray.shape
    crop_rows = int(h * crop_frac)
    search_region = eye_gray[crop_rows:, :]
    if search_region.size == 0:
        return w / 2, h / 2
    cx, cy_in_crop = locate_pupil_center(search_region)
    return cx, cy_in_crop + crop_rows


def get_gaze_ratio_for_frame(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]
    eyes = EYE_CASCADE.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)
    if len(eyes) == 0:
        return None
    h_ratios, v_ratios = [], []
    for (ex, ey, ew, eh) in eyes:
        eye_img = face_roi[ey:ey+eh, ex:ex+ew]
        cx, cy = locate_pupil_center_cropped(eye_img)
        h_ratios.append(cx / ew)
        v_ratios.append(cy / eh)
    return np.mean(h_ratios), np.mean(v_ratios)


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"COULD NOT OPEN: {VIDEO_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / SAMPLE_RATE_PER_SECOND))

    h_ratios, v_ratios, timestamps = [], [], []

    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        result = get_gaze_ratio_for_frame(frame)
        if result is None:
            continue
        h_ratio, v_ratio = result
        h_ratios.append(h_ratio)
        v_ratios.append(v_ratio)
        timestamps.append(frame_idx / fps)

    cap.release()

    h_ratios = np.array(h_ratios)
    v_ratios = np.array(v_ratios)
    timestamps = np.array(timestamps)

    baseline_h = np.median(h_ratios)
    baseline_v = np.median(v_ratios)
    deviation = np.sqrt((h_ratios - baseline_h) ** 2 + (v_ratios - baseline_v) ** 2)

    order = np.argsort(-deviation)  # largest first
    print(f"{len(deviation)} frames analyzed. baseline_h={baseline_h:.3f} baseline_v={baseline_v:.3f}\n")
    print(f"{'timestamp':>10s} | {'deviation':>10s} | flagged @ {CURRENT_THRESHOLD}")
    print("-" * 45)
    for i in order:
        flag = "  <-- FLAGGED" if deviation[i] > CURRENT_THRESHOLD else ""
        print(f"{timestamps[i]:10.1f} | {deviation[i]:10.3f} |{flag}")

    print("\n--- how many frames would be flagged under different thresholds ---")
    for t in [0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        n_flagged = int((deviation > t).sum())
        pct_on_camera = 100 * (1 - n_flagged / len(deviation))
        print(f"  threshold={t:.2f}: {n_flagged:3d} flagged -> on-camera {pct_on_camera:.1f}%")

    print(f"\ndeviation stats: min={deviation.min():.3f} median={np.median(deviation):.3f} "
          f"p90={np.percentile(deviation, 90):.3f} max={deviation.max():.3f}")


if __name__ == "__main__":
    main()
