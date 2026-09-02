"""
Standalone diagnostic - run directly, no notebook/cell dependency:

    python diagnostic_standalone.py

Defines everything it needs itself (cascades, functions) so it doesn't
rely on any other file's cells having been run first. Prints confidence
distributions for IMG_5038 (clean baseline) and IMG_5033 (glasses) so we
can pick a real threshold instead of guessing.

Adjust RAW_VIDEOS_DIR below if your paths differ.
"""

import os
from collections import Counter

import cv2
import numpy as np

RAW_VIDEOS_DIR = "../data/raw_videos"
SAMPLE_RATE_PER_SECOND = 2
PUPIL_CONFIDENCE_THRESHOLD = 0.5

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
EYE_GLASSES_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

_diagnostic_log = {"standard_confidences": [], "path_used": []}


def locate_pupil_center(eye_gray, min_dark_pixels=15):
    h, w = eye_gray.shape
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    min_val = int(blurred.min())
    _, thresh = cv2.threshold(blurred, min_val + 30, 255, cv2.THRESH_BINARY_INV)

    total_dark = cv2.countNonZero(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or total_dark == 0:
        return w / 2, h / 2, 0.0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_dark_pixels:
        return w / 2, h / 2, 0.0

    confidence = area / total_dark
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return w / 2, h / 2, 0.0

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy, confidence


def _eyes_and_confidence(face_roi, cascade):
    eyes = cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)
    if len(eyes) == 0:
        return None
    h_ratios, v_ratios, confidences = [], [], []
    for (ex, ey, ew, eh) in eyes:
        eye_img = face_roi[ey:ey+eh, ex:ex+ew]
        cx, cy, conf = locate_pupil_center(eye_img)
        h_ratios.append(cx / ew)
        v_ratios.append(cy / eh)
        confidences.append(conf)
    return h_ratios, v_ratios, float(np.mean(confidences))


def get_gaze_ratio_for_frame_DIAGNOSTIC(frame_bgr, confidence_threshold=PUPIL_CONFIDENCE_THRESHOLD):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        _diagnostic_log["path_used"].append("no_face")
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]

    standard = _eyes_and_confidence(face_roi, EYE_CASCADE)
    _diagnostic_log["standard_confidences"].append(standard[2] if standard is not None else None)

    if standard is not None and standard[2] >= confidence_threshold:
        _diagnostic_log["path_used"].append("standard")
        h_ratios, v_ratios, _ = standard
        return np.mean(h_ratios), np.mean(v_ratios)

    glasses = _eyes_and_confidence(face_roi, EYE_GLASSES_CASCADE)
    if glasses is not None:
        _diagnostic_log["path_used"].append("glasses_fallback")
        h_ratios, v_ratios, _ = glasses
        return np.mean(h_ratios), np.mean(v_ratios)

    if standard is not None:
        _diagnostic_log["path_used"].append("standard_low_confidence")
        h_ratios, v_ratios, _ = standard
        return np.mean(h_ratios), np.mean(v_ratios)

    _diagnostic_log["path_used"].append("failed")
    return None


def run_diagnostic(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"COULD NOT OPEN: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / SAMPLE_RATE_PER_SECOND))

    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        get_gaze_ratio_for_frame_DIAGNOSTIC(frame)

    cap.release()


def print_diagnostic_summary(video_name):
    confs = [c for c in _diagnostic_log["standard_confidences"] if c is not None]
    print(f"\n=== {video_name} ===")
    print(f"frames where standard cascade found eyes at all: {len(confs)} / {len(_diagnostic_log['path_used'])}")
    if confs:
        confs_arr = np.array(confs)
        print(f"standard confidence: min={confs_arr.min():.3f} "
              f"median={np.median(confs_arr):.3f} max={confs_arr.max():.3f}")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5]:
            pct = 100 * (confs_arr >= t).mean()
            print(f"  % of frames with confidence >= {t}: {pct:.0f}%")
    print("path_used counts:", dict(Counter(_diagnostic_log["path_used"])))


def reset_diagnostic_log():
    _diagnostic_log["standard_confidences"] = []
    _diagnostic_log["path_used"] = []


if __name__ == "__main__":
    reset_diagnostic_log()
    run_diagnostic(os.path.join(RAW_VIDEOS_DIR, "IMG_5038.MOV"))
    print_diagnostic_summary("IMG_5038 (clean baseline)")

    reset_diagnostic_log()
    run_diagnostic(os.path.join(RAW_VIDEOS_DIR, "IMG_5033.MOV"))
    print_diagnostic_summary("IMG_5033 (glasses)")
