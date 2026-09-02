"""
Visualize what the ORIGINAL (reverted) algorithm actually sees on the
flagged frames of a given video - draws the face box, eye boxes, and
computed pupil dot directly onto each flagged frame, and saves them as
images so we can inspect what's really happening instead of guessing.

Run:
    python visualize_flagged_frames.py

Adjust VIDEO_PATH and OUTPUT_DIR below if needed. Saves one annotated
JPEG per flagged frame into OUTPUT_DIR.
"""

import os
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw_videos", "IMG_5033.MOV")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "flagged_frames_IMG_5033_v2_cropped")

SAMPLE_RATE_PER_SECOND = 2
DEVIATION_THRESHOLD = 0.08
MAX_FRAMES_TO_SAVE = 15  # don't dump hundreds of images, just a representative set

EYEBROW_CROP_FRAC = 0.30

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


def locate_pupil_center(eye_gray):
    """Whole-region moments - proven stable baseline version."""
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
    """Excludes the top crop_frac of the box (likely eyebrow) before searching."""
    h, w = eye_gray.shape
    crop_rows = int(h * crop_frac)
    search_region = eye_gray[crop_rows:, :]
    if search_region.size == 0:
        return w / 2, h / 2
    cx, cy_in_crop = locate_pupil_center(search_region)
    cy = cy_in_crop + crop_rows
    return cx, cy


def analyze_and_collect(video_path):
    """
    Re-runs the same analysis as gaze_stability.py, but also keeps the
    raw frame + detected boxes + pupil points for every SAMPLED frame,
    so we can go back and annotate/save the flagged ones afterward.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / SAMPLE_RATE_PER_SECOND))

    records = []  # one dict per successfully-analyzed sampled frame

    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face_roi = gray[y:y+h, x:x+w]
        eyes = EYE_CASCADE.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) == 0:
            continue

        h_ratios, v_ratios, eye_boxes, pupil_points = [], [], [], []
        for (ex, ey, ew, eh) in eyes:
            eye_img = face_roi[ey:ey+eh, ex:ex+ew]
            cx, cy = locate_pupil_center_cropped(eye_img)
            h_ratios.append(cx / ew)
            v_ratios.append(cy / eh)
            eye_boxes.append((ex, ey, ew, eh))
            # pupil point in full-frame coordinates
            pupil_points.append((x + ex + int(cx), y + ey + int(cy)))

        records.append({
            "frame_idx": frame_idx,
            "timestamp": frame_idx / fps,
            "frame": frame,
            "face_box": (x, y, w, h),
            "eye_boxes": eye_boxes,
            "pupil_points": pupil_points,
            "h_ratio": np.mean(h_ratios),
            "v_ratio": np.mean(v_ratios),
        })

    cap.release()
    return records


def annotate_frame(record):
    frame = record["frame"].copy()
    x, y, w, h = record["face_box"]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (ex, ey, ew, eh), (px, py) in zip(record["eye_boxes"], record["pupil_points"]):
        cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 200, 0), 1)
        # magenta line = eyebrow-crop boundary; search only happens BELOW this line
        crop_y = y + ey + int(eh * EYEBROW_CROP_FRAC)
        cv2.line(frame, (x+ex, crop_y), (x+ex+ew, crop_y), (255, 0, 255), 1)
        cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)  # pupil point in red

    label = f"t={record['timestamp']:.1f}s  h={record['h_ratio']:.2f} v={record['v_ratio']:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return frame


def main():
    print(f"Analyzing {VIDEO_PATH} ...")
    records = analyze_and_collect(VIDEO_PATH)
    if not records:
        print("No valid frames found - check VIDEO_PATH.")
        return

    h_ratios = np.array([r["h_ratio"] for r in records])
    v_ratios = np.array([r["v_ratio"] for r in records])
    baseline_h = np.median(h_ratios)
    baseline_v = np.median(v_ratios)
    deviation = np.sqrt((h_ratios - baseline_h) ** 2 + (v_ratios - baseline_v) ** 2)

    flagged_idx = np.where(deviation > DEVIATION_THRESHOLD)[0]
    on_camera_pct = 100 * (1 - len(flagged_idx) / len(records))
    print(f"On-camera %: {on_camera_pct:.1f}  |  flagged frames: {len(flagged_idx)} / {len(records)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save a representative sample of flagged frames, spread across the video
    if len(flagged_idx) > MAX_FRAMES_TO_SAVE:
        pick = np.linspace(0, len(flagged_idx) - 1, MAX_FRAMES_TO_SAVE).astype(int)
        flagged_idx = flagged_idx[pick]

    for i in flagged_idx:
        record = records[i]
        annotated = annotate_frame(record)
        out_path = os.path.join(OUTPUT_DIR, f"flagged_t{record['timestamp']:.1f}s.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"  saved {out_path}  (deviation={deviation[i]:.3f})")

    # Also save a few NON-flagged frames for comparison
    non_flagged_idx = np.where(deviation <= DEVIATION_THRESHOLD)[0]
    pick_n = np.linspace(0, len(non_flagged_idx) - 1, min(5, len(non_flagged_idx))).astype(int)
    for i in non_flagged_idx[pick_n]:
        record = records[i]
        annotated = annotate_frame(record)
        out_path = os.path.join(OUTPUT_DIR, f"NORMAL_t{record['timestamp']:.1f}s.jpg")
        cv2.imwrite(out_path, annotated)
        print(f"  saved {out_path}  (deviation={deviation[i]:.3f}, for comparison)")

    print(f"\nDone. Open the images in {OUTPUT_DIR} and check: does the red dot "
          f"actually land on the pupil in the flagged frames? Or has it drifted "
          f"onto an eyebrow/eyelid/glare spot instead?")


if __name__ == "__main__":
    main()
