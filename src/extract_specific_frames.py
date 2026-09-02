"""
Extract single frames at specific timestamps for visual inspection -
just the two genuine outliers from the deviation distribution (20.6s,
24.3s), plus their annotated eye detection, saved as images.

Run:
    python extract_specific_frames.py
"""

import os
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(SCRIPT_DIR, "..", "data", "raw_videos", "IMG_5033.MOV")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "outputs", "outlier_check_IMG_5033")

TARGET_TIMESTAMPS = [20.6, 24.3]

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
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def locate_pupil_center_cropped(eye_gray, crop_frac=0.30):
    h, w = eye_gray.shape
    crop_rows = int(h * crop_frac)
    region = eye_gray[crop_rows:, :]
    if region.size == 0:
        return w / 2, h / 2
    cx, cy = locate_pupil_center(region)
    return cx, cy + crop_rows


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"COULD NOT OPEN: {VIDEO_PATH}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for ts in TARGET_TIMESTAMPS:
        frame_idx = int(round(ts * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame at t={ts}s")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            print(f"t={ts}s: NO FACE DETECTED")
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"outlier_t{ts}s_NOFACE.jpg"), frame)
            continue

        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        face_roi = gray[y:y+h, x:x+w]
        eyes = EYE_CASCADE.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5)

        if len(eyes) == 0:
            print(f"t={ts}s: face found but NO EYES DETECTED")
        for (ex, ey, ew, eh) in eyes:
            eye_img = face_roi[ey:ey+eh, ex:ex+ew]
            cx, cy = locate_pupil_center_cropped(eye_img)
            cv2.rectangle(frame, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (255, 200, 0), 1)
            crop_y = y + ey + int(eh * 0.30)
            cv2.line(frame, (x+ex, crop_y), (x+ex+ew, crop_y), (255, 0, 255), 1)
            cv2.circle(frame, (x+ex+int(cx), y+ey+int(cy)), 4, (0, 0, 255), -1)

        label = f"t={ts}s  eyes_detected={len(eyes)}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        out_path = os.path.join(OUTPUT_DIR, f"outlier_t{ts}s.jpg")
        cv2.imwrite(out_path, frame)
        print(f"saved {out_path}")

    cap.release()


if __name__ == "__main__":
    main()
