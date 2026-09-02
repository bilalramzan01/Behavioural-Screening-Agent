# %% [markdown]
# # Gaze Stability Component
#
# Measures % of time an interview applicant's gaze remained on-camera.
# Proxy for scripted-reading or off-camera coaching (per spec section 7.4).
#
# Approach: Haar Cascade face+eye detection (offline, no external model
# download needed - chosen after MediaPipe proved fragile in an earlier
# project) + pupil-centroid tracking, calibrated per-video against that
# video's own baseline gaze position (since raw ratio "center" varies by
# camera angle and individual eye geometry - a fixed absolute threshold
# across different videos would be wrong).

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Config

# %%
SAMPLE_RATE_PER_SECOND = 2   # how many frames per second to analyze (full fps not needed)
DEVIATION_THRESHOLD = 0.08   # tuned by visual inspection - see notebook history / README
                              # frames with deviation above this are flagged "off-camera"

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# %% [markdown]
# ## Core functions

# %%
def locate_pupil_center(eye_gray):
    """
    Find the pupil's center by locating the darkest region in the eye image
    (the pupil is reliably the darkest part of the eye) and computing its
    centroid. More robust than assuming the eye box itself is centered.
    """
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    min_val = int(blurred.min())
    _, thresh = cv2.threshold(blurred, min_val + 30, 255, cv2.THRESH_BINARY_INV)

    M = cv2.moments(thresh)
    if M["m00"] == 0:
        h, w = eye_gray.shape
        return w / 2, h / 2  # fallback: geometric center

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy


def get_gaze_ratio_for_frame(frame_bgr):
    """
    Detect face + eyes in a single frame and return the average horizontal/
    vertical pupil position ratio (0-1 within the detected eye box), or None
    if no face/eyes could be detected in this frame.
    """
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
        cx, cy = locate_pupil_center(eye_img)
        h_ratios.append(cx / ew)
        v_ratios.append(cy / eh)

    return np.mean(h_ratios), np.mean(v_ratios)


def analyze_gaze_stability(video_path: str, deviation_threshold: float = DEVIATION_THRESHOLD) -> dict:
    """
    Analyze a video and return gaze stability results.

    Returns a dict matching the shared component output format used across
    all screening-agent components:
        component, video, on_camera_percentage, flagged_timestamps,
        total_frames_analyzed, frames_face_not_detected
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(fps / SAMPLE_RATE_PER_SECOND))

    h_ratios, v_ratios, timestamps = [], [], []
    frames_not_detected = 0
    total_sampled = 0

    for frame_idx in range(0, frame_count, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        total_sampled += 1

        result = get_gaze_ratio_for_frame(frame)
        if result is None:
            frames_not_detected += 1
            continue

        h_ratio, v_ratio = result
        h_ratios.append(h_ratio)
        v_ratios.append(v_ratio)
        timestamps.append(frame_idx / fps)

    cap.release()

    h_ratios = np.array(h_ratios)
    v_ratios = np.array(v_ratios)
    timestamps = np.array(timestamps)

    if len(h_ratios) == 0:
        return {
            "component": "gaze_stability",
            "video": video_path,
            "on_camera_percentage": None,
            "flagged_timestamps": [],
            "total_frames_analyzed": total_sampled,
            "frames_face_not_detected": frames_not_detected,
            "error": "No valid gaze readings - face/eyes never detected.",
        }

    # Calibrate against this video's OWN baseline (median), since the "center"
    # position varies by camera angle and individual eye geometry - a fixed
    # absolute threshold across different videos would be wrong.
    baseline_h = np.median(h_ratios)
    baseline_v = np.median(v_ratios)
    deviation = np.sqrt((h_ratios - baseline_h) ** 2 + (v_ratios - baseline_v) ** 2)

    off_camera_mask = deviation > deviation_threshold
    on_camera_percentage = 100 * (1 - off_camera_mask.mean())
    flagged_timestamps = timestamps[off_camera_mask].round(1).tolist()

    return {
        "component": "gaze_stability",
        "video": video_path,
        "on_camera_percentage": round(float(on_camera_percentage), 1),
        "flagged_timestamps": flagged_timestamps,
        "total_frames_analyzed": total_sampled,
        "frames_face_not_detected": frames_not_detected,
        "_debug": {  # useful for tuning/visualization, not part of the "official" output
            "h_ratios": h_ratios,
            "v_ratios": v_ratios,
            "timestamps": timestamps,
            "deviation": deviation,
            "baseline_h": baseline_h,
            "baseline_v": baseline_v,
        },
    }


# %% [markdown]
# ## Visualization helper
#
# Not part of the production output - useful for tuning the threshold and
# sanity-checking results on new videos.

# %%
def plot_gaze_timeline(result: dict):
    debug = result["_debug"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(debug["timestamps"], debug["deviation"])
    ax.axhline(DEVIATION_THRESHOLD, color="red", linestyle="--", label="off-camera threshold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("deviation from baseline")
    ax.set_title(
        f"{os.path.basename(result['video'])} - "
        f"on-camera: {result['on_camera_percentage']}%"
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Run on a sample video

# %%
if __name__ == "__main__":
    video_path = "../data/raw_videos/IMG_5037.MOV"
    result = analyze_gaze_stability(video_path)

    print(f"On-camera percentage: {result['on_camera_percentage']}%")
    print(f"Flagged (off-camera) timestamps: {result['flagged_timestamps']}")
    print(f"Frames analyzed: {result['total_frames_analyzed']}")
    print(f"Frames face not detected: {result['frames_face_not_detected']}")

    plot_gaze_timeline(result)

# %%
