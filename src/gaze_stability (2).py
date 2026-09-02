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
#
# KNOWN LIMITATION (found during validation, see IMG_5033.MOV):
# the plain haarcascade_eye.xml + darkness-centroid approach produces
# false "off-camera" flags for people wearing glasses - lens glare
# breaks the pupil's dark region into scattered fragments rather than
# one solid blob, which the centroid math misreads as gaze deviation.
# Fix: fall back to haarcascade_eye_tree_eyeglasses.xml (a cascade
# trained specifically on eyes with glasses) whenever the standard
# path fails or produces a low-confidence (likely glare-confused)
# reading. Confidence = fraction of dark-thresholded pixels belonging
# to the single largest contiguous blob; a real pupil is one solid
# blob, glare is scattered speckle.
#
# This does not make gaze-based scoring perfectly fair across eyewear -
# it reduces a clear, measured bias, but flag this limitation in any
# spec/write-up that consumes this component's score.

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
PUPIL_CONFIDENCE_THRESHOLD = 0.5  # below this, retry eye location with the glasses cascade

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
EYE_GLASSES_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


# %% [markdown]
# ## Core functions

# %%
def locate_pupil_center(eye_gray, min_dark_pixels=15):
    """
    Find the pupil's center by locating the darkest region in the eye image
    (the pupil is reliably the darkest part of the eye) and computing its
    centroid. More robust than assuming the eye box itself is centered.

    Now also returns a confidence score: the fraction of all dark-
    thresholded pixels that belong to the single largest contiguous blob.
    A real pupil forms one solid blob (confidence near 1.0). Glasses-lens
    glare breaks the dark region into scattered fragments (confidence
    drops), which is what let glare masquerade as "gaze deviation" before -
    this lets callers detect that failure mode instead of trusting a bad
    centroid.
    """
    h, w = eye_gray.shape
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    min_val = int(blurred.min())
    _, thresh = cv2.threshold(blurred, min_val + 30, 255, cv2.THRESH_BINARY_INV)

    total_dark = cv2.countNonZero(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours or total_dark == 0:
        return w / 2, h / 2, 0.0  # fallback: geometric center, zero confidence

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
    """
    Run one cascade over the face ROI, compute pupil ratios + confidence
    for each detected eye box. Returns (h_ratios, v_ratios, mean_confidence)
    or None if the cascade found no eyes at all.
    """
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


def get_gaze_ratio_for_frame(frame_bgr, confidence_threshold=PUPIL_CONFIDENCE_THRESHOLD):
    """
    Detect face + eyes in a single frame and return the average horizontal/
    vertical pupil position ratio (0-1 within the detected eye box), or None
    if no face/eyes could be detected in this frame.

    Tries the standard eye cascade first. If it finds nothing, or its
    darkness-centroid confidence is low (glare/glasses likely confusing
    it), retries eye location with the glasses-trained cascade before
    giving up. This matters because glasses-wearers were previously
    getting misread as "off-camera" due to lens glare, not real gaze
    deviation (see IMG_5033.MOV in validation).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi = gray[y:y+h, x:x+w]

    standard = _eyes_and_confidence(face_roi, EYE_CASCADE)

    if standard is not None and standard[2] >= confidence_threshold:
        h_ratios, v_ratios, _ = standard
        return np.mean(h_ratios), np.mean(v_ratios)

    # Standard path failed or was low-confidence - retry eye location
    # with the glasses-trained cascade.
    glasses = _eyes_and_confidence(face_roi, EYE_GLASSES_CASCADE)

    if glasses is not None:
        h_ratios, v_ratios, _ = glasses
        return np.mean(h_ratios), np.mean(v_ratios)

    # Both cascades failed to find eyes at all - fall back to whatever
    # low-confidence standard reading we had, rather than nothing, since
    # a rough reading still beats discarding the frame entirely.
    if standard is not None:
        h_ratios, v_ratios, _ = standard
        return np.mean(h_ratios), np.mean(v_ratios)

    return None


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

# %% [markdown]
# ## Validate across ALL videos
#
# The threshold above was only tuned on one video. Before trusting it,
# check whether results look reasonable and consistent across the full
# dataset - not just one lucky/unlucky sample.

# %%
def run_batch(raw_videos_dir="../data/raw_videos"):
    video_files = [
        f for f in os.listdir(raw_videos_dir)
        if f.lower().endswith((".mp4", ".mov"))
    ]

    results = []
    for fname in sorted(video_files):
        path = os.path.join(raw_videos_dir, fname)
        print(f"Processing {fname}...")
        try:
            result = analyze_gaze_stability(path)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")

    return results


def summarize_batch(results):
    print(f"\n{'video':30s} | {'on-camera %':12s} | {'flagged':8s} | {'face missed':12s}")
    print("-" * 70)
    for r in results:
        name = os.path.basename(r["video"])
        pct = r["on_camera_percentage"]
        n_flagged = len(r["flagged_timestamps"])
        n_missed = r["frames_face_not_detected"]
        print(f"{name:30s} | {str(pct):12s} | {n_flagged:8d} | {n_missed:12d}")

    percentages = [r["on_camera_percentage"] for r in results if r["on_camera_percentage"] is not None]
    if percentages:
        print(f"\nAcross {len(percentages)} videos:")
        print(f"  mean on-camera %: {np.mean(percentages):.1f}")
        print(f"  min: {np.min(percentages):.1f}, max: {np.max(percentages):.1f}")


if __name__ == "__main__":
    batch_results = run_batch()
    summarize_batch(batch_results)

# %%
