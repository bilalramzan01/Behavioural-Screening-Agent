# %% [markdown]
# # Gaze Stability Component (Section 7.4, Component 1 - 40% of composite)
#
# Measures % of time an applicant's gaze remained on-camera during the loan
# video interview. Proxy for scripted-reading or off-camera coaching.
# Output feeds the Behavioural Consistency Signal -> Trust Score (Section 8.7)
# only; never the Credit Score. Advisory / continuous, not a hard threshold.
#
# Approach: Haar Cascade face+eye detection + pupil-centroid (darkness)
# tracking, calibrated per-video against that video's own median gaze
# position (raw "center" varies by camera angle / eye geometry per person).
#
# FIX LOG (validated against IMG_5033.MOV, a 33s clip with two real root
# causes previously conflated as one "glasses" bug):
#
# 1. EYEBROW-CROP (kept): EYE_CASCADE box sizes are unstable frame-to-frame
#    for some subjects, occasionally ballooning upward to include the
#    eyebrow. Since locate_pupil_center() averages over the WHOLE box, an
#    overshot box pulls the centroid toward the dark eyebrow, faking a large
#    vertical deviation. Fix: exclude the top EYEBROW_CROP_FRAC of each eye
#    box before darkness detection (pupils are essentially never there).
#    NOTE: an earlier "glasses-glare" hypothesis for the same symptom was
#    investigated via confidence-score diagnostics and ruled out - glasses-
#    video blob confidence was statistically indistinguishable from a clean
#    video's. Do not reintroduce a glasses-specific cascade fallback without
#    re-confirming that diagnostic first.
#
# 2. MOUTH-AS-EYE FILTER (new): EYE_CASCADE occasionally detects a third,
#    spurious box on the mouth/teeth region (confirmed visually - person
#    talking, mouth open). Real eye boxes sit at ey/face_h ~ 0.34-0.39;
#    the spurious mouth box sits at ey/face_h ~ 0.74-0.78. Averaging it in
#    drags h_ratio/v_ratio and produces a false off-camera flag even when
#    both real eyes are correctly on the pupil. Fix: discard any detected
#    "eye" box whose top edge falls below the face box's vertical midpoint.
#
# Combined effect on IMG_5033.MOV: 47.7% (no fix) -> 86.2% (crop only)
# -> 92.3% (crop + mouth filter). Remaining 5/65 flagged frames are a mix
# of one genuine off-camera moment (eyes closed, looking down) and smaller
# residual noise - not one dominant remaining cause.
#
# KNOWN OPEN ISSUE - NOT YET FIXED: single-core decode of 1080p HEVC .MOV
# took ~83s for a 33s clip in testing, vs. the spec's <30s budget for the
# ENTIRE 5-component pipeline. Needs testing on target hardware; if still
# too slow there, consider frame downscaling before cascade detection
# and/or hardware-accelerated decode before optimizing the CV logic itself.

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Config

# %%
SAMPLE_RATE_PER_SECOND = 2
DEVIATION_THRESHOLD = 0.08
EYEBROW_CROP_FRAC = 0.30
MAX_EYE_Y_FRAC = 0.5  # discard "eye" detections whose top sits below the face's vertical midpoint

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# %% [markdown]
# ## Core functions

# %%
def locate_pupil_center(eye_gray):
    """Whole-region darkness-centroid (proven more stable than single-largest-contour)."""
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
    """Excludes the top crop_frac of the eye box (likely eyebrow when the box overshoots)
    before searching, then maps the result back to the original box's coordinates."""
    h, w = eye_gray.shape
    crop_rows = int(h * crop_frac)
    search_region = eye_gray[crop_rows:, :]
    if search_region.size == 0:
        return w / 2, h / 2
    cx, cy_in_crop = locate_pupil_center(search_region)
    cy = cy_in_crop + crop_rows
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

    # Drop spurious detections below the face's vertical midpoint (mouth/teeth,
    # not an eye - see fix log above).
    eyes = [e for e in eyes if e[1] / h < MAX_EYE_Y_FRAC]
    if len(eyes) == 0:
        return None

    h_ratios, v_ratios = [], []
    for (ex, ey, ew, eh) in eyes:
        eye_img = face_roi[ey:ey+eh, ex:ex+ew]
        cx, cy = locate_pupil_center_cropped(eye_img)
        h_ratios.append(cx / ew)
        v_ratios.append(cy / eh)

    return np.mean(h_ratios), np.mean(v_ratios)


def analyze_gaze_stability(video_path: str, deviation_threshold: float = DEVIATION_THRESHOLD) -> dict:
    """
    Analyze a video and return gaze stability results, matching the shared
    component output format (component, video, on_camera_percentage,
    flagged_timestamps, total_frames_analyzed, frames_face_not_detected)
    used across all Section 7.4 screening-agent components.
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
        "_debug": {
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
# ## Validate across ALL videos

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
            results.append(analyze_gaze_stability(path))
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
