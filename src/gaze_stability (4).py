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
# FIX LOG:
#
# IMG_5033.MOV (33s, glasses-wearer, 47.7% on-camera vs ~90% typical):
# 1. EYEBROW-CROP: EYE_CASCADE box sizes are unstable frame-to-frame for
#    some subjects, occasionally ballooning upward to include the eyebrow.
#    Averaging the WHOLE box pulls the centroid toward the dark eyebrow,
#    faking vertical deviation. Fix: exclude the top EYEBROW_CROP_FRAC of
#    each eye box before darkness detection.
#    (An earlier "glasses-glare fragments the pupil" hypothesis for this
#    same video was investigated via confidence-score diagnostics and
#    ruled out - blob confidence was statistically indistinguishable from
#    a clean video's. That is a different mechanism from the side-crop fix
#    below, which is about a competing dark object, not fragmentation.)
# 2. MOUTH-AS-EYE FILTER: EYE_CASCADE occasionally detects a spurious third
#    box on the mouth/teeth (confirmed visually). Real eye boxes sit at
#    ey/face_h ~ 0.34-0.39; the spurious mouth box sits at ~0.74-0.78.
#    Fix: discard any eye detection whose box top falls below the face's
#    vertical midpoint (MAX_EYE_Y_FRAC).
# Combined effect: 47.7% -> 86.2% (crop only) -> 92.3% (crop + mouth filter).
#
# IMG_5030.MOV (long hairstyle + glasses, 62.7% / then 71.7% on-camera):
# 3. SIDE-CROP (new): visually confirmed a genuine glasses-rim/temple-hinge
#    artifact - the dark frame edge inside the eye box can be darker than
#    the actual (glare-washed) pupil, pulling the centroid toward the rim
#    instead of the eye. Verified directly: at one flagged frame (subject
#    visibly looking at camera) deviation dropped 0.217 -> 0.089 once the
#    outer SIDE_CROP_FRAC of each eye box's left/right edges was excluded
#    from the darkness search, same logic as the eyebrow crop but applied
#    to the sides. A competing "require both eyes detected" fix was also
#    tested and rejected: it barely moved on-camera% (71.7 -> 73.8) while
#    tripling the no-signal frame count (2 -> 20 of 60) - discarding data
#    instead of fixing the actual measurement.
# Effect on IMG_5030: 71.7% -> 90.0% on-camera, flagged 17 -> 6.
# Re-verified IMG_5033 is unaffected by this change: still 92.3% / 5 flagged.

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
EYEBROW_CROP_FRAC = 0.30   # fraction excluded from the TOP of each eye box
SIDE_CROP_FRAC = 0.15      # fraction excluded from EACH of the left/right edges
MAX_EYE_Y_FRAC = 0.5       # discard "eye" detections whose top sits below the
                            # face's vertical midpoint (mouth/teeth, not an eye)

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# %% [markdown]
# ## Core functions

# %%
def locate_pupil_center(eye_gray):
    """Whole-region darkness-centroid (proven more stable than picking a
    single "largest" contour - see fix log above)."""
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


def locate_pupil_center_boxed(eye_gray, top_frac=EYEBROW_CROP_FRAC, side_frac=SIDE_CROP_FRAC):
    """
    Excludes the top top_frac (likely eyebrow when the cascade box
    overshoots upward) and the outer side_frac of each side (glasses
    rim / temple hinge - a dark object that can outcompete a glare-washed
    pupil) before searching for the darkest region. Returns (cx, cy)
    mapped back into the ORIGINAL (uncropped) eye_gray's coordinates.
    """
    h, w = eye_gray.shape
    top_rows = int(h * top_frac)
    side_cols = int(w * side_frac)
    region = eye_gray[top_rows:, side_cols:w - side_cols]

    if region.size == 0:
        return w / 2, h / 2

    cx_in_region, cy_in_region = locate_pupil_center(region)
    cx = cx_in_region + side_cols
    cy = cy_in_region + top_rows
    return cx, cy


def _filter_mouth_detections(eyes, face_h, max_y_frac=MAX_EYE_Y_FRAC):
    """Drop spurious eye-cascade detections that are actually the mouth/teeth."""
    return [e for e in eyes if e[1] / face_h < max_y_frac]


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
    eyes = _filter_mouth_detections(eyes, h)
    if len(eyes) == 0:
        return None

    h_ratios, v_ratios = [], []
    for (ex, ey, ew, eh) in eyes:
        eye_img = face_roi[ey:ey+eh, ex:ex+ew]
        cx, cy = locate_pupil_center_boxed(eye_img)
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
# ## Run on a sample video

# %%
if __name__ == "__main__":
    video_path = "../data/raw_videos/IMG_5037.MOV"
    result = analyze_gaze_stability(video_path)

    print(f"On-camera percentage: {result['on_camera_percentage']}%")
    print(f"Flagged (off-camera) timestamps: {result['flagged_timestamps']}")
    print(f"Frames analyzed: {result['total_frames_analyzed']}")
    print(f"Frames face not detected: {result['frames_face_not_detected']}")


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
