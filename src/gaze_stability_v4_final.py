# %% [markdown]
# # Gaze Stability Component
#
# Measures % of time an interview applicant's gaze remained on-camera.
# Proxy for scripted-reading or off-camera coaching (per spec section 7.4).
#
# Approach: Haar Cascade face+eye detection (offline, no external model
# download needed - chosen after MediaPipe proved fragile in an earlier
# project) + pupil-centroid tracking, calibrated per-video against that
# video's own baseline gaze position AND its own noise floor.
#
# IMG_5033.MOV INVESTIGATION LOG (started at 62.3% on-camera, visually
# confirmed the person was looking at the camera almost the whole time):
#
# 1. (REVERTED) Picking the single largest dark contour instead of
#    averaging over the whole region, on the theory glasses glare
#    fragments the pupil. Regressed every video; ruled out by diagnostic
#    logging showing no real confidence difference glasses vs. no glasses.
#
# 2. (KEPT) EYE_CASCADE gives inconsistent box sizes frame-to-frame,
#    sometimes ballooning upward to include the eyebrow, pulling the
#    whole-region centroid up with no real eye movement. Fix: exclude
#    the top 30% of each eye box (EYEBROW_CROP_FRAC) before searching
#    for the dark pupil region. Took IMG_5033 to 88.4%.
#
# 3. (KEPT) The old fixed DEVIATION_THRESHOLD=0.08 sat almost exactly at
#    IMG_5033's own 90th percentile - not a principled cutoff, just
#    whatever flagged a plausible fraction of frames on whichever video
#    it was first tuned on. Replaced with a PER-VIDEO adaptive threshold:
#    median + K_MAD * scaled MAD (robust to the outliers we're trying to
#    detect). Took IMG_5033 to 92.8%, batch mean 95.9%, min 91.1%.
#
# 4. (TRIED, NOT SHIPPED) Visually checked the two largest remaining
#    outliers in IMG_5033 (deviation 0.219, 0.214) - BOTH are the person
#    blinking, not looking away. Hypothesized a "dark area fraction"
#    signal (closed eyelid should show less dark area than an open pupil)
#    to detect and exclude blinks. TESTED AGAINST THE ACTUAL BLINK/OPEN
#    IMAGES rather than synthetic data - the hypothesis did NOT hold:
#    the real blink's dark_area_fraction (0.296) was HIGHER than most
#    open-eye readings (0.06-0.27), likely because eyelash/lid-crease
#    shadow can be as dark as a real pupil. Dropped rather than ship an
#    unvalidated heuristic.
#
#    Good news found while testing: ONE of the two blinks (t=20.6s)
#    already produces zero detected eyes with the existing cascade, so
#    it's already correctly excluded today via the normal
#    "no eyes detected" path - no fix needed for that case. Only the
#    other blink (t=24.3s) slips through as a false flag - a small
#    residual (~1-2 frames out of 71 in this video), left as a known,
#    documented minor limitation rather than a blocking bug. A
#    size/aspect-ratio-based blink signal (closed-eye boxes trended
#    smaller in the real data: 25x25 vs 44-86px open) could be explored
#    later if this needs tightening further.

# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## Config

# %%
SAMPLE_RATE_PER_SECOND = 2
EYEBROW_CROP_FRAC = 0.30     # exclude top X% of each eye box (eyebrow-contamination fix)
K_MAD = 3.0                  # adaptive threshold = median + K_MAD * scaled_MAD
MAD_SCALE = 1.4826           # scales MAD to approximate a standard deviation

FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# %% [markdown]
# ## Core functions

# %%
def locate_pupil_center(eye_gray):
    """
    Find the pupil's center by locating the darkest region in the eye image
    and computing its centroid (whole-region average - proven more stable
    than picking a single "largest" contour, see log above).
    """
    h, w = eye_gray.shape
    blurred = cv2.GaussianBlur(eye_gray, (5, 5), 0)
    min_val = int(blurred.min())
    _, thresh = cv2.threshold(blurred, min_val + 30, 255, cv2.THRESH_BINARY_INV)

    M = cv2.moments(thresh)
    if M["m00"] == 0:
        return w / 2, h / 2

    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return cx, cy


def locate_pupil_center_cropped(eye_gray, crop_frac=EYEBROW_CROP_FRAC):
    """
    Same as locate_pupil_center(), but excludes the top crop_frac of the
    eye box first (avoids the eyebrow-contamination bug). Returns (cx, cy)
    mapped back to the ORIGINAL box's coordinate system.
    """
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
    if no face/eyes could be detected in this frame. Some blink frames are
    naturally excluded here already (eyes not detected at all when closed);
    a small residual of partial blinks may still get a (usually large,
    adaptively-thresholded-out) reading - see log above.
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
        cx, cy = locate_pupil_center_cropped(eye_img)
        h_ratios.append(cx / ew)
        v_ratios.append(cy / eh)

    return np.mean(h_ratios), np.mean(v_ratios)


def analyze_gaze_stability(video_path: str, k_mad: float = K_MAD) -> dict:
    """
    Analyze a video and return gaze stability results.

    Uses a PER-VIDEO adaptive off-camera threshold (median + k_mad * scaled
    MAD) rather than a fixed constant - see log above for why. Note: full
    blinks that produce zero detected eyes are already naturally excluded
    (counted in frames_face_not_detected); a small number of partial
    blinks may still register as a (usually large, threshold-filtered)
    reading - see log above, item 4, for a documented known limitation.

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

    baseline_h = np.median(h_ratios)
    baseline_v = np.median(v_ratios)
    deviation = np.sqrt((h_ratios - baseline_h) ** 2 + (v_ratios - baseline_v) ** 2)

    median_dev = np.median(deviation)
    mad = np.median(np.abs(deviation - median_dev))
    scaled_mad = mad * MAD_SCALE
    adaptive_threshold = median_dev + k_mad * scaled_mad

    off_camera_mask = deviation > adaptive_threshold
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
            "adaptive_threshold": adaptive_threshold,
        },
    }


# %% [markdown]
# ## Visualization helper

# %%
def plot_gaze_timeline(result: dict):
    debug = result["_debug"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(debug["timestamps"], debug["deviation"])
    ax.axhline(debug["adaptive_threshold"], color="red", linestyle="--",
               label=f"adaptive off-camera threshold ({debug['adaptive_threshold']:.3f})")
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
    print(f"Adaptive threshold used: {result['_debug']['adaptive_threshold']:.3f}")

    plot_gaze_timeline(result)

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
            result = analyze_gaze_stability(path)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on {fname}: {e}")

    return results


def summarize_batch(results):
    print(f"\n{'video':30s} | {'on-camera %':12s} | {'flagged':8s} | {'face missed':12s} | {'threshold':10s}")
    print("-" * 90)
    for r in results:
        name = os.path.basename(r["video"])
        pct = r["on_camera_percentage"]
        n_flagged = len(r["flagged_timestamps"])
        n_missed = r["frames_face_not_detected"]
        thresh = r.get("_debug", {}).get("adaptive_threshold")
        thresh_str = f"{thresh:.3f}" if thresh is not None else "n/a"
        print(f"{name:30s} | {str(pct):12s} | {n_flagged:8d} | {n_missed:12d} | {thresh_str:10s}")

    percentages = [r["on_camera_percentage"] for r in results if r["on_camera_percentage"] is not None]
    if percentages:
        print(f"\nAcross {len(percentages)} videos:")
        print(f"  mean on-camera %: {np.mean(percentages):.1f}")
        print(f"  min: {np.min(percentages):.1f}, max: {np.max(percentages):.1f}")


if __name__ == "__main__":
    batch_results = run_batch()
    summarize_batch(batch_results)

# %%
