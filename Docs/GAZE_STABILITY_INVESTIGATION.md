## Gaze Stability Component — Investigation & Fix Log

### Background
The Gaze Stability component estimates the percentage of time an applicant's
gaze stayed on-camera during an interview video, as a proxy signal for
scripted-reading or off-camera coaching (spec section 7.4). It works by:
detecting the face and eyes with Haar cascades, locating the pupil as the
darkest point in each eye region, and comparing the pupil's position
frame-to-frame against that video's own baseline position.

### The problem
During validation across 15 videos, one video (IMG_5033.MOV) scored a
suspiciously low **62.3% on-camera**, far below the ~90% average of the
other videos. Manually reviewing the video confirmed the applicant was
looking at the camera almost the entire time — this was a false positive
in the model, not real off-camera behavior.

### Investigation timeline

**1. Hypothesis: glasses glare (rejected)**
The applicant in IMG_5033 wears glasses. Hypothesized that lens glare was
fragmenting the pupil's dark region and confusing detection. Attempted fix:
pick the single largest dark contour instead of averaging over the whole
eye region, with a glasses-specific Haar cascade as a fallback.
**Result:** regressed every single video, not just glasses ones — mean
on-camera % dropped from 89.7% to 78.6%. Diagnostic logging also showed the
glasses video's detection confidence was statistically indistinguishable
from a clean video's. Hypothesis rejected; change fully reverted.

**2. Real bug found: eyebrow contamination (fixed)**
Visual inspection — overlaying the computed pupil position directly on
flagged frames — showed the actual problem: the eye-detection cascade
produced inconsistent box sizes frame-to-frame, occasionally expanding
upward to include the eyebrow. Since pupil position was computed as an
average over the *entire* detected box, an oversized box let the dark
eyebrow pull the computed position upward, registering as a large fake
"gaze deviation" with no real eye movement.
**Fix:** exclude the top 30% of each detected eye box before running
pupil detection.
**Result:** IMG_5033 improved from 62.3% → 88.4%, with other videos
holding steady or improving slightly.

**3. Threshold was never principled (fixed)**
Logging the full deviation distribution for IMG_5033 showed no clean
separation between "noise" and "real deviation" — values decayed smoothly.
The original fixed threshold (0.08) happened to sit almost exactly at this
video's own 90th percentile, suggesting it was never a validated cutoff,
just a number that produced plausible-looking results on whichever video
it was first tuned on. A fixed constant also assumes every person has the
same amount of natural micro-movement, which isn't true.
**Fix:** replaced the fixed threshold with a per-video adaptive threshold:
`median + 3 × scaled MAD` (median absolute deviation), which sets each
video's cutoff from its own noise floor rather than a shared constant.
**Result:** IMG_5033 improved to 92.8%. Full batch mean rose to 95.9%
(min 91.1%, max 100.0%), with no regressions.

**4. Blink false-positive (identified, not fully fixed — documented limitation)**
The two largest remaining outlier frames in IMG_5033 were visually
confirmed to be the applicant blinking, not looking away — an involuntary,
universal behavior irrelevant to attentiveness. Hypothesized that a closed
eyelid would show less "dark area" than an open pupil, and tried excluding
low-dark-area frames as likely blinks.
**Result when tested against the real blink images:** hypothesis did not
hold — the real blink's dark-area measurement was *higher* than most
open-eye frames, not lower (likely because eyelash/eyelid-crease shadow
can be as dark as a pupil). The fix was dropped rather than shipped
unvalidated.
**Silver lining found while testing:** one of the two blinks already
produces zero detected eyes with the existing cascade, so it's already
correctly excluded today with no extra fix needed. Only the second blink
slips through as a false flag — a small residual affecting ~1 frame out
of 71 sampled frames in that one video. Documented as a known minor
limitation; a box-size/aspect-ratio-based blink signal is a possible
future improvement (closed-eye boxes trended smaller in the real data:
25×25px vs 44–86px for open eyes) but was not pursued further given the
small impact.

### Final result
| Metric | Before | After |
|---|---|---|
| IMG_5033 on-camera % | 62.3% | 92.8% |
| Batch mean (15 videos) | 89.7% | 95.9% |
| Batch min | 62.3% | 91.1% |
| Batch max | 98.2% | 100.0% |

### Known limitations (honest, for viva/report)
- No manually labeled ground truth exists yet, so no formal accuracy/
  precision/recall figure can be claimed — the percentages above are the
  model's *output score*, not a validated accuracy against human judgment.
- A small number of partial blinks (~1 per video, roughly) may still
  register as a false off-camera flag.
- Haar cascades are an older, less accurate detection method than modern
  deep-learning face/eye landmark models (e.g. MediaPipe) — chosen here
  for offline use with no external model download, at some accuracy cost.
- The 2-fps sampling rate is a practical speed/thoroughness trade-off, not
  independently validated; very brief glances away (under ~0.5s) could
  fall between sampled frames and go undetected.

### Suggested next step
Manually label a sample of frames (e.g. 100–150) as ground truth
("on-camera" / "off-camera"), then compute real precision/recall/accuracy
against the model's classifications for a defensible accuracy figure.
