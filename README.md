# Tyrell -- 1-on-1 WR/CB Tracking & Grading (MVP)

Proof-of-concept pipeline for breaking down 1-on-1 WR vs. CB reps from
video: detect and track both athletes, classify the route and coverage,
grade the rep, and hand back drills + pros/cons -- the same idea as the
"analyzing film with computer vision" clips floating around, built out
into something you can actually run on your own footage.

**Where this is at right now:** the tracking + assisted-identification
loop works end to end (see Status below). Route/coverage classification
and grading are rule-based heuristics, not trained models -- accurate
enough to prove the pipeline out, not yet a scouting-grade grade. Real
pixel-to-yard calibration is now a proper homography (see Calibration
below) instead of a single guessed ratio.

## How it works

1. **Detect + track** (`src/track.py`) -- YOLOv8 (pretrained, no training
   needed) finds every person in the clip; **BoT-SORT** (default tracker)
   gives each one a persistent track ID (TID) frame to frame, holding IDs
   together through contact/occlusion better than plain ByteTrack (still
   available via `--tracker bytetrack` to compare).
2. **Identify** (`src/identity.py`, `src/jersey_ocr.py`) -- true automatic
   "who is this athlete" recognition isn't reliable on typical drill or
   broadcast footage without a roster-specific trained model, so a human
   confirms which TID is the WR and which is the CB once per clip (see
   `discover` below). That assignment (PID) is what shows up as "WR" /
   "CB" / a real name in the output, and survives even if the raw tracker
   re-assigns a new TID mid-clip. `discover --ocr-jerseys` will *suggest* a
   jersey number per track using OCR to speed up tagging -- treat it as a
   hint, not an answer (see the caveat in `jersey_ocr.py`).
3. **Pose** (`src/pose_features.py`) -- MediaPipe Pose runs on each
   tracked player's crop for movement/posture features (footwork, hip
   turn, backpedal posture).
4. **Calibration** (`src/calibration.py`) -- maps tracked pixel positions
   to real-world yards/feet via a homography built from 4+ known
   field/court points (`calibrate` subcommand), so speed/separation/route
   depth come out as real units instead of guesses, and perspective
   (far-field pixels covering more ground than near-field ones) is
   accounted for. A single-ratio fallback (`schema.Calibration`) still
   exists for when you only know one distance.
5. **Metrics + classification** (`src/metrics.py`) -- from the tracked
   trajectory: speed, separation, break point/angle, a heuristic route
   guess (slant/out/post/corner/go/etc.), and a heuristic coverage guess
   (press/off, man/zone via mirror correlation).
6. **Grading** (`src/grading.py`) -- football-specific: turns those
   metrics into a 0-100 grade + letter grade, pros/cons bullets, and
   matched drills from `data/drills/drills_library.json`.
7. **Output** (`src/annotate.py`, `src/report.py`) -- an annotated MP4
   (boxes, PID/TID labels, route trail) and a report (JSON + a
   self-contained HTML report card).

Down/distance, hash, formation, and coverage call are **manually supplied**
situational context (`--down`, `--distance`, etc.) -- not derived from the
video. Reliable OCR of a scoreboard/broadcast graphic is a separate,
broadcast-specific problem and out of scope for this MVP.

## Setup

```bash
pip install -r requirements.txt
./scripts/download_models.sh   # fetches the MediaPipe pose model (~6MB)
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run.
`easyocr` (jersey OCR) and `pandas` (Big Data Bowl validation) are optional
-- skip them if you don't need `--ocr-jerseys` or `scripts/validate_against_bdb.py`.

On a minimal Linux box you may also need system GL libs MediaPipe's
runtime links against even for CPU-only inference:
```bash
apt-get install -y libegl1 libgles2
```

## Usage

**Step 1 -- discover who's in the clip:**
```bash
python run_pipeline.py discover --video clip.mp4 --out out/clip1
# add --ocr-jerseys for experimental jersey-number hints
```
This prints a table of every tracked person (TID, how many frames they
were tracked for) and saves a thumbnail per track to `out/clip1/thumbs/`
so you can look and see which TID is the WR and which is the CB.

**Step 2 -- calibrate (optional, but worth it):**
```bash
python run_pipeline.py calibrate --video clip.mp4 --out out/clip1 --frame 0
```
Exports `out/clip1/calibration_frame.png` and a
`out/clip1/calibration_points.json` template pre-filled with real-world
coordinates for 4 football (or `--sport basketball` court) reference
points. Open the frame in any image viewer that shows cursor pixel
position, find where each point actually lands, and fill in the `"px"`
fields. More points (not just 4) make for a better-conditioned homography
if you can identify them -- yard-line/hash intersections across the depth
of the shot work well.

*Caveat:* this assumes a fixed camera (sideline tripod, drill cam). A
broadcast camera that pans/zooms mid-play breaks a single homography --
that needs per-frame calibration, which isn't built yet.

**Step 3 -- grade the rep:**
```bash
python run_pipeline.py grade --video clip.mp4 --out out/clip1 \
    --wr-track-id 3 --wr-name "J. Smith" \
    --cb-track-id 5 --cb-name "A. Jones" \
    --down 3 --distance 7 --formation "11 personnel, trips right" \
    --coverage-call "Cover 1" --level college \
    --calibration-points out/clip1/calibration_points.json
```
No calibration file? Pass `--calibration-px`/`--calibration-yards` for the
old single-ratio fallback, or skip both -- the pipeline still runs, it'll
just leave speed/separation as `null` instead of guessing.

Output lands in `out/clip1/`: `annotated.mp4`, `report.json`, `report.html`
(open the HTML file in a browser -- it embeds the annotated clip next to
both grade cards).

## Basketball

Detection, tracking, pose, and calibration (`discover`, `calibrate --sport
basketball`) are sport-agnostic and work today -- swap in NBA/NCAA court
dimensions instead of field dimensions. **`grade` is football-only right
now** (`--sport basketball` fails fast with an explanation): "break
sharpness" and "separation created" don't mean the same thing for a
basketball cut, and a grading rubric built without knowing what should
actually be scored (a cut vs. a closeout vs. a screen read, defensive
stance, etc.) would just be invented numbers. Tell me what a basketball
rep should be graded on and `grade_offense_bball()`/`grade_defense_bball()`
in `src/grading.py` is a straightforward addition alongside the existing
WR/CB rubric.

## Validating against real data

`scripts/validate_against_bdb.py` scores `metrics.classify_route()`
against labeled routes from the **NFL Big Data Bowl** (Kaggle) tracking
data -- real NFL player tracking + play-by-play route/coverage labels,
the best available ground truth for sanity-checking the heuristics without
needing our own large labeled video set. This sandbox has no route to
kaggle.com, so you'll need to download it yourself (free Kaggle account):
see the script's docstring for the exact steps and expected file layout.

## Status / what's proven vs. what's next

**Proven (this MVP):** YOLO+BoT-SORT reliably tracks both athletes in a
1-on-1 rep with persistent boxes/IDs through cuts, occlusion, and camera
motion; MediaPipe pose extraction runs per-track; the assisted PID
workflow (discover -> calibrate -> grade) produces a full annotated
video + report end to end; calibration is a real homography, not a
single guessed ratio.

**Heuristic, not trained (next steps if this direction is worth
pursuing):**
- Route/coverage classification is rule-based on trajectory shape. A
  trained classifier (given a labeled set of real reps) would be far more
  accurate, especially for route families that look similar from a side
  angle. `scripts/validate_against_bdb.py` is the honest way to find out
  how good the current heuristic actually is.
- Grading weights are a starting rubric, not calibrated against real
  coaching grades -- worth tuning once there's a set of reps a human coach
  has also graded, to compare against.
- No automatic down/distance/situation reading from broadcast graphics.
- No true jersey-number/player recognition -- `--ocr-jerseys` is a
  best-effort hint (plain OCR, no sport-specific training), not reliable
  identification. See `src/jersey_ocr.py` for why, and what a real version
  would need.
- Field-point calibration is manual (you find the pixel coordinates).
  A trained keypoint model (the `roboflow/sports` pattern -- detect field/
  court markings automatically, solve the homography without a human)
  would remove that step, but needs a labeled dataset we don't have yet.
- No basketball grading rubric (see Basketball above).
