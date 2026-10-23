# Tyrell -- 1-on-1 WR/CB Tracking & Grading (MVP)

Proof-of-concept pipeline for breaking down 1-on-1 WR vs. CB reps from
video: detect and track both athletes, classify the route and coverage,
grade the rep, and hand back drills + pros/cons -- the same idea as the
"analyzing film with computer vision" clips floating around, built out
into something you can actually run on your own footage.

**Where this is at right now:** the tracking + assisted-identification
loop works end to end, verified against real NFL broadcast footage, not
just synthetic test data (see Status below). Route/coverage classification
and grading are rule-based heuristics, not trained models -- accurate
enough to prove the pipeline out, not yet a scouting-grade grade. Every
clip now gets real-world units by default (see Calibration below), and
route calls use the actual route tree (slant/out/dig/curl/comeback/post/
corner/go/double-move), not a vague "breaking route" label.

## How it works

1. **Detect + track** (`src/track.py`) -- YOLOv8 (pretrained, no training
   needed) finds every person in the clip; **BoT-SORT** (default tracker)
   gives each one a persistent track ID (TID) frame to frame, holding IDs
   together through contact/occlusion better than plain ByteTrack (still
   available via `--tracker bytetrack` to compare). For genuinely crowded
   scenes -- a huddle, a pile, a sideline cluster -- `--tracker
   botsort-crowd` adds appearance (ReID) matching on top; see Tracking
   options below.
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
   to real-world yards/feet so speed/separation/route depth come out as
   real units instead of raw pixels. Three ways, in order of how it's
   picked:
   - **Auto, from player height (default, no setup needed)** -- uses the
     WR's own tracked bounding-box height as an implicit ruler (assumes a
     population-average skill-position height, ~6'1"). Approximate, but
     turns every clip into real units out of the box. Pass
     `--no-auto-calibration` to see raw pixels instead.
   - **Homography** (`calibrate` subcommand) -- a proper perspective
     transform from 4+ known field/court points you locate by eye. More
     accurate, needs a few minutes of manual point-picking.
   - **Single-ratio** (`--calibration-px`/`--calibration-yards`) -- one
     known distance, no perspective correction. The old fallback.
5. **Metrics + route classification** (`src/metrics.py`) -- from the
   tracked trajectory: speed (outlier-resistant -- see Status), separation,
   break point(s) (double-move aware), and a real route-tree classification
   (go/slant/out/dig/in/curl/comeback/post/corner) using depth buckets and
   post-break trend (continuing downfield vs. coming back toward the LOS).
   `--wr-alignment left|right` resolves the ambiguous pairs (In vs Out,
   Post vs Corner) when you tell it which side of the screen the WR lined
   up on. Coverage classification (press/off, man/zone via mirror
   correlation) is unchanged.
6. **Grading** (`src/grading.py`) -- football-specific: turns those
   metrics into a 0-100 grade + letter grade, pros/cons bullets, and
   matched drills from `data/drills/drills_library.json`. The CB's
   "separation allowed" score is based on real in-route separation (the
   same number the WR side calls "separation created"), not pre-snap
   alignment distance.
7. **Narrative summary** (`src/narrative.py`) -- a short coach-style
   paragraph combining the route call, coverage call, and both grades'
   deciding factors, so there's a "read the whole picture in one go"
   layer on top of the raw numbers.
8. **Output** (`src/annotate.py`, `src/report.py`) -- an annotated MP4
   (boxes, PID/TID labels, route trail) and a report (JSON + a
   self-contained HTML report card, now including the narrative summary).

Down/distance, hash, formation, and coverage call are **manually supplied**
situational context (`--down`, `--distance`, etc.) -- not derived from the
video. Reliable OCR of a scoreboard/broadcast graphic is a separate,
broadcast-specific problem and out of scope for this MVP.

## Setup

```bash
pip install -r requirements.txt
./scripts/download_models.sh   # fetches the MediaPipe pose model (~6MB)
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run, as does
the ReID model if you use `--tracker botsort-crowd`.
`easyocr` (jersey OCR) and `pandas` (Big Data Bowl validation) are optional
-- skip them if you don't need `--ocr-jerseys` or `scripts/validate_against_bdb.py`.

On a minimal Linux box you may also need system GL libs MediaPipe's
runtime links against even for CPU-only inference, and `lap` for the
tracker's linear assignment step if your environment can't auto-install it:
```bash
apt-get install -y libegl1 libgles2
pip install lap
```

## Usage

**Step 1 -- discover who's in the clip:**
```bash
python run_pipeline.py discover --video clip.mp4 --out out/clip1
# add --ocr-jerseys for experimental jersey-number hints
# add --tracker botsort-crowd for huddles/piles/crowded footage
```
This prints a table of every tracked person (TID, how many frames they
were tracked for) and saves a thumbnail per track to `out/clip1/thumbs/`
so you can look and see which TID is the WR and which is the CB.

**Step 2 -- calibrate (optional -- a real calibration beats the automatic
one, but you don't have to):**
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
that needs per-frame calibration, which isn't built yet. The default
auto-calibration (player height) is more robust to this since it's
re-derived from the player's own box each clip, but has its own
population-average-height error margin -- see `calibration.py`.

**Step 3 -- grade the rep:**
```bash
python run_pipeline.py grade --video clip.mp4 --out out/clip1 \
    --wr-track-id 3 --wr-name "J. Smith" \
    --cb-track-id 5 --cb-name "A. Jones" \
    --wr-alignment left \
    --down 3 --distance 7 --formation "11 personnel, trips right" \
    --coverage-call "Cover 1" --level college \
    --calibration-points out/clip1/calibration_points.json
```
No calibration file? It falls back to auto-calibration-from-height
automatically -- no flag needed. Pass `--calibration-px`/`--calibration-yards`
for the old single-ratio fallback instead, or `--no-auto-calibration` to see
raw pixels with no unit guessing at all.

Output lands in `out/clip1/`: `annotated.mp4`, `report.json`, `report.html`
(open the HTML file in a browser -- it embeds the annotated clip, the
narrative summary, and both grade cards).

## Tracking options for crowded scenes

Plain motion/IoU tracking (the default, either tracker) loses or swaps IDs
when players fully occlude each other for more than a frame or two --
there's no visual signal to tell them apart once two boxes merge. For a
huddle, a pile, or a crowded sideline shot, `--tracker botsort-crowd` turns
on BoT-SORT's appearance (ReID) matching plus a longer lost-track buffer
(`config/botsort_crowd.yaml`). It's slower (an extra model pass per box)
and was *not* independently verified against real huddle footage (none was
available to test against) -- on the one real clip it was tried on (not a
huddle), it did measurably reduce track fragmentation (123 -> 111 total
tracks on the same clip), which is a good sign but not proof it handles an
actual huddle well. Compare its `track_summary.json` against the default
tracker's on your own footage before trusting it blind.

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
(using its machine-readable `route_family` tag) against labeled routes
from the **NFL Big Data Bowl** (Kaggle) tracking data -- real NFL player
tracking + play-by-play route/coverage labels, the best available ground
truth for sanity-checking the heuristics without needing our own large
labeled video set. This sandbox has no route to kaggle.com, so you'll
need to download it yourself (free Kaggle account): see the script's
docstring for the exact steps and expected file layout.

## Status / what's proven vs. what's next

**Proven, against real footage, not just synthetic tests:** YOLO+BoT-SORT
reliably tracks both athletes in a 1-on-1 rep with persistent boxes/IDs
through cuts, contact, and camera motion; the full discover -> calibrate ->
grade workflow produces a correct annotated video + report on a real NFL
broadcast clip (Seahawks WR vs. Commanders CB) -- route correctly called
as a vertical go route once the issues below were found and fixed, grade
and narrative summary both sane.

**Real bugs found and fixed by testing against that real clip** (not
hypothetical -- each one changed the actual output):
- Tracker-lock-on jitter in the first couple of tracked frames was nearly
  pure lateral motion where the route actually ran downfield, which (a)
  skewed the release-direction estimate used for every depth figure, and
  (b) produced a single-frame speed spike reported as a "32 mph" top
  speed for an NFL WR -- physically impossible. Fixed by averaging the
  release direction over more frames and taking the max of a *smoothed*
  speed series instead of the raw per-frame max.
- A break-angle sanity cap (added to reject camera-pan artifacts
  masquerading as route breaks) was *also* wrongly rejecting genuine
  comeback/hitch routes, which legitimately approach a full reversal.
  Fixed by making the cap conditional on having real calibrated depth to
  trust a long reversal with.
- Double-move detection missed an obvious synthetic double-move entirely
  -- searching only *after* the first found break blended two nearby
  transitions into one in testing. Fixed with proper binary segmentation
  (search both sides of each found break).
- `grade_cb`'s "separation allowed" score used to fall back to *initial*
  pre-snap separation, which unfairly penalizes normal off-coverage
  alignment as if it were a coverage failure. Fixed to prefer real
  in-route separation.
- The narrative summary blanket-lowercased source strings to splice them
  into sentences, which turned a WR's name ("J. Smith-Njigba") and
  embedded acronyms ("the WR's break") into lowercase nonsense. Fixed to
  only lowercase where it's actually safe to.

**Heuristic, not trained (next steps if this direction is worth
pursuing):**
- Route/coverage classification is rule-based on trajectory shape, not a
  trained model. `scripts/validate_against_bdb.py` is the honest way to
  find out how good it actually is at scale, beyond the handful of real
  and synthetic cases checked so far.
- Grading weights are a starting rubric, not calibrated against real
  coaching grades -- worth tuning once there's a set of reps a human coach
  has also graded, to compare against.
- Whole-clip metrics (total depth, top speed) blend the route-running
  phase with whatever happens after the catch (a tackle, a scrum) if the
  tracked clip runs that long -- trim to the route stem before the catch
  for the cleanest numbers; see the note `classify_route` adds when this
  looks like it's happening.
- No automatic down/distance/situation reading from broadcast graphics.
- No true jersey-number/player recognition -- `--ocr-jerseys` is a
  best-effort hint (plain OCR, no sport-specific training), not reliable
  identification. See `src/jersey_ocr.py` for why, and what a real version
  would need.
- Homography calibration is manual (you find the pixel coordinates); auto
  (player-height) calibration has its own population-average error
  margin. A trained field-keypoint model (the `roboflow/sports` pattern)
  would give an automatic, more accurate homography, but needs a labeled
  dataset we don't have yet.
- `--tracker botsort-crowd` is a reasoned starting config, not tuned
  against real huddle footage -- see Tracking options above.
- No basketball grading rubric (see Basketball above).
