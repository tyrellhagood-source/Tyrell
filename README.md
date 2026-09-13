# Tyrell -- 1-on-1 WR/CB Tracking & Grading (MVP)

Proof-of-concept pipeline for breaking down 1-on-1 WR vs. CB reps from
video: detect and track both athletes, classify the route and coverage,
grade the rep, and hand back drills + pros/cons -- the same idea as the
"analyzing film with computer vision" clips floating around, built out
into something you can actually run on your own footage.

**Where this is at right now:** the tracking + assisted-identification
loop works end to end (see Status below). Route/coverage classification
and grading are rule-based heuristics, not trained models -- accurate
enough to prove the pipeline out, not yet a scouting-grade grade.

## How it works

1. **Detect + track** (`src/track.py`) -- YOLOv8 (pretrained, no training
   needed) finds every person in the clip; ByteTrack gives each one a
   persistent track ID (TID) frame to frame.
2. **Identify** (`src/identity.py`) -- true automatic "who is this athlete"
   recognition (jersey OCR / face ID) isn't reliable on typical drill or
   broadcast footage without a roster-specific trained model, so a human
   confirms which TID is the WR and which is the CB once per clip (see
   `discover` below). That assignment (PID) is what shows up as "WR" /
   "CB" / a real name in the output, and survives even if the raw tracker
   re-assigns a new TID mid-clip.
3. **Pose** (`src/pose_features.py`) -- MediaPipe Pose runs on each
   tracked player's crop for movement/posture features.
4. **Metrics + classification** (`src/metrics.py`) -- from the tracked
   trajectory: speed, separation, break point/angle, a heuristic route
   guess (slant/out/post/corner/go/etc.), and a heuristic coverage guess
   (press/off, man/zone via mirror correlation).
5. **Grading** (`src/grading.py`) -- turns those metrics into a 0-100
   grade + letter grade, pros/cons bullets, and matched drills from
   `data/drills/drills_library.json`.
6. **Output** (`src/annotate.py`, `src/report.py`) -- an annotated MP4
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

On a minimal Linux box you may also need system GL libs MediaPipe's
runtime links against even for CPU-only inference:
```bash
apt-get install -y libegl1 libgles2
```

## Usage

**Step 1 -- discover who's in the clip:**
```bash
python run_pipeline.py discover --video clip.mp4 --out out/clip1
```
This prints a table of every tracked person (TID, how many frames they
were tracked for) and saves a thumbnail per track to `out/clip1/thumbs/`
so you can look and see which TID is the WR and which is the CB.

**Step 2 -- grade the rep:**
```bash
python run_pipeline.py grade --video clip.mp4 --out out/clip1 \
    --wr-track-id 3 --wr-name "J. Smith" \
    --cb-track-id 5 --cb-name "A. Jones" \
    --down 3 --distance 7 --formation "11 personnel, trips right" \
    --coverage-call "Cover 1" --level college \
    --calibration-px 400 --calibration-yards 10
```
`--calibration-px`/`--calibration-yards` map a known real-world distance
(hash-to-hash, a 10-yard line segment, etc.) visible in the shot to pixels,
so speed/separation come out in mph/yards instead of raw pixels. Skip them
and the pipeline still runs -- it'll just say so and grade on
calibration-independent signals only (break sharpness, mirror
correlation), leaving speed/separation-dependent numbers as `null` instead
of guessing.

Output lands in `out/clip1/`: `annotated.mp4`, `report.json`, `report.html`
(open the HTML file in a browser -- it embeds the annotated clip next to
both grade cards).

## Status / what's proven vs. what's next

**Proven (this MVP):** YOLO+ByteTrack reliably tracks both athletes in a
1-on-1 rep with persistent boxes/IDs through cuts, occlusion, and camera
motion; MediaPipe pose extraction runs per-track; the assisted PID
workflow (discover -> label -> grade) produces a full annotated
video + report end to end.

**Heuristic, not trained (next steps if this direction is worth
pursuing):**
- Route/coverage classification is rule-based on trajectory shape. A
  trained classifier (given a labeled set of real reps) would be far more
  accurate, especially for route families that look similar from a side
  angle.
- Grading weights are a starting rubric, not calibrated against real
  coaching grades -- worth tuning once there's a set of reps a human coach
  has also graded, to compare against.
- No automatic down/distance/situation reading from broadcast graphics.
- No true jersey-number/player recognition -- identification is
  human-assisted (by design, for reliability) rather than automatic.
