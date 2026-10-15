#!/usr/bin/env python3
"""CLI entry point for the 1-on-1 WR/CB tracking + grading MVP.

Three-step workflow, because automatic "who is this athlete" identification
isn't reliable enough to fully automate -- a human confirms which tracked
person is the WR and which is the CB once per clip, and real-world
calibration needs a few points a human locates in the frame:

  1. discover   -- detect + track every person in the clip (YOLO + BoT-SORT),
                   save a thumbnail per track so you can see who's who.
  2. calibrate  -- (optional but recommended) export a still frame + a
                   points-file template; fill in where known field/court
                   points land in pixels to get a real homography instead
                   of guessed pixel ratios.
  3. grade      -- point the WR/CB track IDs (from step 1) at real
                   names/roles, add situation info (down/distance/etc),
                   optionally pass the calibration from step 2, and produce
                   the annotated video + grade report.

Example:
    python run_pipeline.py discover --video clip.mp4 --out out/clip1
    # look at out/clip1/thumbs/track_*.jpg, note which TID is which player

    python run_pipeline.py calibrate --video clip.mp4 --out out/clip1 --frame 0
    # open out/clip1/calibration_frame.png, fill in the "px" fields in
    # out/clip1/calibration_points.json

    python run_pipeline.py grade --video clip.mp4 --out out/clip1 \\
        --wr-track-id 3 --wr-name "J. Smith" \\
        --cb-track-id 5 --cb-name "A. Jones" \\
        --down 3 --distance 7 --level college \\
        --calibration-points out/clip1/calibration_points.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from schema import Calibration, PlayerIdentity, Situation  # noqa: E402
import calibration as calibration_mod  # noqa: E402
import track as track_mod  # noqa: E402
import identity as identity_mod  # noqa: E402
import jersey_ocr  # noqa: E402
import pose_features  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import grading  # noqa: E402
import annotate  # noqa: E402
import report as report_mod  # noqa: E402


def cmd_calibrate(args):
    """Exports a still frame + a points-file template so you can build a
    FieldCalibration by hand: open the frame in any image viewer that shows
    cursor pixel position, read off where each known field/court point
    lands, and fill in the "px" entries this leaves as null.
    """
    import cv2
    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: could not read frame {args.frame} from {args.video}")
        sys.exit(1)

    frame_path = os.path.join(args.out, "calibration_frame.png")
    cv2.imwrite(frame_path, frame)
    print(f"Saved frame {args.frame} -> {frame_path}")
    print("Open it in any viewer that shows cursor pixel position and read off "
          "where each point below actually lands in the shot.")

    dims = calibration_mod.FOOTBALL_DIMENSIONS_YD if args.sport == "football" else calibration_mod.BASKETBALL_DIMENSIONS_FT
    unit = "yd" if args.sport == "football" else "ft"
    if args.sport == "football":
        # A practical 4-point minimum: both hash marks at two different yard lines
        # 5 yards apart gives a well-conditioned quad (not collinear, spans depth).
        hw = dims["ncaa_hash_width"] / 2  # NCAA default; edit if NFL/HS
        template_points = [
            {"px": None, "world": [-hw, 0], "label": "left hash, near yard line"},
            {"px": None, "world": [hw, 0], "label": "right hash, near yard line"},
            {"px": None, "world": [-hw, dims["yard_line_spacing"]], "label": "left hash, next yard line (5 yd further)"},
            {"px": None, "world": [hw, dims["yard_line_spacing"]], "label": "right hash, next yard line (5 yd further)"},
        ]
    else:
        hw = dims["key_width"] / 2
        template_points = [
            {"px": None, "world": [-hw, 0], "label": "near baseline, left edge of the key"},
            {"px": None, "world": [hw, 0], "label": "near baseline, right edge of the key"},
            {"px": None, "world": [-hw, dims["free_throw_distance"]], "label": "free-throw line, left edge of the key"},
            {"px": None, "world": [hw, dims["free_throw_distance"]], "label": "free-throw line, right edge of the key"},
        ]

    points_path = os.path.join(args.out, "calibration_points.json")
    with open(points_path, "w") as f:
        json.dump({"sport": args.sport, "unit": unit, "points": template_points}, f, indent=2)
    print(f"Saved a 4-point template -> {points_path}")
    print(f'Fill in each "px": [x, y] (pixel coords read off {frame_path}), then pass '
          f"--calibration-points {points_path} to 'grade'. Add more points (any count >= 4) "
          "for a better-conditioned homography if you can identify them.")


def cmd_discover(args):
    os.makedirs(args.out, exist_ok=True)
    print(f"Running detection + tracking ({args.tracker}) on {args.video} ...")
    tracks, fps, frame_size = track_mod.run_tracking(args.video, model_path=args.model, conf=args.conf,
                                                       tracker=f"{args.tracker}.yaml")

    tracks_json_path = os.path.join(args.out, "tracks.json")
    with open(tracks_json_path, "w") as f:
        json.dump(track_mod.tracks_to_json(tracks, fps, frame_size), f)
    print(f"Saved raw tracks -> {tracks_json_path}")

    thumbs_dir = os.path.join(args.out, "thumbs")
    saved = identity_mod.save_track_thumbnails(args.video, tracks, thumbs_dir)
    print(f"Saved {len(saved)} track thumbnails -> {thumbs_dir}")

    jersey_guesses = {}
    if args.ocr_jerseys:
        print("Running experimental jersey-number OCR (best-effort, low confidence -- see README) ...")
        jersey_guesses = jersey_ocr.suggest_jersey_numbers(args.video, tracks)

    summary = track_mod.summarize_tracks(tracks)
    print("\nTrack summary (most-present first):")
    header = f"{'TID':>4}  {'frames':>6}  {'first':>6}  {'last':>6}  {'avg_w':>6}  {'avg_h':>6}"
    header += f"  {'jersey?':>8}" if args.ocr_jerseys else ""
    header += "  thumbnail"
    print(header)
    for row in summary:
        thumb = saved.get(row["track_id"], "")
        line = (f"{row['track_id']:>4}  {row['frames_present']:>6}  {row['first_frame']:>6}  "
                f"{row['last_frame']:>6}  {row['avg_box_w']:>6}  {row['avg_box_h']:>6}")
        if args.ocr_jerseys:
            guess = jersey_guesses.get(row["track_id"])
            line += f"  {(guess[0] + f' ({guess[1]:.0%})') if guess else '--':>8}"
        line += f"  {thumb}"
        print(line)

    with open(os.path.join(args.out, "track_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if jersey_guesses:
        with open(os.path.join(args.out, "jersey_guesses.json"), "w") as f:
            json.dump({str(k): {"number": v[0], "confidence": v[1]} for k, v in jersey_guesses.items()}, f, indent=2)


def cmd_grade(args):
    os.makedirs(args.out, exist_ok=True)
    tracks_json_path = os.path.join(args.out, "tracks.json")

    if args.reuse_tracks and os.path.exists(tracks_json_path):
        print(f"Reusing cached tracks from {tracks_json_path}")
        with open(tracks_json_path) as f:
            tracks, fps, frame_size = track_mod.tracks_from_json(json.load(f))
    else:
        print(f"Running detection + tracking ({args.tracker}) on {args.video} ...")
        tracks, fps, frame_size = track_mod.run_tracking(args.video, model_path=args.model, conf=args.conf,
                                                           tracker=f"{args.tracker}.yaml")
        with open(tracks_json_path, "w") as f:
            json.dump(track_mod.tracks_to_json(tracks, fps, frame_size), f)

    if args.wr_track_id not in tracks or args.cb_track_id not in tracks:
        print(f"ERROR: track id {args.wr_track_id} or {args.cb_track_id} not found. "
              f"Available: {sorted(tracks.keys())}. Run 'discover' first.")
        sys.exit(1)

    wr_track = tracks[args.wr_track_id]
    cb_track = tracks[args.cb_track_id]

    calibration = None
    if args.calibration_points:
        calibration = calibration_mod.FieldCalibration.from_points_file(args.calibration_points)
        print(f"Loaded {len(calibration.points)}-point homography calibration from {args.calibration_points} "
              f"(perspective-correct, ~{calibration.yards_per_pixel:.4f} units/px avg).")
    elif args.calibration_px and args.calibration_yards:
        calibration = Calibration(px_distance=args.calibration_px, yard_distance=args.calibration_yards)
        print("Using single-ratio calibration (no perspective correction) -- "
              "prefer --calibration-points (see the 'calibrate' subcommand) when you can.")
    else:
        print("NOTE: no calibration given -- speeds/distances will be approximate (pixel-based), not real yards/mph. "
              "Run 'python run_pipeline.py calibrate --video ... --out ...' to build one.")

    if args.sport == "basketball":
        print("ERROR: grading is football-specific right now (WR/CB roles, route/coverage calls, "
              "down & distance). Tracking + pose + calibration all work for basketball via 'discover' "
              "and 'calibrate --sport basketball', but there's no basketball grading rubric yet -- "
              "that needs your input on what should actually be graded (a cut? a closeout? a screen "
              "read?) before it's worth building. See README.")
        sys.exit(1)

    situation = Situation(
        sport=args.sport, down=args.down, distance=args.distance, hash_mark=args.hash_mark,
        offense_formation=args.formation, coverage_call=args.coverage_call,
        level=args.level, notes=args.notes,
    )

    print("Running pose extraction (MediaPipe) on WR + CB tracks ...")
    pose_features.add_pose_to_track(args.video, wr_track)
    pose_features.add_pose_to_track(args.video, cb_track)

    print("Computing route + coverage metrics ...")
    route = metrics_mod.classify_route(wr_track, calibration)
    coverage = metrics_mod.classify_coverage(cb_track, wr_track, calibration)
    if calibration is not None:
        route.separation_at_break_yd = metrics_mod.separation_at(wr_track, cb_track, calibration, route.break_frame)
        route.separation_at_catch_yd = metrics_mod.separation_at(wr_track, cb_track, calibration, wr_track.points[-1].frame)
    else:
        route.notes.append("Separation @ break/catch needs calibration -- skipped (would be pixels, not yards).")

    print("Grading ...")
    wr_grade = grading.grade_wr(route, situation)
    cb_grade = grading.grade_cb(coverage, situation)

    wr_identity = PlayerIdentity(track_id=args.wr_track_id, role="WR", name=args.wr_name or "WR", number=args.wr_number)
    cb_identity = PlayerIdentity(track_id=args.cb_track_id, role="CB", name=args.cb_name or "CB", number=args.cb_number)
    identities = {args.wr_track_id: wr_identity, args.cb_track_id: cb_identity}

    print("Rendering annotated video ...")
    out_video = os.path.join(args.out, "annotated.mp4")
    annotate.render(args.video, out_video, {args.wr_track_id: wr_track, args.cb_track_id: cb_track},
                     identities, route_label=route.route_guess, coverage_label=coverage.technique_guess)

    report = report_mod.build_report_dict(situation, wr_identity, cb_identity, route, coverage,
                                           wr_grade, cb_grade, video_file=os.path.basename(args.video))
    json_path = os.path.join(args.out, "report.json")
    html_path = os.path.join(args.out, "report.html")
    report_mod.save_report(report, wr_grade, cb_grade, "annotated.mp4", json_path, html_path)

    print(f"\nDone.\n  Annotated video -> {out_video}\n  JSON report     -> {json_path}\n  HTML report     -> {html_path}")
    print(f"\nWR grade: {wr_grade.score}/100 ({wr_grade.letter})   CB grade: {cb_grade.score}/100 ({cb_grade.letter})")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("discover", help="Detect + track everyone in the clip; save thumbnails to identify TIDs.")
    d.add_argument("--video", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--model", default="yolov8n.pt")
    d.add_argument("--conf", type=float, default=0.35)
    d.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort",
                    help="botsort (default): better ID persistence through contact/occlusion. "
                         "bytetrack: faster, more ID switches -- useful to A/B against botsort.")
    d.add_argument("--ocr-jerseys", action="store_true",
                    help="Experimental: attempt jersey-number OCR per track (low confidence, see README). "
                         "Off by default since it's slow and often wrong -- use it as a hint, not an answer.")
    d.set_defaults(func=cmd_discover)

    c = sub.add_parser("calibrate", help="Export a still frame + points-file template for real-world calibration.")
    c.add_argument("--video", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--frame", type=int, default=0, help="Frame number to export (pick one with field/court markings clearly visible).")
    c.add_argument("--sport", choices=["football", "basketball"], default="football")
    c.set_defaults(func=cmd_calibrate)

    g = sub.add_parser("grade", help="Grade a WR/CB 1-on-1 rep given confirmed track IDs. Football only -- see 'discover'/'calibrate' for basketball.")
    g.add_argument("--video", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", default="yolov8n.pt")
    g.add_argument("--conf", type=float, default=0.35)
    g.add_argument("--tracker", choices=["botsort", "bytetrack"], default="botsort")
    g.add_argument("--reuse-tracks", action="store_true", default=True)
    g.add_argument("--sport", choices=["football", "basketball"], default="football")
    g.add_argument("--wr-track-id", type=int, required=True)
    g.add_argument("--cb-track-id", type=int, required=True)
    g.add_argument("--wr-name")
    g.add_argument("--wr-number")
    g.add_argument("--cb-name")
    g.add_argument("--cb-number")
    g.add_argument("--calibration-points", help="Path to a calibration_points.json from 'calibrate' (preferred -- perspective-correct).")
    g.add_argument("--calibration-px", type=float, help="Fallback: pixel distance of one known real-world distance (e.g. hash-to-hash).")
    g.add_argument("--calibration-yards", type=float, help="That same distance in yards.")
    g.add_argument("--down", type=int)
    g.add_argument("--distance", type=int)
    g.add_argument("--hash-mark", choices=["left", "middle", "right"])
    g.add_argument("--formation")
    g.add_argument("--coverage-call")
    g.add_argument("--level", choices=["high_school", "college", "nfl"], default="college")
    g.add_argument("--notes")
    g.set_defaults(func=cmd_grade)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
