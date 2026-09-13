#!/usr/bin/env python3
"""CLI entry point for the 1-on-1 WR/CB tracking + grading MVP.

Two-step workflow, because automatic "who is this athlete" identification
isn't reliable enough to fully automate -- a human confirms which tracked
person is the WR and which is the CB once per clip:

  1. discover  -- detect + track every person in the clip, save a thumbnail
                  per track so you can see who's who.
  2. grade     -- point the WR/CB track IDs (from step 1) at real
                  names/roles, add situation info (down/distance/etc), and
                  produce the annotated video + grade report.

Example:
    python run_pipeline.py discover --video clip.mp4 --out out/clip1
    # look at out/clip1/thumbs/track_*.jpg, note which TID is which player
    python run_pipeline.py grade --video clip.mp4 --out out/clip1 \\
        --wr-track-id 3 --wr-name "J. Smith" \\
        --cb-track-id 5 --cb-name "A. Jones" \\
        --down 3 --distance 7 --level college \\
        --calibration-px 400 --calibration-yards 10
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from schema import Calibration, PlayerIdentity, Situation  # noqa: E402
import track as track_mod  # noqa: E402
import identity as identity_mod  # noqa: E402
import pose_features  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import grading  # noqa: E402
import annotate  # noqa: E402
import report as report_mod  # noqa: E402


def cmd_discover(args):
    os.makedirs(args.out, exist_ok=True)
    print(f"Running detection + tracking on {args.video} ...")
    tracks, fps, frame_size = track_mod.run_tracking(args.video, model_path=args.model, conf=args.conf)

    tracks_json_path = os.path.join(args.out, "tracks.json")
    with open(tracks_json_path, "w") as f:
        json.dump(track_mod.tracks_to_json(tracks, fps, frame_size), f)
    print(f"Saved raw tracks -> {tracks_json_path}")

    thumbs_dir = os.path.join(args.out, "thumbs")
    saved = identity_mod.save_track_thumbnails(args.video, tracks, thumbs_dir)
    print(f"Saved {len(saved)} track thumbnails -> {thumbs_dir}")

    summary = track_mod.summarize_tracks(tracks)
    print("\nTrack summary (most-present first):")
    print(f"{'TID':>4}  {'frames':>6}  {'first':>6}  {'last':>6}  {'avg_w':>6}  {'avg_h':>6}  thumbnail")
    for row in summary:
        thumb = saved.get(row["track_id"], "")
        print(f"{row['track_id']:>4}  {row['frames_present']:>6}  {row['first_frame']:>6}  "
              f"{row['last_frame']:>6}  {row['avg_box_w']:>6}  {row['avg_box_h']:>6}  {thumb}")

    with open(os.path.join(args.out, "track_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def cmd_grade(args):
    os.makedirs(args.out, exist_ok=True)
    tracks_json_path = os.path.join(args.out, "tracks.json")

    if args.reuse_tracks and os.path.exists(tracks_json_path):
        print(f"Reusing cached tracks from {tracks_json_path}")
        with open(tracks_json_path) as f:
            tracks, fps, frame_size = track_mod.tracks_from_json(json.load(f))
    else:
        print(f"Running detection + tracking on {args.video} ...")
        tracks, fps, frame_size = track_mod.run_tracking(args.video, model_path=args.model, conf=args.conf)
        with open(tracks_json_path, "w") as f:
            json.dump(track_mod.tracks_to_json(tracks, fps, frame_size), f)

    if args.wr_track_id not in tracks or args.cb_track_id not in tracks:
        print(f"ERROR: track id {args.wr_track_id} or {args.cb_track_id} not found. "
              f"Available: {sorted(tracks.keys())}. Run 'discover' first.")
        sys.exit(1)

    wr_track = tracks[args.wr_track_id]
    cb_track = tracks[args.cb_track_id]

    calibration = None
    if args.calibration_px and args.calibration_yards:
        calibration = Calibration(px_distance=args.calibration_px, yard_distance=args.calibration_yards)
    else:
        print("NOTE: no --calibration-px/--calibration-yards given -- speeds/distances will be "
              "approximate (pixel-based), not real yards/mph.")

    situation = Situation(
        down=args.down, distance=args.distance, hash_mark=args.hash_mark,
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
    d.set_defaults(func=cmd_discover)

    g = sub.add_parser("grade", help="Grade a WR/CB 1-on-1 rep given confirmed track IDs.")
    g.add_argument("--video", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--model", default="yolov8n.pt")
    g.add_argument("--conf", type=float, default=0.35)
    g.add_argument("--reuse-tracks", action="store_true", default=True)
    g.add_argument("--wr-track-id", type=int, required=True)
    g.add_argument("--cb-track-id", type=int, required=True)
    g.add_argument("--wr-name")
    g.add_argument("--wr-number")
    g.add_argument("--cb-name")
    g.add_argument("--cb-number")
    g.add_argument("--calibration-px", type=float, help="Pixel distance of a known real-world distance (e.g. hash-to-hash).")
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
