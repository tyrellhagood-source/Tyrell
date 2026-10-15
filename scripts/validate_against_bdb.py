#!/usr/bin/env python3
"""Validates metrics.classify_route / classify_coverage against real labeled
NFL tracking data from Kaggle's NFL Big Data Bowl competitions.

This sandbox can't download Kaggle data itself (no network access to
kaggle.com), so this script expects you to have already downloaded it:

    1. Get the data (needs a free Kaggle account):
       https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data
       (or whichever year -- 2024 is the one with per-route/coverage labels
       in player_play.csv; adjust COLUMN NAMES below if a different year's
       schema differs slightly).
    2. Unzip it somewhere, e.g. ~/bdb2024/
    3. Run:
       python scripts/validate_against_bdb.py --data-dir ~/bdb2024 --weeks 1

What it does: for every pass play in the given week(s), pulls the targeted
receiver's tracking data, converts it into our Track schema (BDB tracking
is already in real field yards, so no pixel calibration is needed here --
that's a nice side benefit of validating against this dataset), runs our
route/coverage heuristics on it, and compares against the labeled
`routeRan` / coverage columns to report accuracy. This is the honest way
to find out whether the heuristics in metrics.py are any good, instead of
guessing from a handful of eyeballed clips.
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schema import Track, TrackPoint  # noqa: E402
import metrics as metrics_mod  # noqa: E402

BDB_FPS = 10.0

# Our heuristic route_guess strings are free text ("Post / Corner (deep
# breaking route)"); BDB's routeRan is a fixed vocabulary. This maps BDB
# labels to the keyword our classifier would need to produce for a "hit".
# Loose on purpose -- the heuristic was never meant to split hairs between
# e.g. CORNER and POST, so scoring at this granularity is the fair test.
ROUTE_LABEL_GROUPS = {
    "GO": "vertical", "FLAT": "shallow", "SLANT": "shallow", "SCREEN": "shallow",
    "HITCH": "shallow", "OUT": "shallow", "CROSS": "intermediate", "IN": "intermediate",
    "ANGLE": "intermediate", "CORNER": "deep", "POST": "deep", "WHEEL": "deep",
}


def _our_group(route_guess: str) -> str:
    g = route_guess.lower()
    if "vertical" in g or "go / fade" in g or "seam" in g:
        return "vertical"
    if "shallow" in g:
        return "shallow"
    if "intermediate" in g:
        return "intermediate"
    if "deep" in g:
        return "deep"
    return "unknown"


def build_track(frames_df: pd.DataFrame) -> Track:
    points = []
    for _, row in frames_df.sort_values("frameId").iterrows():
        points.append(TrackPoint(
            frame=int(row["frameId"]), t=row["frameId"] / BDB_FPS,
            bbox=(0, 0, 0, 0), centroid=(float(row["x"]), float(row["y"])),
        ))
    return Track(track_id=int(frames_df["nflId"].iloc[0]), points=points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="Directory with the unzipped BDB CSVs.")
    ap.add_argument("--weeks", nargs="+", type=int, default=[1], help="tracking_week_N.csv files to use.")
    ap.add_argument("--max-plays", type=int, default=200, help="Cap for a quick run; omit for the full week(s).")
    args = ap.parse_args()

    plays_path = os.path.join(args.data_dir, "plays.csv")
    play_path = os.path.join(args.data_dir, "player_play.csv")
    if not os.path.exists(plays_path) or not os.path.exists(play_path):
        print(f"ERROR: expected plays.csv and player_play.csv under {args.data_dir} -- "
              "did you unzip the Kaggle download there? See this script's docstring.")
        sys.exit(1)

    player_play = pd.read_csv(play_path)
    if "routeRan" not in player_play.columns:
        print(f"ERROR: player_play.csv has no 'routeRan' column (columns: {list(player_play.columns)}). "
              "This script was written against the BDB 2024 schema -- adjust ROUTE_LABEL_GROUPS / "
              "column names at the top of this file for your year's schema.")
        sys.exit(1)

    route_rows = player_play.dropna(subset=["routeRan"])

    group_correct = Counter()
    group_total = Counter()
    confusion = defaultdict(Counter)
    n_scored = 0

    for week in args.weeks:
        tracking_path = os.path.join(args.data_dir, f"tracking_week_{week}.csv")
        if not os.path.exists(tracking_path):
            print(f"WARNING: {tracking_path} not found, skipping week {week}.")
            continue
        print(f"Loading {tracking_path} ...")
        tracking = pd.read_csv(tracking_path)

        week_routes = route_rows.merge(
            tracking[["gameId", "playId", "nflId"]].drop_duplicates(),
            on=["gameId", "playId", "nflId"], how="inner",
        )

        for _, play_row in week_routes.head(args.max_plays).iterrows():
            frames = tracking[
                (tracking.gameId == play_row.gameId) & (tracking.playId == play_row.playId)
                & (tracking.nflId == play_row.nflId)
            ]
            if len(frames) < 8:
                continue
            track = build_track(frames)
            route = metrics_mod.classify_route(track, calibration=None)  # BDB coords are already yards
            predicted_group = _our_group(route.route_guess)
            true_group = ROUTE_LABEL_GROUPS.get(str(play_row.routeRan).upper(), "unknown")
            if true_group == "unknown":
                continue

            n_scored += 1
            group_total[true_group] += 1
            confusion[true_group][predicted_group] += 1
            if predicted_group == true_group:
                group_correct[true_group] += 1

    if n_scored == 0:
        print("No plays scored -- check --data-dir / --weeks.")
        return

    print(f"\nScored {n_scored} receiver routes against labeled routeRan (grouped: vertical/shallow/intermediate/deep).")
    overall = sum(group_correct.values()) / n_scored
    print(f"Overall group accuracy: {overall:.1%}\n")
    print(f"{'true group':<14}{'n':>6}{'accuracy':>10}")
    for g in sorted(group_total):
        acc = group_correct[g] / group_total[g]
        print(f"{g:<14}{group_total[g]:>6}{acc:>9.1%}")

    print("\nConfusion (rows = true, cols = predicted):")
    true_groups = sorted(group_total)
    predicted_groups = sorted({g2 for row in confusion.values() for g2 in row})
    all_cols = sorted(set(true_groups) | set(predicted_groups))
    print(" " * 14 + "".join(f"{g:>12}" for g in all_cols))
    for g in true_groups:
        print(f"{g:<14}" + "".join(f"{confusion[g][g2]:>12}" for g2 in all_cols))


if __name__ == "__main__":
    main()
