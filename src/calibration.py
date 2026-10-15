"""Real pixel -> world-coordinate calibration via a homography.

Upgrades the old single-ratio calibration (still in schema.py as
`Calibration`, kept for the "I only know one distance" case) to a proper
perspective transform: give it >=4 known reference points visible in the
shot (yard-line/hash-mark intersections for football, court markings for
basketball) and it maps any pixel in that camera view to real-world
yards/feet. That matters because broadcast/sideline camera perspective
makes far-field pixels cover more ground per pixel than near-field ones --
a single px-per-yard ratio can't capture that; a homography can, as long
as the camera doesn't move mid-clip (true for a fixed sideline/tripod shot;
not true for a broadcast camera that pans/zooms mid-play -- see README).

No trained keypoint detector is wired up to find these points
automatically (that's the roboflow/sports-style model mentioned in
README.md, and it needs a labeled dataset we don't have yet). You supply
the points yourself -- read pixel coordinates off a paused frame (e.g. in
any video player, or `ffplay`/image viewer with a cursor readout) and
pair each with its known real-world position using the dimension presets
below.
"""
import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ReferencePoint:
    px: tuple  # (x, y) pixel coords in the source frame
    world: tuple  # (x, y) real-world coords, yards (football) or feet (basketball)


class FieldCalibration:
    """Homography-based pixel -> world-coordinate mapping for one fixed camera view."""

    def __init__(self, points: list):
        if len(points) < 4:
            raise ValueError(
                f"Need at least 4 reference points for a homography, got {len(points)}. "
                "For just one known distance, use schema.Calibration instead."
            )
        src = np.array([p.px for p in points], dtype=np.float32)
        dst = np.array([p.world for p in points], dtype=np.float32)
        method = cv2.RANSAC if len(points) > 4 else 0
        H, mask = cv2.findHomography(src, dst, method=method, ransacReprojThreshold=3.0)
        if H is None:
            raise ValueError("Could not solve a homography from these points -- check they aren't collinear.")
        self.H = H
        self.points = points
        self.inliers = int(mask.sum()) if mask is not None else len(points)

    def pixel_to_world(self, px: float, py: float):
        vec = np.array([px, py, 1.0])
        out = self.H @ vec
        if out[2] == 0:
            raise ValueError(
                f"Homography sends pixel ({px}, {py}) to a point at infinity -- "
                "this calibration's reference points are likely too close to collinear."
            )
        out /= out[2]
        return float(out[0]), float(out[1])

    @property
    def yards_per_pixel(self):
        """Rough single-number summary for display only -- the real conversion
        always goes through pixel_to_world(), which varies across the frame
        under perspective. Don't use this for anything but a sanity-check label.
        """
        xs = [p.px[0] for p in self.points]
        ys = [p.px[1] for p in self.points]
        wx = [p.world[0] for p in self.points]
        wy = [p.world[1] for p in self.points]
        px_span = max(max(xs) - min(xs), max(ys) - min(ys))
        world_span = max(max(wx) - min(wx), max(wy) - min(wy))
        return world_span / px_span if px_span else 0.0

    @classmethod
    def from_points_file(cls, path: str) -> "FieldCalibration":
        with open(path) as f:
            data = json.load(f)
        points = [ReferencePoint(px=tuple(p["px"]), world=tuple(p["world"])) for p in data["points"]]
        return cls(points)

    def save_points_file(self, path: str):
        with open(path, "w") as f:
            json.dump({"points": [{"px": list(p.px), "world": list(p.world)} for p in self.points]}, f, indent=2)


# -- Reference dimension presets -----------------------------------------
# Real field/court dimensions, not pixel coordinates -- you still have to
# find where each point actually lands in your frame. These just save you
# from looking up hash width / three-point radius every time you calibrate
# a new camera angle.

FOOTBALL_DIMENSIONS_YD = {
    "nfl_hash_width": 18.5 / 3,       # NFL hashes are 18'6" apart -> yards
    "ncaa_hash_width": 40.0 / 3,      # NCAA hashes are 40' apart -> yards
    "hs_hash_width": 53.33 / 3,       # most high-school hashes are at the sideline numbers -> yards (varies by state)
    "sideline_to_sideline": 53.33,    # yards (160 ft)
    "yard_line_spacing": 5.0,         # yards between successive yard-line markers
    "goal_line_to_goal_line": 100.0,  # yards
}

BASKETBALL_DIMENSIONS_FT = {
    "court_length": 94.0,             # NBA / NCAA
    "court_width": 50.0,
    "key_width": 16.0,                # NBA / NCAA lane width
    "free_throw_distance": 15.0,      # from the backboard
    "three_point_radius_nba": 23.75,
    "three_point_radius_ncaa": 22.15,
    "backboard_to_baseline": 4.0,
}
