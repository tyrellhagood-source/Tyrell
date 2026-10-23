"""Shared data structures used across the tracking/grading pipeline."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrackPoint:
    frame: int
    t: float  # seconds
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    centroid: tuple  # (cx, cy) in pixels
    keypoints: Optional[dict] = None  # MediaPipe pose landmarks, pixel coords


@dataclass
class Track:
    track_id: int
    points: list = field(default_factory=list)  # list[TrackPoint]

    def as_xy_series(self):
        return [(p.t, p.centroid[0], p.centroid[1]) for p in self.points]


@dataclass
class Calibration:
    """Simple fallback: maps pixels to real-world yards using a single known
    distance (e.g. hash-to-hash), assuming a uniform scale across the frame.
    No perspective correction -- for that, use calibration.FieldCalibration
    with 4+ reference points instead.
    """
    px_distance: float
    yard_distance: float

    @property
    def yards_per_pixel(self) -> float:
        if self.px_distance <= 0:
            return 0.0
        return self.yard_distance / self.px_distance

    def pixel_to_world(self, px: float, py: float):
        """Same duck-typed interface as FieldCalibration.pixel_to_world, so
        metrics.py can treat either calibration type identically. Differencing
        two of these cancels the arbitrary origin, so a uniform scale from
        (0, 0) is fine for distance/speed purposes even though it isn't a
        real coordinate mapping.
        """
        ypp = self.yards_per_pixel
        return px * ypp, py * ypp


@dataclass
class Situation:
    """Manually supplied game/rep context. Not derived from video -- input by the user."""
    sport: str = "football"  # "football" | "basketball"
    down: Optional[int] = None
    distance: Optional[int] = None
    hash_mark: Optional[str] = None  # "left" | "middle" | "right"
    offense_formation: Optional[str] = None
    coverage_call: Optional[str] = None  # if known/self-reported, e.g. "Cover 1" or "switch"
    wr_alignment: Optional[str] = None  # "left" | "right" -- screen-side at the snap, disambiguates In/Out, Post/Corner
    quarter_or_period: Optional[int] = None  # basketball
    shot_clock: Optional[float] = None  # basketball, seconds remaining
    level: str = "college"  # "high_school" | "college" | "nfl" / "nba" -- used for grading benchmarks
    notes: Optional[str] = None


@dataclass
class PlayerIdentity:
    track_id: int
    role: str  # "WR" or "CB"
    name: str
    number: Optional[str] = None
    team: Optional[str] = None
