"""Trajectory-based route/coverage metrics and heuristic classification.

Everything here works off tracked centroid positions over time (pixels or,
if a Calibration is supplied, yards). None of this is a trained route/
coverage-recognition model -- it's a transparent, tunable heuristic so the
tracking + grading loop is testable today. Swapping in a trained classifier
later (given labeled reps) is a drop-in replacement for classify_route() /
classify_coverage().
"""
import math
from dataclasses import dataclass, field
from typing import Optional

from schema import Calibration, Track

MPH_PER_YPS = 2.04545  # yards/sec -> mph


def _smooth(series, window=5):
    if len(series) < window:
        return series[:]
    out = []
    half = window // 2
    for i in range(len(series)):
        lo, hi = max(0, i - half), min(len(series), i + half + 1)
        out.append(sum(series[lo:hi]) / (hi - lo))
    return out


def _to_yards(px_x, px_y, calibration: Optional[Calibration]):
    if calibration is None:
        return px_x, px_y
    ypp = calibration.yards_per_pixel
    return px_x * ypp, px_y * ypp


@dataclass
class RouteMetrics:
    route_guess: str
    confidence: str  # "low" | "medium" | "high" -- heuristic, be honest about it
    break_frame: Optional[int]
    break_angle_deg: Optional[float]
    depth_at_break_yd: Optional[float]
    total_depth_yd: Optional[float]
    avg_speed_mph: Optional[float]
    top_speed_mph: Optional[float]
    separation_at_break_yd: Optional[float] = None
    separation_at_catch_yd: Optional[float] = None
    notes: list = field(default_factory=list)


@dataclass
class CoverageMetrics:
    technique_guess: str  # "Press" | "Off" | "Soft"
    leverage_guess: str  # "Man (likely)" | "Zone (likely)" | "Unclear"
    initial_separation_yd: Optional[float]
    mirror_correlation: Optional[float]  # -1..1, higher = tracks WR's lateral movement closely
    avg_speed_mph: Optional[float]
    top_speed_mph: Optional[float]
    notes: list = field(default_factory=list)


def _speed_series(track: Track, calibration: Optional[Calibration]):
    """Returns list of (t, speed_yps) between consecutive points."""
    speeds = []
    pts = track.points
    for a, b in zip(pts, pts[1:]):
        dt = b.t - a.t
        if dt <= 0:
            continue
        ax, ay = _to_yards(*a.centroid, calibration)
        bx, by = _to_yards(*b.centroid, calibration)
        dist = math.hypot(bx - ax, by - ay)
        speeds.append((b.t, dist / dt))
    return speeds


def _direction_series(track: Track, calibration: Optional[Calibration]):
    """Per-step (dx, dy) displacement vectors in yards (or px if uncalibrated)."""
    vecs = []
    pts = track.points
    for a, b in zip(pts, pts[1:]):
        ax, ay = _to_yards(*a.centroid, calibration)
        bx, by = _to_yards(*b.centroid, calibration)
        vecs.append((bx - ax, by - ay))
    return vecs


def _release_axis(vecs, n=5):
    """Average direction of the first n movement steps -> the 'downfield' unit vector."""
    sample = vecs[:n] if len(vecs) >= n else vecs
    sx = sum(v[0] for v in sample)
    sy = sum(v[1] for v in sample)
    mag = math.hypot(sx, sy)
    if mag == 0:
        return (0.0, 1.0)
    return (sx / mag, sy / mag)


def _project(vec, axis):
    return vec[0] * axis[0] + vec[1] * axis[1]


def _perp(axis):
    return (-axis[1], axis[0])


def find_break_point(track: Track, calibration: Optional[Calibration], min_depth_yd: float = 1.0):
    """Finds the frame with the sharpest sustained direction change after the
    stem has some depth. Returns (index_into_points, angle_deg) or (None, None).
    """
    vecs = _direction_series(track, calibration)
    if len(vecs) < 8:
        return None, None

    axis = _release_axis(vecs)
    depths = []
    running = 0.0
    for v in vecs:
        running += _project(v, axis)
        depths.append(running)

    # smooth headings to avoid frame-to-frame jitter from detection noise
    headings = [math.atan2(v[1], v[0]) for v in vecs]
    window = 5
    best_i, best_delta = None, 0.0
    for i in range(window, len(headings) - window):
        if depths[i] < min_depth_yd:
            continue
        before = headings[i - window:i]
        after = headings[i:i + window]
        avg_before = math.atan2(sum(math.sin(a) for a in before), sum(math.cos(a) for a in before))
        avg_after = math.atan2(sum(math.sin(a) for a in after), sum(math.cos(a) for a in after))
        delta = abs(math.degrees(math.atan2(math.sin(avg_after - avg_before), math.cos(avg_after - avg_before))))
        if delta > best_delta:
            best_delta, best_i = delta, i

    if best_i is None or best_delta < 25:  # no real break detected -- treat as a vertical/go route
        return None, None
    return best_i, best_delta


def classify_route(track: Track, calibration: Optional[Calibration]) -> RouteMetrics:
    vecs = _direction_series(track, calibration)
    notes = []
    if len(vecs) < 8:
        return RouteMetrics("Unknown (clip too short)", "low", None, None, None, None, None, None,
                             notes=["Need more tracked frames to call a route."])

    axis = _release_axis(vecs)
    total_depth = sum(_project(v, axis) for v in vecs)
    unit_scale = "yd" if calibration else "px"

    break_i, break_angle = find_break_point(track, calibration)
    depth_at_break = None
    if break_i is not None:
        depth_at_break = sum(_project(v, axis) for v in vecs[:break_i])

    speeds = [s for _, s in _speed_series(track, calibration)]
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    top_speed = max(speeds) if speeds else None
    if calibration:
        avg_speed_mph = avg_speed * MPH_PER_YPS if avg_speed else None
        top_speed_mph = top_speed * MPH_PER_YPS if top_speed else None
    else:
        avg_speed_mph = top_speed_mph = None
        notes.append("No calibration provided -- speed shown in px/s equivalents only, not mph.")

    if break_i is None:
        depth_note = f"{total_depth:.1f} {unit_scale}" if calibration else "uncalibrated"
        route = "Go / Fade / Vertical seam"
        confidence = "medium" if calibration else "low"
        notes.append(f"No sharp break detected; ran vertically for ~{depth_note}.")
    else:
        depth_yd = depth_at_break if calibration else None
        if not calibration:
            notes.append("No calibration provided -- route depth/angle thresholds are approximate.")
        # crude depth buckets in yards; fall back to angle-only when uncalibrated
        if calibration and depth_at_break is not None:
            if depth_at_break < 6:
                route = "Slant / Out (shallow breaking route)"
            elif depth_at_break < 12:
                route = "Dig / Out (intermediate breaking route)"
            else:
                route = "Post / Corner (deep breaking route)"
        else:
            route = "Breaking route (depth unknown without calibration)"

        if break_angle is not None and break_angle < 50:
            route += " -- soft angle, could be a curl/comeback"
        confidence = "medium" if calibration else "low"

    return RouteMetrics(
        route_guess=route,
        confidence=confidence,
        break_frame=track.points[break_i].frame if break_i is not None else None,
        break_angle_deg=round(break_angle, 1) if break_angle else None,
        depth_at_break_yd=round(depth_at_break, 2) if (calibration and depth_at_break is not None) else None,
        total_depth_yd=round(total_depth, 2) if calibration else None,
        avg_speed_mph=round(avg_speed_mph, 1) if avg_speed_mph else None,
        top_speed_mph=round(top_speed_mph, 1) if top_speed_mph else None,
        notes=notes,
    )


def _separation_series(wr: Track, cb: Track, calibration: Optional[Calibration]):
    """Returns list of (t, distance) for frames both tracks have a point at
    (matched by nearest timestamp)."""
    cb_by_frame = {p.frame: p for p in cb.points}
    out = []
    for wp in wr.points:
        cp = cb_by_frame.get(wp.frame)
        if cp is None:
            continue
        ax, ay = _to_yards(*wp.centroid, calibration)
        bx, by = _to_yards(*cp.centroid, calibration)
        out.append((wp.t, math.hypot(bx - ax, by - ay)))
    return out


def classify_coverage(cb: Track, wr: Track, calibration: Optional[Calibration]) -> CoverageMetrics:
    notes = []
    sep_series = _separation_series(wr, cb, calibration)
    initial_sep = sep_series[0][1] if sep_series else None

    if not calibration:
        notes.append("No calibration provided -- press/off threshold is a rough pixel-based guess.")

    if initial_sep is None:
        technique = "Unknown"
    elif calibration:
        technique = "Press" if initial_sep <= 1.5 else ("Off" if initial_sep <= 7 else "Soft")
    else:
        technique = "Press (close)" if initial_sep < 60 else "Off (cushioned)"  # px fallback

    # mirror correlation: does the CB's lateral movement track the WR's lateral movement?
    wr_vecs = _direction_series(wr, calibration)
    cb_vecs = _direction_series(cb, calibration)
    axis = _release_axis(wr_vecs) if wr_vecs else (1.0, 0.0)
    perp = _perp(axis)

    n = min(len(wr_vecs), len(cb_vecs))
    corr = None
    if n >= 8:
        wr_lat = [_project(v, perp) for v in wr_vecs[:n]]
        cb_lat = [_project(v, perp) for v in cb_vecs[:n]]
        mw = sum(wr_lat) / n
        mc = sum(cb_lat) / n
        num = sum((wr_lat[i] - mw) * (cb_lat[i] - mc) for i in range(n))
        dw = math.sqrt(sum((x - mw) ** 2 for x in wr_lat))
        dc = math.sqrt(sum((x - mc) ** 2 for x in cb_lat))
        if dw > 0 and dc > 0:
            corr = num / (dw * dc)

    if corr is None:
        leverage = "Unclear (not enough overlapping frames)"
    elif corr > 0.5:
        leverage = "Man (likely) -- mirrors WR's lateral movement"
    elif corr < 0.1:
        leverage = "Zone (likely) -- doesn't mirror the WR's break"
    else:
        leverage = "Unclear"

    speeds = [s for _, s in _speed_series(cb, calibration)]
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    top_speed = max(speeds) if speeds else None
    if calibration and avg_speed is not None:
        avg_speed_mph, top_speed_mph = avg_speed * MPH_PER_YPS, top_speed * MPH_PER_YPS
    else:
        avg_speed_mph = top_speed_mph = None

    return CoverageMetrics(
        technique_guess=technique,
        leverage_guess=leverage,
        initial_separation_yd=round(initial_sep, 2) if (calibration and initial_sep is not None) else None,
        mirror_correlation=round(corr, 2) if corr is not None else None,
        avg_speed_mph=round(avg_speed_mph, 1) if avg_speed_mph else None,
        top_speed_mph=round(top_speed_mph, 1) if top_speed_mph else None,
        notes=notes,
    )


def separation_at(wr: Track, cb: Track, calibration: Optional[Calibration], frame: Optional[int]):
    if frame is None:
        return None
    cb_by_frame = {p.frame: p for p in cb.points}
    wr_by_frame = {p.frame: p for p in wr.points}
    wp, cp = wr_by_frame.get(frame), cb_by_frame.get(frame)
    if wp is None or cp is None:
        return None
    ax, ay = _to_yards(*wp.centroid, calibration)
    bx, by = _to_yards(*cp.centroid, calibration)
    return math.hypot(bx - ax, by - ay)
