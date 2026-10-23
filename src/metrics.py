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

from schema import Track

# calibration is duck-typed: schema.Calibration or calibration.FieldCalibration,
# both exposing .pixel_to_world(px, py) -- or None for an uncalibrated clip.

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


def _to_yards(px_x, px_y, calibration):
    """calibration is either None, a schema.Calibration (single-ratio
    fallback), or a calibration.FieldCalibration (homography, perspective-
    correct) -- both of the latter two expose .pixel_to_world(px, py)."""
    if calibration is None:
        return px_x, px_y
    return calibration.pixel_to_world(px_x, px_y)


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
    route_family: Optional[str] = None  # machine-readable tag: "go","slant","out","in","dig",
    # "curl","comeback","hitch","post","corner","double_move","unknown" -- for programmatic use
    # (grading, validation scripts); route_guess is the human-readable version of the same call.
    break_count: int = 0
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


def _speed_series(track: Track, calibration: Optional[object]):
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


def _robust_top_speed(speeds, window=5):
    """Max of the *smoothed* speed series, not the raw max.

    A single bad frame -- the tracker still locking on right as a track
    starts, a brief detection jump across an occlusion -- turns into a
    one-frame speed spike that's physically impossible (confirmed on real
    data: a reported "32 mph" top speed for an NFL WR, track-opening-frame
    artifact, versus a real sustained burst around 20-23 mph visible a
    moment later). A real sprint burst is fast for several consecutive
    frames, not one, so smoothing over a small window keeps genuine bursts
    while flattening single-frame artifacts.
    """
    if not speeds:
        return None
    return max(_smooth(speeds, window=window))


def _direction_series(track: Track, calibration: Optional[object]):
    """Per-step (dx, dy) displacement vectors in yards (or px if uncalibrated)."""
    vecs = []
    pts = track.points
    for a, b in zip(pts, pts[1:]):
        ax, ay = _to_yards(*a.centroid, calibration)
        bx, by = _to_yards(*b.centroid, calibration)
        vecs.append((bx - ax, by - ay))
    return vecs


def _release_axis(vecs, n=15):
    """Average direction of the first n movement steps -> the 'downfield' unit vector.

    n=15, not the first handful: on real tracked video the first couple of
    steps right as a track starts can be dominated by the tracker's box
    still settling (confirmed on real data -- the first 2 steps of an NFL
    WR's tracked release were nearly pure lateral motion, not downfield at
    all, visibly skewing a short-window axis estimate off the direction
    the route actually ran). Averaging over more of the release is a more
    stable estimate of the true downfield direction.
    """
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


def find_break_point(track: Track, calibration: Optional[object], min_depth_yd: float = 1.0):
    """Finds the frame with the sharpest sustained direction change after the
    stem has some depth. Returns (index_into_points, angle_deg) or (None, None).

    A heading averaged over a window where the player is wobbling in place
    (adjusting footwork/posture, e.g. settling under a catch) rather than
    moving ballistically is almost pure noise, and can produce a huge
    spurious "angle change" that has nothing to do with a real cut.
    min_depth_yd alone doesn't catch this: without calibration it's compared
    against raw pixel sums, which clear it almost immediately, so it's
    effectively a no-op on an uncalibrated clip. The straightness gate below
    is calibration-independent and catches exactly this case (confirmed
    against a real broadcast clip where the depth-only version flagged a
    ~180 degree "break" at the moment of the catch, not an actual route
    break -- that window's net displacement was a small fraction of its
    total step distance, the signature of wobble rather than travel).
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

    best_i, best_delta = _find_single_break(vecs, depths, 0, len(vecs), min_depth_yd, calibrated=calibration is not None)
    return best_i, best_delta


def _straightness(vec_slice):
    """1.0 = clean ballistic motion in one direction; near 0 = wobbling
    (noisy detection jitter, or a player adjusting posture/footwork
    without much net translation -- e.g. settling under a catch). A
    heading average computed over a wobbly window isn't a real direction,
    so it shouldn't be trusted as one side of a 'break'."""
    sx = sum(v[0] for v in vec_slice)
    sy = sum(v[1] for v in vec_slice)
    net = math.hypot(sx, sy)
    total = sum(math.hypot(*v) for v in vec_slice)
    return net / total if total > 0 else 0.0


def _find_single_break(vecs, depths, start, end, min_depth_yd, window=5, calibrated=False):
    """Finds the sharpest sustained direction change within vecs[start:end].
    Returns (index_into_vecs, angle_deg) or (None, None). Shared by
    find_break_point (first break) and _find_breaks (subsequent breaks, for
    double-move detection) so both get the same noise filtering.

    A heading averaged over a wobbly window isn't a real heading -- see
    _straightness. min_depth_yd alone doesn't gate out early noise on an
    uncalibrated clip (raw pixel sums clear a "1.0 yard" threshold
    immediately), so a frame-count-based minimum (first ~15% of the
    *segment*) backs it up.

    The angle cap below is deliberately *not* a flat "no real route reverses
    this sharply" rule -- a hitch/curl/comeback genuinely does, planting and
    running almost straight back the way it came (confirmed against a
    synthetic comeback trajectory that a flat ~150 degree cap wrongly
    rejected outright). The real distinguishing signal is whether there's
    trustworthy depth evidence of a real release/stem first: with a
    calibration (so min_depth_yd means real yards, not a no-op pixel sum),
    a long reversal after an established stem is plausible and allowed up to
    175 degrees; without one, there's no reliable way to confirm a stem
    happened at all, so it stays capped at 150 as a safety margin against
    camera pan corrupting the pixel-space trajectory (confirmed against a
    real broadcast clip where the uncapped, depth-gate-is-a-no-op version
    confidently flagged a vertical go route as a sharp break).
    """
    headings = [math.atan2(v[1], v[0]) for v in vecs]
    seg_len = end - start
    min_elapsed = start + max(window, int(seg_len * 0.15))
    angle_cap = 175 if calibrated else 150
    best_i, best_delta = None, 0.0
    for i in range(max(min_elapsed, start + window), min(end, len(headings)) - window):
        if depths[i] - depths[start] < min_depth_yd:
            continue
        before = headings[i - window:i]
        after = headings[i:i + window]
        if min(_straightness(vecs[i - window:i]), _straightness(vecs[i:i + window])) < 0.5:
            continue
        avg_before = math.atan2(sum(math.sin(a) for a in before), sum(math.cos(a) for a in before))
        avg_after = math.atan2(sum(math.sin(a) for a in after), sum(math.cos(a) for a in after))
        delta = abs(math.degrees(math.atan2(math.sin(avg_after - avg_before), math.cos(avg_after - avg_before))))
        if delta > best_delta and delta <= angle_cap:
            best_delta, best_i = delta, i

    if best_i is None or best_delta < 25:
        return None, None
    return best_i, best_delta


def _find_breaks(track: Track, calibration, min_depth_yd: float = 1.0, max_breaks: int = 2):
    """Like find_break_point, but keeps looking for additional breaks
    (double moves -- out-and-up, slant-and-go, etc). Returns a list of
    (index, angle_deg) in time order.

    Uses binary segmentation, not a linear "keep searching after the last
    break": searching only the remainder *after* a found break missed a
    real double move in testing -- when the segment between two close-
    together transitions is short, a single whole-range search finds one
    *blended* break roughly between them instead of either real one, and
    then there's nothing break-shaped left in the remainder to find a
    second break in. Recursing into the sub-ranges on *both* sides of each
    found break (not just the right side) resolves the same blended
    region into its two real, cleaner transitions.
    """
    vecs = _direction_series(track, calibration)
    if len(vecs) < 8:
        return []
    axis = _release_axis(vecs)
    depths = []
    running = 0.0
    for v in vecs:
        running += _project(v, axis)
        depths.append(running)

    window = 5
    calibrated = calibration is not None
    found = []
    ranges_to_search = [(0, len(vecs))]
    while ranges_to_search and len(found) < max_breaks:
        start, end = ranges_to_search.pop(0)
        if end - start < 2 * window + 3:
            continue
        i, delta = _find_single_break(vecs, depths, start, end, min_depth_yd, window=window, calibrated=calibrated)
        if i is None:
            continue
        found.append((i, delta))
        ranges_to_search.append((start, i))
        ranges_to_search.append((i + window, end))

    found.sort(key=lambda b: b[0])
    return found[:max_breaks]


def _post_break_trend(vecs, axis, break_i, window=8):
    """Is the player still gaining depth after the break ('continuing'),
    losing it ('comeback' -- coming back toward the LOS), or moving mostly
    sideways ('flat')? Compares the average per-step depth gain just after
    the break to just before it."""
    before = vecs[max(0, break_i - window):break_i]
    after = vecs[break_i:break_i + window]
    if not before or not after:
        return "flat"
    pre_rate = sum(_project(v, axis) for v in before) / len(before)
    post_rate = sum(_project(v, axis) for v in after) / len(after)
    if post_rate < -0.15 * abs(pre_rate) - 1e-6 or post_rate < 0 and pre_rate > 0:
        return "comeback"
    if post_rate > 0.4 * pre_rate:
        return "continuing"
    return "flat"


def _lateral_direction(vecs, axis, break_i, wr_alignment, window=8):
    """Which way (toward the WR's own sideline = 'out', or toward the
    middle of the field = 'in') did the break go? Only answerable when the
    caller tells us which side of the screen the WR lined up on
    (wr_alignment) -- screen-space geometry alone can't say which side is
    the sideline. Returns 'out' | 'in' | None (can't tell)."""
    if wr_alignment not in ("left", "right"):
        return None
    perp = _perp(axis)
    before = vecs[max(0, break_i - window):break_i]
    after = vecs[break_i:break_i + window]
    if not before or not after:
        return None
    pre_lat = sum(_project(v, perp) for v in before) / len(before)
    post_lat = sum(_project(v, perp) for v in after) / len(after)
    delta_lat = post_lat - pre_lat
    # perp = (-axis_y, axis_x); its sign relative to screen-x depends on the
    # release axis's own orientation, so recover a screen-x-relative sign
    # directly instead of assuming one: project perp onto the screen x-axis.
    perp_points_screen_right = perp[0] > 0
    moves_screen_right = (delta_lat > 0) == perp_points_screen_right
    if abs(delta_lat) < 1e-6:
        return None
    broke_toward_own_side = (wr_alignment == "left" and not moves_screen_right) or \
                             (wr_alignment == "right" and moves_screen_right)
    return "out" if broke_toward_own_side else "in"


def classify_route(track: Track, calibration: Optional[object], wr_alignment: Optional[str] = None) -> RouteMetrics:
    vecs = _direction_series(track, calibration)
    notes = []
    if len(vecs) < 8:
        return RouteMetrics("Unknown (clip too short)", "low", None, None, None, None, None, None,
                             notes=["Need more tracked frames to call a route."])

    axis = _release_axis(vecs)
    total_depth = sum(_project(v, axis) for v in vecs)

    breaks = _find_breaks(track, calibration, max_breaks=2)
    break_i, break_angle = breaks[0] if breaks else (None, None)
    depth_at_break = sum(_project(v, axis) for v in vecs[:break_i]) if break_i is not None else None

    speeds = [s for _, s in _speed_series(track, calibration)]
    avg_speed = sum(speeds) / len(speeds) if speeds else None
    top_speed = _robust_top_speed(speeds)
    if calibration:
        avg_speed_mph = avg_speed * MPH_PER_YPS if avg_speed else None
        top_speed_mph = top_speed * MPH_PER_YPS if top_speed else None
    else:
        avg_speed_mph = top_speed_mph = None
        notes.append("No calibration provided -- speed shown in px/s equivalents only, not mph.")
    if not calibration:
        notes.append("No calibration provided -- route depth thresholds are in raw pixels, not yards; "
                      "family/name calls below are angle-and-trend only, not depth-aware.")

    def depth_bucket(depth):
        """Buckets only mean something with real units -- returns None
        uncalibrated. Thresholds follow typical route-tree depth, not an
        even split: shallow routes (slant/flat/hitch) break under ~6 yd,
        intermediate (dig/in/curl) in the 6-12 yd range, post/corner-depth
        routes break beyond that."""
        if depth is None or not calibration:
            return None
        if depth < 6:
            return "shallow"
        if depth < 12:
            return "intermediate"
        return "deep"

    if break_i is None:
        route_family = "go"
        route = "Go / Fly (vertical, no break detected)"
        confidence = "medium" if calibration else "low"
        depth_note = f"{total_depth:.1f} yd" if calibration else "uncalibrated depth"
        notes.append(f"Ran vertically for ~{depth_note} with no sustained direction change.")
        if calibration and total_depth < 5:
            notes.append("Net depth over the whole tracked clip is small/negative -- if this track "
                          "extends past the catch into a tackle, that's expected (getting driven "
                          "back counts against net depth), not a tracking error. Trim the clip to "
                          "the route stem before the catch for a cleaner depth number.")
    else:
        trend = _post_break_trend(vecs, axis, break_i)
        lateral = _lateral_direction(vecs, axis, break_i, wr_alignment)
        bucket = depth_bucket(depth_at_break)
        confidence = "medium" if calibration else "low"

        if len(breaks) >= 2:
            route_family = "double_move"
            route = "Double move (two sustained direction changes -- e.g. slant-and-go / out-and-up)"
        elif trend == "comeback":
            if bucket in (None, "shallow"):
                route_family, name = "hitch", "Hitch"
            elif bucket == "intermediate":
                route_family, name = "curl", "Curl"
            else:
                route_family, name = "comeback", "Comeback"
            route = f"{name} (breaks back toward the LOS)"
        else:
            # trend == "continuing" or "flat": family depends on depth + (if we
            # know it) which way relative to the WR's own sideline.
            if bucket in (None, "shallow"):
                if lateral == "out":
                    route_family, route = "out", "Out (shallow, toward the sideline)"
                elif lateral == "in":
                    route_family, route = "slant", "Slant (shallow, toward the middle)"
                else:
                    route_family, route = "shallow_break", "Slant / Out (shallow breaking route -- side unknown, pass --wr-alignment to tell them apart)"
            elif bucket == "intermediate":
                if lateral == "out":
                    route_family, route = "out", "Out / Comeback-out (intermediate, toward the sideline)"
                elif lateral == "in":
                    route_family, route = "dig", "Dig / In (intermediate, toward the middle)"
                else:
                    route_family, route = "intermediate_break", "Dig / Out (intermediate breaking route -- side unknown, pass --wr-alignment to tell them apart)"
            else:
                if lateral == "out":
                    route_family, route = "corner", "Corner (deep, toward the sideline)"
                elif lateral == "in":
                    route_family, route = "post", "Post (deep, toward the middle)"
                else:
                    route_family, route = "deep_break", "Post / Corner (deep breaking route -- side unknown, pass --wr-alignment to tell them apart)"

    return RouteMetrics(
        route_guess=route,
        confidence=confidence,
        break_frame=track.points[break_i].frame if break_i is not None else None,
        break_angle_deg=round(break_angle, 1) if break_angle else None,
        depth_at_break_yd=round(depth_at_break, 2) if (calibration and depth_at_break is not None) else None,
        total_depth_yd=round(total_depth, 2) if calibration else None,
        avg_speed_mph=round(avg_speed_mph, 1) if avg_speed_mph else None,
        top_speed_mph=round(top_speed_mph, 1) if top_speed_mph else None,
        route_family=route_family,
        break_count=len(breaks),
        notes=notes,
    )


def _separation_series(wr: Track, cb: Track, calibration: Optional[object]):
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


def classify_coverage(cb: Track, wr: Track, calibration: Optional[object]) -> CoverageMetrics:
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
    top_speed = _robust_top_speed(speeds)
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


def separation_at(wr: Track, cb: Track, calibration: Optional[object], frame: Optional[int]):
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
