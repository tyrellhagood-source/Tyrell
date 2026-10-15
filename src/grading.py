"""Rubric-based grading: turns the heuristic metrics from metrics.py into a
0-100 grade, a letter grade, pros/cons bullets, and matched drills from
data/drills/drills_library.json.

This is a transparent, tunable rubric (not a trained model) so it's easy to
see exactly why a rep got the grade it did, and to recalibrate the weights
once real graded reps are available to compare against.

Football (WR/CB) only. Tracking, pose, and calibration (track.py,
pose_features.py, calibration.py) are sport-agnostic and already work for
basketball -- this rubric doesn't, because "break sharpness" and
"separation created" don't mean the same thing for a basketball cut, and
guessing a basketball grading rubric without knowing what a coach actually
wants scored (a cut vs. a closeout vs. a screen read, etc.) would just be
making numbers up. Write grade_offense_bball()/grade_defense_bball() here
once that's defined.
"""
import json
import os
from dataclasses import dataclass, field
from typing import Optional

from metrics import CoverageMetrics, RouteMetrics
from schema import Situation

DRILLS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "drills", "drills_library.json")

# Rough top-speed benchmarks (mph) by level, used only when the clip is
# calibrated to real-world units. These are ballpark figures, not
# position-specific combine data.
SPEED_BENCHMARKS_MPH = {
    "high_school": 17.5,
    "college": 19.5,
    "nfl": 21.0,
}

GRADE_BANDS = [
    (90, "A"), (80, "B"), (70, "C"), (60, "D"), (0, "F"),
]


def _letter(score: float) -> str:
    for threshold, letter in GRADE_BANDS:
        if score >= threshold:
            return letter
    return "F"


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def _load_drills():
    with open(DRILLS_PATH) as f:
        return json.load(f)


@dataclass
class GradeResult:
    role: str
    score: float
    letter: str
    breakdown: dict
    pros: list = field(default_factory=list)
    cons: list = field(default_factory=list)
    drills: list = field(default_factory=list)


def grade_wr(route: RouteMetrics, situation: Situation) -> GradeResult:
    drills = _load_drills()["wr"]
    breakdown = {}
    pros, cons, weaknesses = [], [], []

    # 1. Break sharpness (0-100)
    if route.break_angle_deg is not None:
        sharpness = _clamp(route.break_angle_deg / 100 * 100 + 10)
        if sharpness >= 75:
            pros.append("Crisp, sudden direction change at the break.")
        elif sharpness < 45:
            cons.append("Break looked rounded rather than a sharp cut.")
            weaknesses.append("wide_break_radius")
    else:
        sharpness = 65  # neutral -- likely a vertical route, not a real weakness signal
        pros.append("Stayed vertical through the stem (no rounded break to grade).")
    breakdown["break_sharpness"] = round(sharpness, 1)

    # 2. Separation created
    sep_values = [v for v in (route.separation_at_break_yd, route.separation_at_catch_yd) if v is not None]
    if sep_values:
        avg_sep = sum(sep_values) / len(sep_values)
        sep_score = _clamp(avg_sep / 3.0 * 100)  # 3+ yards of separation ~= max score
        if avg_sep >= 2:
            pros.append(f"Created good separation (~{avg_sep:.1f} yd).")
        elif avg_sep < 1:
            cons.append(f"Limited separation from coverage (~{avg_sep:.1f} yd).")
            weaknesses.append("low_separation_at_catch")
    else:
        sep_score = 60  # neutral -- no CB track / calibration to compare against
    breakdown["separation_created"] = round(sep_score, 1)

    # 3. Top speed vs. benchmark (only meaningful if calibrated)
    if route.top_speed_mph is not None:
        benchmark = SPEED_BENCHMARKS_MPH.get(situation.level, SPEED_BENCHMARKS_MPH["college"])
        speed_score = _clamp(route.top_speed_mph / benchmark * 100)
        if speed_score >= 100:
            pros.append(f"Plus long speed ({route.top_speed_mph:.1f} mph top speed).")
        elif speed_score < 70:
            cons.append(f"Top speed ({route.top_speed_mph:.1f} mph) is below the {situation.level} benchmark.")
            weaknesses.append("inconsistent_top_speed")
    else:
        speed_score = 65  # neutral -- uncalibrated clip, can't compare to a real mph benchmark
    breakdown["top_speed"] = round(speed_score, 1)

    score = _clamp(0.4 * breakdown["break_sharpness"] + 0.4 * breakdown["separation_created"] + 0.2 * breakdown["top_speed"])

    if not pros:
        pros.append("Completed the rep with a trackable, consistent route stem.")
    if not cons and score < 100:
        cons.append("No major flags from the tracked data -- verify against the coaching eye too.")

    matched_drills = [drills[w] for w in dict.fromkeys(weaknesses) if w in drills]
    return GradeResult(role="WR", score=round(score, 1), letter=_letter(score),
                        breakdown=breakdown, pros=pros, cons=cons, drills=matched_drills)


def grade_cb(coverage: CoverageMetrics, situation: Situation) -> GradeResult:
    drills = _load_drills()["cb"]
    breakdown = {}
    pros, cons, weaknesses = [], [], []

    # 1. Mirror / transition quality
    if coverage.mirror_correlation is not None:
        mirror_score = _clamp((coverage.mirror_correlation + 1) / 2 * 100)
        if mirror_score >= 70:
            pros.append("Mirrored the WR's lateral movement closely through the route.")
        elif mirror_score < 40:
            cons.append("Didn't mirror the WR's break well -- late transition or wrong leverage.")
            weaknesses.append("slow_transition")
    else:
        mirror_score = 60
    breakdown["mirror_transition"] = round(mirror_score, 1)

    # 2. Separation allowed (inverse of what the WR created)
    if coverage.initial_separation_yd is not None:
        # use initial separation as a coarse proxy when no explicit "sep allowed" was passed in
        sep_allowed = coverage.initial_separation_yd
        sep_score = _clamp(100 - (sep_allowed / 4.0 * 100))
        if sep_score >= 70:
            pros.append("Stayed tight to the WR, limiting the throwing window.")
        elif sep_score < 40:
            cons.append("Allowed too much cushion/separation for the coverage call.")
            weaknesses.append("high_separation_allowed")
    else:
        sep_score = 60
    breakdown["separation_allowed"] = round(sep_score, 1)

    # 3. Technique fit vs. situation (only a soft signal -- coverage_call is self-reported)
    technique_score = 65
    if situation.coverage_call:
        cc = situation.coverage_call.lower()
        if "press" in cc and "press" in coverage.technique_guess.lower():
            technique_score = 85
            pros.append("Technique matched the called coverage (press).")
        elif "press" in cc and "press" not in coverage.technique_guess.lower():
            technique_score = 45
            cons.append("Coverage call was press but alignment looked off/soft.")
            weaknesses.append("poor_press_technique")
    breakdown["technique_fit"] = technique_score

    score = _clamp(0.45 * breakdown["mirror_transition"] + 0.4 * breakdown["separation_allowed"] + 0.15 * breakdown["technique_fit"])

    if not pros:
        pros.append("Stayed in phase for a trackable rep.")
    if not cons and score < 100:
        cons.append("No major flags from the tracked data -- verify against the coaching eye too.")

    matched_drills = [drills[w] for w in dict.fromkeys(weaknesses) if w in drills]
    return GradeResult(role="CB", score=round(score, 1), letter=_letter(score),
                        breakdown=breakdown, pros=pros, cons=cons, drills=matched_drills)
