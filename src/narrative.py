"""Generates a short, coach-style written summary of a graded rep by
combining the route call, coverage call, and both grades into prose --
the "read the whole picture in one paragraph" layer on top of the
individual numbers, which is what a coach actually wants out of a report,
not just a table of stats.
"""
from typing import Optional

from grading import GradeResult
from metrics import CoverageMetrics, RouteMetrics
from schema import PlayerIdentity, Situation


def _situation_phrase(situation: Situation) -> str:
    bits = []
    if situation.down and situation.distance:
        bits.append(f"On {_ordinal(situation.down)} & {situation.distance}")
    elif situation.down:
        bits.append(f"On {_ordinal(situation.down)} down")
    if situation.hash_mark:
        bits.append(f"from the {situation.hash_mark} hash")
    if situation.offense_formation:
        bits.append(f"out of {situation.offense_formation}")
    if not bits:
        return "On this rep"
    return " ".join(bits)


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= (n % 100) <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _route_phrase(route: RouteMetrics, wr_name: str) -> str:
    base = f"{wr_name} ran what looks like a {route.route_guess.split(' (')[0].lower()}"
    if route.depth_at_break_yd is not None:
        base += f", breaking at about {route.depth_at_break_yd:.0f} yards"
    elif route.total_depth_yd is not None and route.break_frame is None:
        base += f", getting to about {route.total_depth_yd:.0f} yards downfield"
    if route.confidence == "low":
        base += " (low confidence -- verify against the film)"
    return base + "."


def _coverage_phrase(coverage: CoverageMetrics, cb_name: str) -> str:
    phrase = f"{cb_name} showed {coverage.technique_guess.lower()} technique, playing what looked like {coverage.leverage_guess.lower()}"
    return phrase + "."


def _separation_phrase(route: RouteMetrics) -> Optional[str]:
    vals = [v for v in (route.separation_at_break_yd, route.separation_at_catch_yd) if v is not None]
    if not vals:
        return None
    avg = sum(vals) / len(vals)
    descriptor = "plenty of" if avg >= 2.5 else ("some" if avg >= 1.2 else "very little")
    return f"The WR created {descriptor} separation (~{avg:.1f} yd) through the rep."


def _grade_phrase(grade: GradeResult, name: str) -> str:
    lead_factor = grade.cons[0] if grade.score < 70 and grade.cons else (grade.pros[0] if grade.pros else None)
    phrase = f"{name} graded out {grade.score:.0f}/100 ({grade.letter})"
    if lead_factor:
        connector = "held back mainly by" if grade.score < 70 else "driven mainly by"
        phrase += f", {connector} \"{lead_factor.rstrip('.').lower()}\""
    return phrase + "."


def generate_summary(route: RouteMetrics, coverage: CoverageMetrics,
                      wr_grade: GradeResult, cb_grade: GradeResult,
                      wr_identity: PlayerIdentity, cb_identity: PlayerIdentity,
                      situation: Situation) -> str:
    # route_sentence starts with the WR's name (a proper noun) -- don't
    # lowercase it when splicing onto the situation clause, or a name like
    # "J. Smith-Njigba" turns into the wrong "j. Smith-Njigba".
    route_sentence = _route_phrase(route, wr_identity.name)
    parts = [
        f"{_situation_phrase(situation)}, {route_sentence}",
        _coverage_phrase(coverage, cb_identity.name),
    ]
    sep_phrase = _separation_phrase(route)
    if sep_phrase:
        parts.append(sep_phrase)
    parts.append(_grade_phrase(wr_grade, wr_identity.name))
    parts.append(_grade_phrase(cb_grade, cb_identity.name))
    return " ".join(parts)
