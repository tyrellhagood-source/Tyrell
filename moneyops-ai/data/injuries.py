"""Injury lookups (ESPN injuries endpoint) filtered to the two teams."""
from __future__ import annotations

from data.schemas import InjuryReport
from utils.teams import match_team


def injuries_for_teams(reports: list[InjuryReport], teams: list[str]) -> dict[str, list[InjuryReport]]:
    out: dict[str, list[InjuryReport]] = {t: [] for t in teams}
    report_teams = sorted({r.team for r in reports})
    for team in teams:
        matched, _ = match_team(team, report_teams)
        if matched:
            out[team] = [r for r in reports if r.team == matched]
    return out


SIGNIFICANT_STATUSES = {"out", "doubtful", "injured reserve", "suspension", "day-to-day", "questionable"}


def significant(reports: list[InjuryReport]) -> list[InjuryReport]:
    return [r for r in reports if r.status.strip().lower() in SIGNIFICANT_STATUSES]
