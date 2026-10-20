"""Independent schedule lookups used by the fact-checker."""
from __future__ import annotations

from datetime import datetime, timedelta

from data.provider_base import ProviderError
from data.schemas import Event, ScheduledGame
from data.sports_registry import SportConfig
from utils.teams import match_team
from utils.timeutil import minutes_between


def fetch_schedule(espn, cfg: SportConfig, start: datetime, end: datetime) -> list[ScheduledGame]:
    if not cfg.espn_path:
        raise ProviderError(f"No independent schedule source for {cfg.code}")
    return espn.scoreboard(cfg.code, cfg.espn_path, start - timedelta(days=1), end + timedelta(days=1),
                           cfg.espn_groups)


def find_scheduled_game(event: Event, schedule: list[ScheduledGame]) -> tuple[ScheduledGame | None, dict]:
    """Match an odds event to an independently sourced game by team names
    and start time. Returns (game, match details)."""
    best, best_detail = None, {"matched": False}
    for g in schedule:
        h, hs = match_team(event.home_team, [g.home_team])
        a, as_ = match_team(event.away_team, [g.away_team])
        swapped = False
        if not (h and a):
            h2, hs2 = match_team(event.home_team, [g.away_team])
            a2, as2 = match_team(event.away_team, [g.home_team])
            if h2 and a2:
                h, a, hs, as_, swapped = h2, a2, hs2, as2, True
            else:
                continue
        diff = abs(minutes_between(g.start_time, event.commence_time))
        if diff > 36 * 60:
            continue
        if best is None or diff < best_detail["start_diff_minutes"]:
            best = g
            best_detail = {"matched": True, "espn_game_id": g.game_id, "home_similarity": hs,
                           "away_similarity": as_, "home_away_swapped": swapped,
                           "start_diff_minutes": round(diff, 1)}
    return best, best_detail
