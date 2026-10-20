"""Final-result lookup for settlement, cross-checked across sources."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from data.provider_base import ProviderError
from data.schemas import FinalScore
from data.sports_registry import SPORTS
from utils.teams import match_team
from utils.timeutil import minutes_between

log = logging.getLogger(__name__)


@dataclass
class VerifiedResult:
    status: str                    # FINAL | NOT_FINAL | CONFLICT | UNAVAILABLE | POSTPONED
    home_score: int | None = None
    away_score: int | None = None
    sources: tuple[str, ...] = ()
    detail: str = ""


class ResultsService:
    def __init__(self, espn=None, odds=None, settings=None):
        self.espn, self.odds, self.settings = espn, odds, settings
        self._scores_cache: dict[str, dict[str, FinalScore]] = {}

    def _espn_final(self, sport_code, home, away, commence: datetime):
        cfg = SPORTS.get(sport_code)
        if not cfg or not cfg.espn_path or self.espn is None:
            return None, "no ESPN coverage"
        games = self.espn.scoreboard(cfg.code, cfg.espn_path, commence - timedelta(days=1),
                                     commence + timedelta(days=1), cfg.espn_groups, ttl=300)
        for g in games:
            h, _ = match_team(home, [g.home_team])
            a, _ = match_team(away, [g.away_team])
            if h and a and abs(minutes_between(g.start_time, commence)) < 36 * 60:
                if "postpon" in g.status_detail.lower() or "cancel" in g.status_detail.lower():
                    return "POSTPONED", g.status_detail
                if not g.completed or g.home_score is None:
                    return "NOT_FINAL", g.status_detail
                return FinalScore(home, away, g.home_score, g.away_score, True, "espn", g.retrieved_at), ""
        return None, "game not found on ESPN"

    def _odds_final(self, sport_key, event_id):
        if self.odds is None or not (self.settings and self.settings.use_odds_api_scores):
            return None
        if sport_key not in self._scores_cache:
            self._scores_cache[sport_key] = self.odds.get_scores(sport_key, 3)
        return self._scores_cache[sport_key].get(event_id)

    def lookup(self, sport_code, sport_key, event_id, home, away, commence: datetime) -> VerifiedResult:
        finals: list[FinalScore] = []
        notes = []
        try:
            r, note = self._espn_final(sport_code, home, away, commence)
            if r == "POSTPONED":
                return VerifiedResult("POSTPONED", detail=note, sources=("espn",))
            if r == "NOT_FINAL":
                return VerifiedResult("NOT_FINAL", detail=note, sources=("espn",))
            if isinstance(r, FinalScore):
                finals.append(r)
            elif note:
                notes.append(note)
        except ProviderError as exc:
            notes.append(f"espn: {exc}")
        try:
            r2 = self._odds_final(sport_key, event_id)
            if r2:
                finals.append(r2)
        except ProviderError as exc:
            notes.append(f"odds-api scores: {exc}")
        if not finals:
            return VerifiedResult("UNAVAILABLE", detail="; ".join(notes))
        scores = {(f.home_score, f.away_score) for f in finals}
        if len(scores) > 1:
            return VerifiedResult("CONFLICT", detail=f"sources disagree: {scores}",
                                  sources=tuple(f.source for f in finals))
        hs, as_ = scores.pop()
        return VerifiedResult("FINAL", hs, as_, tuple(f.source for f in finals), "; ".join(notes))
