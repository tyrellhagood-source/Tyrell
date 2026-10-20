"""ESPN public site API provider: schedules, final scores, history, injuries.

IMPORTANT: this is ESPN's *unofficial, undocumented* public JSON API. It needs
no key and is widely used for personal research, but it has no SLA, may change
without notice, and its terms do not grant commercial redistribution rights.
For commercial use replace it with a licensed feed (SportsDataIO, Sportradar)
by implementing the same methods in another ProviderBase subclass.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from data.provider_base import ProviderBase, ProviderError
from data.schemas import HistoricalGame, InjuryReport, ScheduledGame
from utils.timeutil import parse_iso, utcnow

log = logging.getLogger(__name__)


def _score(value) -> int | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("value", value.get("displayValue"))
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_scoreboard(body: dict, sport_code: str, retrieved_at: datetime) -> list[ScheduledGame]:
    games: list[ScheduledGame] = []
    for ev in (body or {}).get("events") or []:
        for comp in ev.get("competitions") or [ev]:
            competitors = comp.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), None)
            away = next((c for c in competitors if c.get("homeAway") == "away"), None)
            if not home or not away:
                continue
            start = parse_iso(comp.get("date") or ev.get("date"))
            if start is None:
                continue
            status_type = ((comp.get("status") or ev.get("status") or {}).get("type") or {})
            state = status_type.get("state", "pre")
            completed = bool(status_type.get("completed", False))
            games.append(ScheduledGame(
                game_id=str(comp.get("id") or ev.get("id")), sport_code=sport_code, start_time=start,
                home_team=(home.get("team") or {}).get("displayName", ""),
                away_team=(away.get("team") or {}).get("displayName", ""),
                state=state, status_detail=status_type.get("detail") or status_type.get("description", ""),
                completed=completed, home_score=_score(home.get("score")), away_score=_score(away.get("score")),
                neutral_site=bool(comp.get("neutralSite", False)), source="espn", retrieved_at=retrieved_at))
    return games


def parse_injuries(body: dict, retrieved_at: datetime) -> list[InjuryReport]:
    out: list[InjuryReport] = []
    for team_block in (body or {}).get("injuries") or []:
        team = team_block.get("displayName") or (team_block.get("team") or {}).get("displayName", "")
        for inj in team_block.get("injuries") or []:
            athlete = (inj.get("athlete") or {}).get("displayName")
            if not athlete:
                continue
            out.append(InjuryReport(team=team, player=athlete, status=str(inj.get("status", "unknown")),
                                    detail=str(inj.get("shortComment") or inj.get("longComment") or ""),
                                    source="espn/injuries", retrieved_at=retrieved_at))
    return out


class ESPNProvider(ProviderBase):
    name = "espn"
    secret_params = ()

    def is_configured(self) -> bool:
        return bool(self.settings.espn_enabled)

    def _url(self, path: str, endpoint: str) -> str:
        return f"{self.settings.espn_base_url.rstrip('/')}/{path}/{endpoint}"

    def scoreboard(self, sport_code: str, path: str, start: datetime, end: datetime,
                   groups: str | None = None, ttl: int | None = None) -> list[ScheduledGame]:
        if not self.is_configured():
            raise ProviderError("ESPN provider disabled (ESPN_ENABLED=false)")
        params = {"dates": f"{start:%Y%m%d}-{end:%Y%m%d}", "limit": 1000}
        if groups:
            params["groups"] = groups
        body, meta = self.get_json(self._url(path, "scoreboard"), params,
                                   self.settings.schedule_cache_ttl_seconds if ttl is None else ttl)
        return parse_scoreboard(body, sport_code, meta.fetched_at)

    def history(self, sport_code: str, path: str, end: datetime, lookback_days: int,
                groups: str | None = None, chunk_days: int = 21) -> list[HistoricalGame]:
        """Completed games in [end - lookback, end). Old chunks are cached for
        30 days (finished games do not change); recent chunks for a few hours."""
        start = end - timedelta(days=lookback_days)
        games: dict[str, HistoricalGame] = {}
        cursor = start
        now = utcnow()
        errors = 0
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            old = (now - chunk_end) > timedelta(days=3)
            ttl = 30 * 86400 if old else self.settings.recent_history_cache_ttl_seconds
            try:
                for g in self.scoreboard(sport_code, path, cursor, chunk_end, groups, ttl):
                    if g.completed and g.home_score is not None and g.away_score is not None and g.start_time < end:
                        games[g.game_id] = HistoricalGame(g.game_id, g.start_time, g.home_team, g.away_team,
                                                          g.home_score, g.away_score, g.neutral_site, "espn")
            except ProviderError as exc:
                errors += 1
                log.warning("ESPN history chunk %s..%s failed: %s", cursor.date(), chunk_end.date(), exc)
                if errors >= 3:
                    raise
            cursor = chunk_end + timedelta(days=1)
        return sorted(games.values(), key=lambda g: g.date)

    def injuries(self, path: str) -> list[InjuryReport]:
        body, meta = self.get_json(self._url(path, "injuries"), {}, self.settings.schedule_cache_ttl_seconds)
        return parse_injuries(body, meta.fetched_at)
