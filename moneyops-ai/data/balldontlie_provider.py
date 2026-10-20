"""balldontlie provider (https://www.balldontlie.io) - FREE plan features.

Free plan (per balldontlie's pricing, 2026): teams, players and GAMES only,
5 requests/minute, per sport. MoneyOps uses the games endpoint as:
* a third independent schedule/status source for the fact-checker (Agent 2),
* a third final-score source for settlement (cross-checked),
* a fallback season-results source for the team model.
Game stats, injuries, box scores and odds need paid tiers (All-Star / GOAT)
and are not called.

Auth: the API key is sent in the `Authorization` header (never in the URL),
registered for log redaction, and read only from BALLDONTLIE_API_KEY.

Request format (NBA, documented): GET /v1/games?start_date=YYYY-MM-DD
&end_date=YYYY-MM-DD&per_page=100[&cursor=N] -> {"data": [...],
"meta": {"next_cursor": N}}. Other sports use /<sport>/v1/games with the
same shape; their field names are parsed defensively (fail closed).
Throttled to BALLDONTLIE_REQUESTS_PER_MINUTE (default 5) and capped per run.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from data.provider_base import ProviderBase, ProviderBudgetExceeded, ProviderError, ProviderNotConfigured
from data.schemas import HistoricalGame, ScheduledGame
from utils.security import register_secret
from utils.timeutil import parse_iso, utcnow

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
CANCELLED = ("postpon", "cancel", "suspend")
FINAL = ("final",)


def _team_name(t) -> str | None:
    if not isinstance(t, dict):
        return None
    if t.get("full_name"):
        return str(t["full_name"])
    parts = [t.get("city") or t.get("location"), t.get("name")]
    name = " ".join(str(p) for p in parts if p)
    return name or None


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def parse_game(item: dict, sport_code: str, retrieved_at: datetime) -> ScheduledGame | None:
    home = _team_name(item.get("home_team"))
    away = _team_name(item.get("visitor_team") or item.get("away_team"))
    gid = item.get("id")
    if not (home and away and gid is not None):
        return None
    status = str(item.get("status") or "").strip()
    start = parse_iso(item.get("datetime"))
    if start is None and "T" in status:
        start = parse_iso(status)              # scheduled NBA games carry the tip-off time in `status`
    raw = str(item.get("date") or "")
    if start is None and "T" in raw and not raw.endswith(("T00:00:00.000Z", "T00:00:00Z")):
        start = parse_iso(raw)                 # e.g. NFL kick-off timestamps
    time_known = start is not None
    if start is None:
        if len(raw) < 10:
            return None
        try:                                   # date only: noon US Eastern keeps the calendar date right
            start = datetime.strptime(raw[:10], "%Y-%m-%d").replace(hour=12, tzinfo=ET)
        except ValueError:
            return None
    low = status.lower()
    hs = _int(item.get("home_team_score"))
    vs = _int(item.get("visitor_team_score", item.get("away_team_score")))
    cancelled = any(w in low for w in CANCELLED)
    completed = low.startswith(FINAL) and hs is not None and vs is not None
    scheduled = (not low) or "T" in status or low.endswith(" et") or "scheduled" in low
    state = "post" if (completed or cancelled) else ("pre" if scheduled else "in")
    return ScheduledGame(
        game_id=f"bdl:{gid}", sport_code=sport_code, start_time=start.astimezone(timezone.utc),
        home_team=home, away_team=away, state=state, status_detail=status, completed=completed,
        home_score=hs if completed else None, away_score=vs if completed else None,
        neutral_site=bool(item.get("neutral_site", False)), source="balldontlie", retrieved_at=retrieved_at,
        time_known=time_known)


class BallDontLieProvider(ProviderBase):
    name = "balldontlie"
    secret_params = ()

    def __init__(self, settings, cache=None, session=None, clock=time.monotonic, sleeper=time.sleep):
        super().__init__(settings, cache, session)
        register_secret(settings.balldontlie_api_key)
        self.requests_this_run = 0
        self._clock, self._sleep, self._last = clock, sleeper, None

    def is_configured(self) -> bool:
        return bool(self.settings.balldontlie_api_key)

    def _before_request(self) -> None:
        if self.requests_this_run >= self.settings.balldontlie_max_requests_per_run:
            raise ProviderBudgetExceeded(
                f"balldontlie request budget ({self.settings.balldontlie_max_requests_per_run}/run) reached")
        gap = 60.0 / max(1, self.settings.balldontlie_requests_per_minute)
        if self._last is not None:
            wait = gap - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        self.requests_this_run += 1

    def _url(self, path: str) -> str:
        return f"{self.settings.balldontlie_base_url.rstrip('/')}/{path.strip('/')}/games"

    def games(self, sport_code: str, path: str, start: datetime, end: datetime, ttl: int | None = None,
              max_pages: int = 20) -> list[ScheduledGame]:
        if not self.is_configured():
            raise ProviderNotConfigured("BALLDONTLIE_API_KEY is not set")
        start_d = start.astimezone(ET).date().isoformat()
        end_d = end.astimezone(ET).date().isoformat()
        out, cursor = [], None
        headers = {"Authorization": self.settings.balldontlie_api_key}
        for _ in range(max_pages):
            params = {"start_date": start_d, "end_date": end_d, "per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor
            body, meta = self.get_json(self._url(path), params,
                                       self.settings.schedule_cache_ttl_seconds if ttl is None else ttl, headers)
            if not isinstance(body, dict) or not isinstance(body.get("data"), list):
                raise ProviderError(f"balldontlie {path}: unexpected payload")
            for item in body["data"]:
                g = parse_game(item, sport_code, meta.fetched_at) if isinstance(item, dict) else None
                if g is not None:
                    out.append(g)
            cursor = (body.get("meta") or {}).get("next_cursor")
            if not cursor:
                return out
        raise ProviderError(f"balldontlie {path}: more than {max_pages} pages for {start_d}..{end_d}")

    def history(self, sport_code: str, path: str, end: datetime, lookback_days: int,
                chunk_days: int = 31) -> list[HistoricalGame]:
        out: dict[str, HistoricalGame] = {}
        cursor = end - timedelta(days=lookback_days)
        now = utcnow()
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            ttl = 30 * 86400 if (now - chunk_end) > timedelta(days=3) else self.settings.recent_history_cache_ttl_seconds
            for g in self.games(sport_code, path, cursor, chunk_end, ttl):
                if g.completed and g.start_time < end:
                    out[g.game_id] = HistoricalGame(g.game_id, g.start_time, g.home_team, g.away_team, g.home_score,
                                                    g.away_score, g.neutral_site, "balldontlie")
            cursor = chunk_end + timedelta(days=1)
        return sorted(out.values(), key=lambda g: g.date)
