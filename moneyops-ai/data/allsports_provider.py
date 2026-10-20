"""AllSportsAPI provider (https://allsportsapi.com, profiled at
https://github.com/api-evangelist/all-sports-api).

Confirmed from public material: base host apiv2.allsportsapi.com/<sport>,
`APIkey` query parameter, operations chosen with `met=` (Leagues, Fixtures,
Livescore, Odds, Probabilities, H2H, Standings ...), JSON envelope
{"success": 1, "result": ...}, fixture fields such as `event_home_team` and
`event_final_result` ("0 - 0"). Documented sports: football, basketball,
cricket, tennis (hockey/baseball/American football are sold as add-ons with
undocumented paths - configure ALLSPORTS_PATH_OVERRIDES to enable them).

NOT yet confirmed against a live response (the build environment could not
reach the host): the exact Odds field names (odd_bookmakers, odd_epoch,
odd_1/odd_x/odd_2, o+N/u+N) and Probabilities field names (event_HW, event_D,
event_AW). Every parser here FAILS CLOSED: an unexpected shape yields no data
and a recorded coverage gap, never a guessed value. Run
`python scripts/probe_allsports.py` once with your key to confirm fields.

Roles in MoneyOps:
* second INDEPENDENT schedule / status source for the fact-checker (Agent 2)
* second FINAL-RESULT source for settlement (cross-checked against ESPN)
* fallback HISTORY source for the team models when ESPN is unavailable
* second ODDS source (more bookmakers + cross-provider price check)
* external PROBABILITIES shown as an outside opinion (never blended in)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from data.provider_base import ProviderBase, ProviderBudgetExceeded, ProviderError, ProviderNotConfigured
from data.schemas import Event, HistoricalGame, OutcomePrice, ScheduledGame
from utils.teams import normalize_team

log = logging.getLogger(__name__)

FINISHED = {"finished", "after et", "after pen.", "after penalties", "after over time", "after ot", "ended", "ft",
            "final", "aet", "ap"}
CANCELLED = ("postpon", "cancel", "abandon", "suspend", "interrupt", "walkover", "retired", "awarded")
SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
TOTAL_KEY_RE = re.compile(r"^([ou])\+(\d+(?:\.\d+)?)$")


def _score_pair(value) -> tuple[int, int] | None:
    if not isinstance(value, str):
        return None
    m = SCORE_RE.match(value)
    return (int(m.group(1)), int(m.group(2))) if m else None


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v not in (None, ""):
            return v
    return None


def parse_fixture(item: dict, sport_code: str, tz: str, retrieved_at: datetime, has_draw: bool = False):
    """Return (ScheduledGame, extra) or None if required fields are missing.

    For draw sports (soccer) betting markets settle on the 90-minute result:
    `event_ft_result` is used when present; a game decided in extra time or
    penalties without a 90-minute score is reported as not settleable."""
    key = _first(item, "event_key")
    home = _first(item, "event_home_team", "event_first_player")
    away = _first(item, "event_away_team", "event_second_player")
    date = _first(item, "event_date", "event_date_start")
    if not (key and home and away and date):
        return None
    time_s = _first(item, "event_time") or "00:00"
    try:
        local = datetime.strptime(f"{date} {time_s[:5]}", "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(tz))
    except ValueError:
        return None
    status = str(_first(item, "event_status") or "").strip()
    low = status.lower()
    completed = low in FINISHED
    cancelled = any(w in low for w in CANCELLED)
    live = str(item.get("event_live", "0")) == "1" or (not completed and not cancelled and bool(low) and
                                                      (low.rstrip("'").isdigit() or "half" in low or "quarter" in low
                                                       or "set" in low or "inning" in low or "period" in low))
    state = "post" if (completed or cancelled) else ("in" if live else "pre")
    scores, settle_note = None, ""
    if completed:
        if has_draw:
            scores = _score_pair(item.get("event_ft_result"))
            if scores is None and low == "finished":
                scores = _score_pair(item.get("event_final_result"))
            if scores is None:
                settle_note = "no 90-minute result for a game decided after extra time/penalties"
        else:
            scores = _score_pair(item.get("event_final_result"))
    game = ScheduledGame(
        game_id=str(key), sport_code=sport_code, start_time=local.astimezone(timezone.utc), home_team=str(home),
        away_team=str(away), state=state, status_detail=status, completed=completed and scores is not None,
        home_score=scores[0] if scores else None, away_score=scores[1] if scores else None,
        neutral_site=False, source="allsportsapi", retrieved_at=retrieved_at)
    return game, {"league": item.get("league_name"), "league_key": item.get("league_key"),
                  "season": item.get("league_season"), "settle_note": settle_note, "cancelled": cancelled}


def parse_odds(result, fixtures: dict[str, ScheduledGame], sport_code: str, sport_key: str, league: str,
               has_draw: bool, retrieved_at: datetime) -> list[Event]:
    """`result` is expected to be {match_key: [ {bookmaker row}, ... ]}.

    Moneyline: odd_1 / odd_2 (+ odd_x for draw sports). For sports without
    draws a quoted odd_x means a 3-way regulation market, which is NOT the
    2-way moneyline, so it is skipped. Totals: keys "o+2.5" / "u+2.5".
    Asian handicap keys are skipped (side/sign convention unverified).
    Rows without an `odd_epoch` timestamp are skipped (cannot be age-checked)."""
    if not isinstance(result, dict):
        return []
    events = []
    for match_key, rows in result.items():
        game = fixtures.get(str(match_key))
        if game is None or not isinstance(rows, list):
            continue
        ev = Event(event_id=f"as:{match_key}", sport_code=sport_code, sport_key=sport_key, league=league,
                   commence_time=game.start_time, home_team=game.home_team, away_team=game.away_team,
                   source="allsportsapi", retrieved_at=retrieved_at)
        for row in rows:
            if not isinstance(row, dict):
                continue
            book = str(row.get("odd_bookmakers") or "").strip()
            try:
                ts = datetime.fromtimestamp(int(row["odd_epoch"]), tz=timezone.utc)
            except (KeyError, TypeError, ValueError, OSError):
                continue
            if not book:
                continue
            bkey = "as:" + normalize_team(book).replace(" ", "_")
            title = f"{book} (AllSportsAPI)"

            def price(k):
                try:
                    v = float(row.get(k))
                except (TypeError, ValueError):
                    return None
                return v if v > 1.0 else None

            def add(market, name, d, point=None):
                ev.prices.append(OutcomePrice(market, bkey, title, name, d, point, ts, provider="allsportsapi"))

            p1, px, p2 = price("odd_1"), price("odd_x"), price("odd_2")
            if p1 and p2 and ((has_draw and px) or (not has_draw and px is None)):
                add("h2h", game.home_team, p1)
                add("h2h", game.away_team, p2)
                if has_draw:
                    add("h2h", "Draw", px)
            totals: dict[float, dict[str, float]] = {}
            for k in row:
                m = TOTAL_KEY_RE.match(str(k))
                if m and price(k):
                    totals.setdefault(float(m.group(2)), {})[m.group(1)] = price(k)
            for point, sides in totals.items():
                if "o" in sides and "u" in sides:
                    add("totals", "Over", sides["o"], point)
                    add("totals", "Under", sides["u"], point)
        if ev.prices:
            events.append(ev)
    return events


def parse_probabilities(result) -> dict[str, dict[str, float]]:
    """{match_key: {"home": p, "draw": p, "away": p}} from percentage strings;
    rows with missing/invalid values or not summing to ~100% are dropped."""
    out = {}
    for item in result if isinstance(result, list) else []:
        try:
            key = str(item["event_key"])
            hw, d, aw = (float(item[k]) for k in ("event_HW", "event_D", "event_AW"))
        except (KeyError, TypeError, ValueError):
            continue
        total = hw + d + aw
        if not 97 <= total <= 103:
            continue
        out[key] = {"home": hw / total, "draw": d / total, "away": aw / total}
    return out


class AllSportsProvider(ProviderBase):
    name = "allsportsapi"
    secret_params = ("APIkey",)

    def __init__(self, settings, cache=None, session=None):
        super().__init__(settings, cache, session)
        self.requests_this_run = 0
        self._league_cache: dict[str, list] = {}

    def is_configured(self) -> bool:
        return bool(self.settings.allsports_api_key)

    def _call(self, path: str, met: str, params: dict, ttl: int):
        if not self.is_configured():
            raise ProviderNotConfigured("ALLSPORTS_API_KEY is not set")
        if self.requests_this_run >= self.settings.allsports_max_requests_per_run:
            raise ProviderBudgetExceeded(
                f"AllSportsAPI request budget ({self.settings.allsports_max_requests_per_run}/run) reached")
        url = f"{self.settings.allsports_base_url.rstrip('/')}/{path}/"
        body, meta = self.get_json(url, {"met": met, "APIkey": self.settings.allsports_api_key, **params}, ttl)
        if not meta.from_cache:
            self.requests_this_run += 1
        if not isinstance(body, dict):
            raise ProviderError(f"allsportsapi {met}: unexpected payload type {type(body).__name__}")
        if str(body.get("success")) != "1":
            raise ProviderError(f"allsportsapi {met}: request not successful ({str(body)[:200]})")
        return body.get("result"), meta.fetched_at

    # -------------------------------------------------------------- leagues
    def resolve_league(self, path: str, league_name: str, country: str | None) -> str | None:
        if path not in self._league_cache:
            result, _ = self._call(path, "Leagues", {}, 7 * 86400)
            self._league_cache[path] = result if isinstance(result, list) else []
        target = normalize_team(league_name)
        rows = [r for r in self._league_cache[path] if normalize_team(str(r.get("league_name", ""))) == target]
        if country:
            exact = [r for r in rows if normalize_team(str(r.get("country_name", ""))) == normalize_team(country)]
            rows = exact or ([] if len(rows) != 1 else rows)
        if len(rows) != 1:
            return None
        return str(rows[0].get("league_key"))

    # ------------------------------------------------------------- fixtures
    def fixtures(self, sport_code: str, path: str, start: datetime, end: datetime, league_key: str | None,
                 has_draw: bool, ttl: int | None = None) -> tuple[list[ScheduledGame], dict]:
        params = {"from": f"{start:%Y-%m-%d}", "to": f"{end:%Y-%m-%d}", "timezone": self.settings.allsports_timezone}
        if league_key:
            params["leagueId"] = league_key
        result, fetched = self._call(path, "Fixtures", params,
                                     self.settings.schedule_cache_ttl_seconds if ttl is None else ttl)
        games, extras = [], {}
        for item in result if isinstance(result, list) else []:
            parsed = parse_fixture(item, sport_code, self.settings.allsports_timezone, fetched, has_draw)
            if parsed:
                games.append(parsed[0])
                extras[parsed[0].game_id] = parsed[1]
        return games, extras

    def history(self, sport_code: str, path: str, end: datetime, lookback_days: int, league_key: str | None,
                has_draw: bool, chunk_days: int = 31) -> list[HistoricalGame]:
        out: dict[str, HistoricalGame] = {}
        cursor = end - timedelta(days=lookback_days)
        now = datetime.now(timezone.utc)
        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
            ttl = 30 * 86400 if (now - chunk_end) > timedelta(days=3) else self.settings.recent_history_cache_ttl_seconds
            games, _ = self.fixtures(sport_code, path, cursor, chunk_end, league_key, has_draw, ttl)
            for g in games:
                if g.completed and g.start_time < end:
                    out[g.game_id] = HistoricalGame(g.game_id, g.start_time, g.home_team, g.away_team, g.home_score,
                                                    g.away_score, False, "allsportsapi")
            cursor = chunk_end + timedelta(days=1)
        return sorted(out.values(), key=lambda g: g.date)

    # ----------------------------------------------------------- odds/probs
    def odds(self, sport_code: str, sport_key: str, league: str, path: str, start: datetime, end: datetime,
             league_key: str | None, has_draw: bool, fixtures: dict[str, ScheduledGame]) -> list[Event]:
        params = {"from": f"{start:%Y-%m-%d}", "to": f"{end:%Y-%m-%d}"}
        if league_key:
            params["leagueId"] = league_key
        result, fetched = self._call(path, "Odds", params, self.settings.odds_cache_ttl_seconds)
        return parse_odds(result, fixtures, sport_code, sport_key, league, has_draw, fetched)

    def probabilities(self, path: str, start: datetime, end: datetime, league_key: str | None) -> dict:
        params = {"from": f"{start:%Y-%m-%d}", "to": f"{end:%Y-%m-%d}"}
        if league_key:
            params["leagueId"] = league_key
        result, _ = self._call(path, "Probabilities", params, self.settings.odds_cache_ttl_seconds)
        return parse_probabilities(result)
