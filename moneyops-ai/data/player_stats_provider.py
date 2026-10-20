"""Player game logs for prop research and prop settlement.

* NBA: nba_api (https://github.com/swar/nba_api, v1.11.x, maintained 2026) ->
  stats.nba.com `LeagueGameLog` in player mode: ONE request returns every
  player-game line of a season (PTS, REB, AST, FG3M, MIN, MATCHUP ...).
* NFL: nflreadpy (https://github.com/nflverse/nflreadpy, maintained 2026) ->
  nflverse weekly player stats (`stats_player_week_<season>`) joined to the
  nflverse schedule for game dates, home/away and final scores.

Both sources are free and keyless. Column names are read defensively: a
missing column means "no data" (coverage gap), never a guessed value.
Seasons loaded: the current one plus the previous one (early-season sample),
with each line dated so nothing on/after kick-off is ever used.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from data.provider_base import ProviderError, ProviderNotConfigured, ProviderStatus
from utils.teams import match_team, normalize_team
from utils.timeutil import utcnow

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# The Odds API prop market -> tuple of normalised stat names (summed)
PROP_STATS = {
    "nba": {"player_points": ("points",), "player_rebounds": ("total_rebounds",), "player_assists": ("assists",),
            "player_threes": ("three_pointers",), "player_blocks": ("blocks",), "player_steals": ("steals",),
            "player_turnovers": ("turnovers",),
            "player_points_rebounds_assists": ("points", "total_rebounds", "assists"),
            "player_points_rebounds": ("points", "total_rebounds"), "player_points_assists": ("points", "assists"),
            "player_rebounds_assists": ("total_rebounds", "assists")},
    "nfl": {"player_pass_yds": ("passing_yards",), "player_pass_tds": ("passing_touchdowns",),
            "player_pass_completions": ("completed_passes",), "player_pass_attempts": ("attempted_passes",),
            "player_pass_interceptions": ("interceptions_thrown",), "player_rush_yds": ("rush_yards",),
            "player_rush_attempts": ("rush_attempts",), "player_reception_yds": ("receiving_yards",),
            "player_receptions": ("receptions",)},
}
NBA_COLUMNS = {"points": "PTS", "total_rebounds": "REB", "assists": "AST", "three_pointers": "FG3M",
               "blocks": "BLK", "steals": "STL", "turnovers": "TOV"}
# nflverse column candidates (new weekly release first, legacy names second)
NFL_COLUMNS = {"passing_yards": ("passing_yards",), "passing_touchdowns": ("passing_tds",),
               "completed_passes": ("completions",), "attempted_passes": ("attempts",),
               "interceptions_thrown": ("passing_interceptions", "interceptions"),
               "rush_yards": ("rushing_yards",), "rush_attempts": ("carries",),
               "receiving_yards": ("receiving_yards",), "receptions": ("receptions",)}
NFL_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"}
NFL_NAME = {v: k for k, v in NFL_ABBR.items()}


@dataclass
class PlayerGameLine:
    player: str
    team: str
    opponent: str
    date: str                 # ISO local game date (US Eastern)
    home: bool
    game_id: str
    minutes: float | None
    stats: dict = field(default_factory=dict)
    source: str = ""


def lines_to_dicts(lines: list[PlayerGameLine]) -> list[dict]:
    return [asdict(l) for l in lines]


def game_date_et(dt: datetime) -> str:
    return dt.astimezone(ET).date().isoformat()


def nba_season(dt: datetime) -> str:
    y = dt.year if dt.month >= 8 else dt.year - 1
    return f"{y}-{str(y + 1)[2:]}"


def nfl_season(dt: datetime) -> int:
    return dt.year if dt.month >= 3 else dt.year - 1


def _num(v):
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _minutes(v):
    if isinstance(v, str) and ":" in v:
        m, s = v.split(":", 1)
        return _num(m) + (_num(s) or 0) / 60 if _num(m) is not None else None
    return _num(v)


class _CachedSource:
    """Shared caching/status plumbing for keyless dataset sources."""
    name = "base"
    module = ""

    def __init__(self, settings, cache=None):
        self.settings, self.cache = settings, cache
        self.status = ProviderStatus(self.name, configured=self.is_configured())
        self.requests_this_run = 0

    def is_configured(self) -> bool:
        return True

    def _cached(self, key: str, ttl: int, loader):
        if self.cache is not None:
            hit = self.cache.get(self.cache.make_key(self.name, key, {}))
            if hit is not None:
                self.status.cache_hits += 1
                return hit[0]
        if not self.is_configured():
            raise ProviderNotConfigured(f"{self.name} disabled or not installed")
        self.requests_this_run += 1
        self.status.requests_made += 1
        try:
            rows = loader()
        except (ProviderError, ProviderNotConfigured):
            raise
        except Exception as exc:  # network errors, HTTP blocks, schema changes
            msg = f"{self.name} {key}: {type(exc).__name__}: {exc}"
            self.status.last_error, self.status.last_error_at = msg, utcnow()
            raise ProviderError(msg) from exc
        if not isinstance(rows, list):
            raise ProviderError(f"{self.name} {key}: unexpected payload {type(rows).__name__}")
        self.status.last_success = utcnow()
        if self.cache is not None:
            self.cache.set(self.cache.make_key(self.name, key, {}), self.name, key, rows, {}, ttl)
        return rows

    def lines(self, before: datetime) -> list[PlayerGameLine]:
        raise NotImplementedError

    def team_player_logs(self, team: str, before: datetime) -> list[PlayerGameLine]:
        """All lines (current + previous season, strictly before `before`) of
        players whose most recent appearance was for `team`, newest
        `games_per_player` per player."""
        all_lines = [l for l in self.lines(before) if l.date < game_date_et(before)]
        names = sorted({l.team for l in all_lines})
        canonical, _ = match_team(team, names)
        if canonical is None:
            raise ProviderError(f"{self.name}: team '{team}' not found in player logs")
        latest: dict[str, PlayerGameLine] = {}
        for l in sorted(all_lines, key=lambda l: l.date):
            latest[normalize_team(l.player)] = l
        players = {p for p, l in latest.items() if l.team == canonical}
        out = [l for l in all_lines if normalize_team(l.player) in players]
        n = self.settings.player_stats_games_per_player
        per: dict[str, list] = {}
        for l in sorted(out, key=lambda l: l.date):
            per.setdefault(normalize_team(l.player), []).append(l)
        return [l for v in per.values() for l in v[-n:]]


class NbaApiSource(_CachedSource):
    name = "nba_api"
    module = "nba"

    def __init__(self, settings, cache=None, fetch=None):
        self._fetch = fetch      # injectable: fetch(season, season_type) -> list[dict]
        super().__init__(settings, cache)

    def is_configured(self) -> bool:
        if not self.settings.nba_api_enabled:
            return False
        if self._fetch is not None:
            return True
        try:
            import nba_api  # noqa: F401
        except ImportError:
            return False
        return True

    def _default_fetch(self, season: str, season_type: str) -> list[dict]:
        from nba_api.stats.endpoints import leaguegamelog
        ep = leaguegamelog.LeagueGameLog(player_or_team_abbreviation="P", season=season,
                                         season_type_all_star=season_type,
                                         timeout=self.settings.http_timeout_seconds * 2)
        return ep.get_normalized_dict().get("LeagueGameLog", [])

    def season_rows(self, season: str, season_type: str = "Regular Season", ttl: int | None = None) -> list[dict]:
        current = season == nba_season(utcnow())
        ttl = ttl if ttl is not None else (self.settings.recent_history_cache_ttl_seconds if current else 30 * 86400)
        fetch = self._fetch or self._default_fetch
        return self._cached(f"leaguegamelog:P:{season}:{season_type}", ttl, lambda: fetch(season, season_type))

    @staticmethod
    def _team_name(abbr: str) -> str:
        try:
            from nba_api.stats.static import teams
            t = teams.find_team_by_abbreviation(abbr)
            return t["full_name"] if t else abbr
        except ImportError:
            return abbr

    def parse_rows(self, rows: list[dict]) -> list[PlayerGameLine]:
        out = []
        for r in rows:
            name, matchup, date = r.get("PLAYER_NAME"), r.get("MATCHUP"), r.get("GAME_DATE")
            abbr = r.get("TEAM_ABBREVIATION")
            if not (name and matchup and date and abbr):
                continue
            home = " vs. " in matchup
            sep = " vs. " if home else " @ "
            if sep not in matchup:
                continue
            opp_abbr = matchup.split(sep, 1)[1].strip()
            stats = {k: _num(r.get(col)) for k, col in NBA_COLUMNS.items()}
            out.append(PlayerGameLine(str(name), r.get("TEAM_NAME") or self._team_name(abbr), self._team_name(opp_abbr),
                                      str(date)[:10], home, str(r.get("GAME_ID", "")), _minutes(r.get("MIN")), stats,
                                      "nba_api/stats.nba.com"))
        return out

    def lines(self, before: datetime) -> list[PlayerGameLine]:
        cur = nba_season(before)
        prev_start = int(cur[:4]) - 1
        prev = f"{prev_start}-{str(prev_start + 1)[2:]}"
        rows = []
        for season in (prev, cur):
            try:
                rows += self.season_rows(season)
            except ProviderError:
                if season == cur:
                    raise
        return self.parse_rows(rows)

    def player_stat_in_game(self, team: str, commence: datetime, player: str, attrs: tuple):
        day = game_date_et(commence)
        lines = self.parse_rows(self.season_rows(nba_season(commence), ttl=600))
        names = sorted({l.team for l in lines})
        canonical, _ = match_team(team, names)
        team_lines = [l for l in lines if l.team == canonical and l.date == day]
        if not team_lines:
            return "UNAVAILABLE", None, f"no {day} box score for {team} on stats.nba.com yet"
        target = normalize_team(player)
        for l in team_lines:
            if normalize_team(l.player) == target:
                vals = [l.stats.get(a) for a in attrs]
                if not l.minutes:
                    return "DNP", None, f"stats.nba.com game {l.game_id}"
                if any(v is None for v in vals):
                    return "UNAVAILABLE", None, "stat missing"
                return "FINAL", float(sum(vals)), f"stats.nba.com game {l.game_id}"
        return "DNP", None, f"{player} not in {team} box score {team_lines[0].game_id}"


class NflReadSource(_CachedSource):
    name = "nflreadpy"
    module = "nfl"

    def __init__(self, settings, cache=None, fetch_stats=None, fetch_schedules=None):
        self._fetch_stats, self._fetch_schedules = fetch_stats, fetch_schedules
        super().__init__(settings, cache)

    def is_configured(self) -> bool:
        if not self.settings.nflreadpy_enabled:
            return False
        if self._fetch_stats is not None:
            return True
        try:
            import nflreadpy  # noqa: F401
        except ImportError:
            return False
        return True

    @staticmethod
    def _records(df) -> list[dict]:
        if hasattr(df, "to_dicts"):
            return df.to_dicts()          # polars
        return df.to_dict("records")      # pandas

    def _stats(self, season: int, ttl: int | None = None) -> list[dict]:
        def load():
            if self._fetch_stats:
                return self._fetch_stats(season)
            import nflreadpy
            return self._records(nflreadpy.load_player_stats([season], summary_level="week"))
        current = season == nfl_season(utcnow())
        ttl = ttl if ttl is not None else (self.settings.recent_history_cache_ttl_seconds if current else 30 * 86400)
        return self._cached(f"player_stats_week:{season}", ttl, load)

    def _schedule(self, season: int, ttl: int | None = None) -> list[dict]:
        def load():
            if self._fetch_schedules:
                return self._fetch_schedules(season)
            import nflreadpy
            return self._records(nflreadpy.load_schedules([season]))
        return self._cached(f"schedules:{season}", ttl if ttl is not None else 3600, load)

    def parse(self, stats: list[dict], schedule: list[dict]) -> list[PlayerGameLine]:
        games = {}
        for g in schedule:
            key_base = (g.get("season"), g.get("week"))
            for side in ("home", "away"):
                games[key_base + (g.get(f"{side}_team"),)] = (g, side == "home")
        out = []
        for r in stats:
            name = r.get("player_display_name") or r.get("player_name")
            team = r.get("team") or r.get("recent_team")
            found = games.get((r.get("season"), r.get("week"), team))
            if not (name and team and found):
                continue
            g, home = found
            opp = g.get("away_team") if home else g.get("home_team")
            stats_row = {}
            for k, cands in NFL_COLUMNS.items():
                stats_row[k] = next((_num(r[c]) for c in cands if c in r), None)
            out.append(PlayerGameLine(str(name), NFL_NAME.get(team, team), NFL_NAME.get(opp, opp),
                                      str(g.get("gameday", ""))[:10], home, str(g.get("game_id", "")), None,
                                      stats_row, "nflreadpy/nflverse"))
        return out

    def lines(self, before: datetime) -> list[PlayerGameLine]:
        cur = nfl_season(before)
        out = []
        for season in (cur - 1, cur):
            try:
                out += self.parse(self._stats(season), self._schedule(season))
            except ProviderError:
                if season == cur:
                    raise
        return out

    def player_stat_in_game(self, team: str, commence: datetime, player: str, attrs: tuple):
        season = nfl_season(commence)
        day = game_date_et(commence)
        abbr = NFL_ABBR.get(match_team(team, list(NFL_ABBR))[0] or "")
        sched = self._schedule(season, ttl=600)
        game = next((g for g in sched if str(g.get("gameday", ""))[:10] == day
                     and abbr in (g.get("home_team"), g.get("away_team"))), None)
        if game is None:
            return "UNAVAILABLE", None, f"{team} game on {day} not in nflverse schedule"
        if game.get("home_score") is None:
            return "UNAVAILABLE", None, "game not final in nflverse schedule"
        rows = [r for r in self._stats(season, ttl=600)
                if r.get("week") == game.get("week") and (r.get("team") or r.get("recent_team")) == abbr]
        if not rows:
            return "UNAVAILABLE", None, "nflverse weekly stats not yet published for this game"
        target = normalize_team(player)
        for line in self.parse(rows, [game]):
            if normalize_team(line.player) == target:
                vals = [line.stats.get(a) for a in attrs]
                if any(v is None for v in vals):
                    return "UNAVAILABLE", None, "stat column missing"
                return "FINAL", float(sum(vals)), f"nflverse game {game.get('game_id')}"
        return "DNP", None, f"{player} has no stat line in nflverse game {game.get('game_id')}"


class PlayerStatsService:
    """Routes a sport module ("nba" / "nfl") to its source."""

    def __init__(self, settings, cache=None, sources: dict | None = None):
        self.sources = sources if sources is not None else {
            "nba": NbaApiSource(settings, cache), "nfl": NflReadSource(settings, cache)}

    def is_configured(self, module: str | None = None) -> bool:
        if module is None:
            return any(s.is_configured() for s in self.sources.values())
        src = self.sources.get(module)
        return bool(src and src.is_configured())

    def _src(self, module):
        src = self.sources.get(module)
        if src is None or not src.is_configured():
            raise ProviderNotConfigured(f"no enabled player-stats source for {module}")
        return src

    def team_player_logs(self, module: str, team: str, before: datetime) -> list[PlayerGameLine]:
        return self._src(module).team_player_logs(team, before)

    def player_stat_in_game(self, module: str, team: str, commence: datetime, player: str, attrs: tuple):
        return self._src(module).player_stat_in_game(team, commence, player, attrs)

    @property
    def statuses(self):
        return [s.status for s in self.sources.values()]

    @property
    def requests_this_run(self):
        return sum(s.requests_this_run for s in self.sources.values())
