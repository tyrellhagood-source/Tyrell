"""sportsipy provider (https://github.com/roclark/sportsipy): player game logs
for player-prop research and prop settlement, scraped from sports-reference.com.

IMPORTANT
* sportsipy is UNMAINTAINED (last commit March 2022). sports-reference pages
  have changed since, so individual parsers may fail; failures are reported
  as coverage gaps, never patched with guessed numbers.
* sports-reference.com limits automated traffic (roughly 20 requests/minute;
  faster clients get temporarily blocked). This provider throttles every page
  load (default 15/min, hard cap 19/min), caps requests per run, and caches
  each finished box score permanently so a page is fetched at most once.
* It is a scraper of a third-party website: use for personal research and
  follow sports-reference.com's terms of use.

Each Schedule(...) or Boxscore(...) constructor in sportsipy performs one page
load, so this provider calls them only through `_load`, which throttles.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from data.provider_base import ProviderBudgetExceeded, ProviderError, ProviderNotConfigured, ProviderStatus
from utils.teams import match_team, normalize_team
from utils.timeutil import utcnow

log = logging.getLogger(__name__)

# Full team name (as used by The Odds API / ESPN) -> sports-reference abbreviation
TEAM_ABBR = {
    "nba": {
        "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK", "Charlotte Hornets": "CHO",
        "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET", "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO", "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
        "Washington Wizards": "WAS"},
    "nfl": {
        "Arizona Cardinals": "crd", "Atlanta Falcons": "atl", "Baltimore Ravens": "rav", "Buffalo Bills": "buf",
        "Carolina Panthers": "car", "Chicago Bears": "chi", "Cincinnati Bengals": "cin", "Cleveland Browns": "cle",
        "Dallas Cowboys": "dal", "Denver Broncos": "den", "Detroit Lions": "det", "Green Bay Packers": "gnb",
        "Houston Texans": "htx", "Indianapolis Colts": "clt", "Jacksonville Jaguars": "jax",
        "Kansas City Chiefs": "kan", "Las Vegas Raiders": "rai", "Los Angeles Chargers": "sdg",
        "Los Angeles Rams": "ram", "Miami Dolphins": "mia", "Minnesota Vikings": "min",
        "New England Patriots": "nwe", "New Orleans Saints": "nor", "New York Giants": "nyg",
        "New York Jets": "nyj", "Philadelphia Eagles": "phi", "Pittsburgh Steelers": "pit",
        "San Francisco 49ers": "sfo", "Seattle Seahawks": "sea", "Tampa Bay Buccaneers": "tam",
        "Tennessee Titans": "oti", "Washington Commanders": "was"},
    "nhl": {
        "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
        "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
        "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET", "Edmonton Oilers": "EDM",
        "Florida Panthers": "FLA", "Los Angeles Kings": "LAK", "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL",
        "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
        "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
        "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA", "St Louis Blues": "STL",
        "Tampa Bay Lightning": "TBL", "Toronto Maple Leafs": "TOR", "Utah Mammoth": "UTA",
        "Utah Hockey Club": "UTA", "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VEG",
        "Washington Capitals": "WSH", "Winnipeg Jets": "WPG"},
}

# The Odds API prop market -> tuple of sportsipy player attributes summed
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
    "nhl": {"player_points": ("points",), "player_goals": ("goals",), "player_assists": ("assists",),
            "player_shots_on_goal": ("shots_on_goal",)},
}
ALL_ATTRS = {k: sorted({a for v in m.values() for a in v} | ({"minutes_played"} if k == "nba" else set()))
             for k, m in PROP_STATS.items()}


@dataclass
class PlayerGameLine:
    player: str
    team: str
    opponent: str
    date: str                 # ISO date (sports-reference local game date)
    home: bool
    game_id: str
    minutes: float | None
    stats: dict = field(default_factory=dict)
    source: str = "sportsipy/sports-reference"


def season_year(module: str, on: datetime) -> int:
    """sports-reference season label: NBA/NHL use the year the season ends
    (Oct 2026 -> 2027); NFL uses the year it starts (Jan 2027 -> 2026)."""
    if module == "nfl":
        return on.year if on.month >= 8 else on.year - 1
    return on.year + 1 if on.month >= 8 else on.year


def team_abbr(module: str, team: str) -> str | None:
    names = TEAM_ABBR.get(module, {})
    m, _ = match_team(team, list(names))
    return names.get(m) if m else None


class SportsipyProvider:
    name = "sportsipy"

    def __init__(self, settings, cache=None, loader=None, clock=time.monotonic, sleeper=time.sleep):
        self.settings, self.cache = settings, cache
        self._loader = loader          # injectable for tests: loader(module, kind, *args)
        self.status = ProviderStatus(self.name, configured=self.is_configured())
        self.requests_this_run = 0
        self._clock, self._sleep = clock, sleeper
        self._last = None
        self._schedules: dict = {}

    def is_configured(self) -> bool:
        if not self.settings.sportsipy_enabled:
            return False
        if self._loader is not None:
            return True
        try:
            import sportsipy  # noqa: F401
        except ImportError:
            return False
        return True

    # ------------------------------------------------------------ plumbing
    def _load(self, module: str, kind: str, *args):
        if not self.is_configured():
            raise ProviderNotConfigured("sportsipy disabled (SPORTSIPY_ENABLED=false) or not installed")
        if self.requests_this_run >= self.settings.sportsipy_max_requests_per_run:
            raise ProviderBudgetExceeded(f"sportsipy page budget ({self.settings.sportsipy_max_requests_per_run}/run) reached")
        gap = 60.0 / max(1, min(19, self.settings.sportsipy_max_requests_per_minute))
        if self._last is not None:
            wait = gap - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        self.requests_this_run += 1
        self.status.requests_made += 1
        try:
            if self._loader is not None:
                obj = self._loader(module, kind, *args)
            else:
                import importlib
                mod = importlib.import_module(f"sportsipy.{module}.{kind.lower()}")
                obj = getattr(mod, kind)(*args)
        except (ProviderError, ProviderBudgetExceeded):
            raise
        except Exception as exc:  # sportsipy raises bare exceptions on layout changes / HTTP errors
            msg = f"sportsipy {module}.{kind}{args}: {type(exc).__name__}: {exc}"
            self.status.last_error, self.status.last_error_at = msg, utcnow()
            raise ProviderError(msg) from exc
        self.status.last_success = utcnow()
        return obj

    def _cache_get(self, key):
        if self.cache is None:
            return None
        hit = self.cache.get(self.cache.make_key(self.name, key, {}))
        if hit is not None:
            self.status.cache_hits += 1
            return hit[0]
        return None

    def _cache_set(self, key, value, ttl):
        if self.cache is not None:
            self.cache.set(self.cache.make_key(self.name, key, {}), self.name, key, value, {}, ttl)

    # ------------------------------------------------------------- queries
    def team_games(self, module: str, team: str, before: datetime) -> list[dict]:
        """Completed games of `team` strictly before `before` (date-level)."""
        abbr = team_abbr(module, team)
        if abbr is None:
            raise ProviderError(f"no sports-reference abbreviation for {team}")
        year = season_year(module, before)
        key = f"schedule:{module}:{abbr}:{year}:{before:%Y-%m-%d}"
        games = self._cache_get(key)
        if games is None:
            sched = self._load(module, "Schedule", abbr, year)
            games = []
            for g in sched:
                scored = getattr(g, "points_scored", None) if module != "nhl" else getattr(g, "goals_scored", None)
                dt = getattr(g, "datetime", None)
                if scored is None or dt is None or not getattr(g, "boxscore_index", None):
                    continue
                games.append({"date": dt.strftime("%Y-%m-%d"), "boxscore": g.boxscore_index,
                              "home": str(getattr(g, "location", "")).lower().startswith("home"),
                              "opponent": getattr(g, "opponent_name", "") or getattr(g, "opponent_abbr", "")})
            self._cache_set(key, games, self.settings.recent_history_cache_ttl_seconds)
        cutoff = before.astimezone(timezone.utc).date().isoformat()
        return sorted([g for g in games if g["date"] < cutoff], key=lambda g: g["date"])

    def boxscore_lines(self, module: str, boxscore_id: str) -> dict:
        """{"home": [player dicts], "away": [...]} for one finished game (cached forever)."""
        key = f"boxscore:{module}:{boxscore_id}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        box = self._load(module, "Boxscore", boxscore_id)
        out = {}
        for side in ("home", "away"):
            rows = []
            for p in getattr(box, f"{side}_players", None) or []:
                stats = {}
                for attr in ALL_ATTRS[module]:
                    try:
                        stats[attr] = getattr(p, attr)
                    except Exception:  # missing column on this page
                        stats[attr] = None
                rows.append({"player": getattr(p, "name", None), "stats": stats})
            out[side] = rows
        if not out.get("home") and not out.get("away"):
            raise ProviderError(f"sportsipy returned no player rows for {boxscore_id} (page layout changed?)")
        self._cache_set(key, out, 365 * 86400)
        return out

    def team_player_logs(self, module: str, team: str, before: datetime, n_games: int | None = None
                         ) -> list[PlayerGameLine]:
        games = self.team_games(module, team, before)[-(n_games or self.settings.sportsipy_games_per_team):]
        lines = []
        for g in games:
            box = self.boxscore_lines(module, g["boxscore"])
            for row in box["home" if g["home"] else "away"]:
                if not row.get("player"):
                    continue
                st = row["stats"]
                lines.append(PlayerGameLine(row["player"], team, g["opponent"], g["date"], g["home"], g["boxscore"],
                                            st.get("minutes_played"), {k: v for k, v in st.items()
                                                                       if k != "minutes_played"}))
        return lines

    def player_stat_in_game(self, module: str, team: str, game_date: datetime, player: str, attrs: tuple
                            ) -> tuple[str, float | None, str]:
        """For settlement: ("FINAL", value, source) / ("DNP", None, src) / ("UNAVAILABLE", None, why)."""
        games = self.team_games(module, team, game_date + timedelta(days=2))
        day = {(game_date - timedelta(hours=h)).strftime("%Y-%m-%d") for h in (0, 12)}
        match = [g for g in games if g["date"] in day]
        if not match:
            return "UNAVAILABLE", None, "game not found on sports-reference schedule"
        g = match[-1]
        rows = self.boxscore_lines(module, g["boxscore"])["home" if g["home"] else "away"]
        target = normalize_team(player)
        for r in rows:
            if r.get("player") and normalize_team(r["player"]) == target:
                vals = [r["stats"].get(a) for a in attrs]
                if module == "nba" and not r["stats"].get("minutes_played"):
                    return "DNP", None, f"sports-reference box {g['boxscore']}"
                if any(v is None for v in vals):
                    return "UNAVAILABLE", None, "stat missing in box score"
                return "FINAL", float(sum(vals)), f"sports-reference box {g['boxscore']}"
        return "DNP", None, f"player not in box score {g['boxscore']}"


def lines_to_dicts(lines: list[PlayerGameLine]) -> list[dict]:
    return [asdict(l) for l in lines]
