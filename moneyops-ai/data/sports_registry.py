"""Supported sports, their provider identifiers and model families.

`odds_key` is The Odds API sport key. `espn_path` is the ESPN site-API path
used for schedules, results, history and injuries. A sport without an
`espn_path` or `model` has a documented coverage gap (no independent
verification / no statistical model) and can never be fully QUALIFIED.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SportConfig:
    code: str
    name: str
    league: str
    odds_key: str | None
    espn_path: str | None
    model: str | None            # "gaussian" | "poisson" | None
    has_draw: bool = False       # 3-way moneyline (soccer)
    ties_resolved: bool = True   # OT / extra innings decide 2-way moneyline
    tie_adds_one: bool = False   # shootout / extra-inning winner gets +1 in final score
    espn_groups: str | None = None
    max_score: int = 15
    injuries: bool = False
    allsports_path: str | None = None            # AllSportsAPI sport path (football, basketball, tennis, cricket)
    allsports_league: tuple[str, str | None] | None = None   # (league_name, country) resolved via met=Leagues
    player_logs: str | None = None               # player game-log source key: "nba" (nba_api), "nfl" (nflreadpy)


SPORTS: dict[str, SportConfig] = {s.code: s for s in [
    SportConfig("NFL", "American Football", "NFL", "americanfootball_nfl", "football/nfl", "gaussian", injuries=True, player_logs="nfl"),
    SportConfig("NCAAF", "American Football", "NCAA Football (FBS)", "americanfootball_ncaaf",
                "football/college-football", "gaussian", espn_groups="80"),
    SportConfig("NBA", "Basketball", "NBA", "basketball_nba", "basketball/nba", "gaussian", injuries=True, allsports_path="basketball", allsports_league=("NBA", "USA"), player_logs="nba"),
    SportConfig("NCAAB", "Basketball", "NCAA Men's Basketball", "basketball_ncaab",
                "basketball/mens-college-basketball", "gaussian", espn_groups="50"),
    SportConfig("NCAAW", "Basketball", "NCAA Women's Basketball", "basketball_wncaab",
                "basketball/womens-college-basketball", "gaussian", espn_groups="50"),
    SportConfig("WNBA", "Basketball", "WNBA", "basketball_wnba", "basketball/wnba", "gaussian", injuries=True, allsports_path="basketball", allsports_league=("WNBA", "USA")),
    SportConfig("MLB", "Baseball", "MLB", "baseball_mlb", "baseball/mlb", "poisson",
                tie_adds_one=True, max_score=25, injuries=True, allsports_league=("MLB", "USA")),
    SportConfig("NHL", "Ice Hockey", "NHL", "icehockey_nhl", "hockey/nhl", "poisson",
                tie_adds_one=True, max_score=12, injuries=True, allsports_league=("NHL", "USA")),
    SportConfig("EPL", "Soccer", "English Premier League", "soccer_epl", "soccer/eng.1", "poisson",
                has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("Premier League", "England")),
    SportConfig("MLS", "Soccer", "MLS", "soccer_usa_mls", "soccer/usa.1", "poisson",
                has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("MLS", "USA")),
    SportConfig("UCL", "Soccer", "UEFA Champions League", "soccer_uefa_champs_league", "soccer/uefa.champions",
                "poisson", has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("UEFA Champions League", None)),
    SportConfig("LALIGA", "Soccer", "La Liga", "soccer_spain_la_liga", "soccer/esp.1", "poisson",
                has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("La Liga", "Spain")),
    SportConfig("BUNDESLIGA", "Soccer", "Bundesliga", "soccer_germany_bundesliga", "soccer/ger.1", "poisson",
                has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("Bundesliga", "Germany")),
    SportConfig("SERIEA", "Soccer", "Serie A", "soccer_italy_serie_a", "soccer/ita.1", "poisson",
                has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("Serie A", "Italy")),
    SportConfig("LIGUE1", "Soccer", "Ligue 1", "soccer_france_ligue_one", "soccer/fra.1", "poisson",
                has_draw=True, ties_resolved=False, max_score=10, allsports_path="football", allsports_league=("Ligue 1", "France")),
    # Coverage gaps: odds are available but no independent schedule source or
    # statistical model is implemented. Candidates are shown, never QUALIFIED.
    SportConfig("MMA", "MMA", "UFC / MMA", "mma_mixed_martial_arts", None, None),
]}

ODDS_KEY_TO_CODE = {s.odds_key: s.code for s in SPORTS.values() if s.odds_key}


def sport_for_odds_key(odds_key: str, group: str = "", title: str = "") -> SportConfig:
    """Known sports map to their config; others (tennis tournaments etc.)
    get a coverage-gap config built on the fly."""
    code = ODDS_KEY_TO_CODE.get(odds_key)
    if code:
        return SPORTS[code]
    prefix = "TENNIS" if odds_key.startswith("tennis") else "OTHER"
    path = {"tennis": "tennis", "cricket": "cricket"}.get(odds_key.split("_")[0])
    return SportConfig(f"{prefix}:{odds_key}", group or prefix.title(), title or odds_key, odds_key, None, None,
                       allsports_path=path)


def allsports_path(cfg: SportConfig, overrides: tuple[str, ...] = ()) -> str | None:
    """Sport path, allowing ALLSPORTS_PATH_OVERRIDES like ("NHL:hockey",) for add-on sports."""
    for item in overrides:
        code, _, path = item.partition(":")
        if code.strip().upper() == cfg.code and path.strip():
            return path.strip()
    return cfg.allsports_path
