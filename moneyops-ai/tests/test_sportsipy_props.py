"""sportsipy provider and player-prop model tests using FAKE sportsipy
objects (the real library scrapes sports-reference.com, unreachable here)."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from config import Settings
from data.provider_base import ProviderBudgetExceeded, ProviderError, ProviderNotConfigured
from data.sportsipy_provider import SportsipyProvider, lines_to_dicts, season_year, team_abbr
from models.props import bootstrap_interval, distribution_probability, player_values, prop_history_report

T = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)


def fake_world(n_games=12, star_points=None, start=datetime(2026, 10, 22), only_abbr=None):
    """Celtics schedule + box scores. Tatum scores star_points[i] in game i."""
    star_points = star_points or [20 + (i % 7) for i in range(n_games)]
    sched = [SimpleNamespace(datetime=start + timedelta(days=i), boxscore_index=f"20261{i:03d}BOS",
                             location="Home" if i % 2 == 0 else "Away", opponent_name=f"Opp {i}",
                             points_scored=100 + i) for i in range(n_games)]
    sched.append(SimpleNamespace(datetime=datetime(2026, 12, 25), boxscore_index="future", location="Home",
                                 opponent_name="X", points_scored=None))     # unplayed

    def player(name, pts, mins="36:00"):
        return SimpleNamespace(name=name, points=pts, total_rebounds=8, assists=5, three_pointers=3, blocks=1,
                               steals=1, turnovers=2, minutes_played=None if mins is None else 36.0)

    boxes = {}
    for i, g in enumerate(sched[:-1]):
        mine = [player("Jayson Tatum", star_points[i]), player("Bench Guy", 2, None)]
        other = [player("Opponent Star", 30)]
        home, away = (mine, other) if g.location == "Home" else (other, mine)
        boxes[g.boxscore_index] = SimpleNamespace(home_players=home, away_players=away)
    calls = []

    def loader(module, kind, *args):
        calls.append((kind, args))
        if kind == "Schedule":
            if only_abbr and args[0] != only_abbr:
                raise RuntimeError(f"no fake schedule for {args[0]}")
            return sched
        if args[0] not in boxes:
            raise RuntimeError("layout changed")
        return boxes[args[0]]
    return loader, calls


class Clock:
    def __init__(self):
        self.t, self.slept = 0.0, []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


def make(loader, **kw):
    clock = Clock()
    s = Settings(sportsipy_enabled=True, **kw)
    return SportsipyProvider(s, None, loader, clock, clock.sleep), clock


def test_season_year_and_abbreviations():
    assert season_year("nba", datetime(2026, 10, 20)) == 2027
    assert season_year("nba", datetime(2027, 3, 1)) == 2027
    assert season_year("nfl", datetime(2027, 1, 10)) == 2026
    assert team_abbr("nba", "Brooklyn Nets") == "BRK" and team_abbr("nfl", "Houston Texans") == "htx"
    assert team_abbr("nhl", "St. Louis Blues") == "STL" and team_abbr("nba", "Nowhere FC") is None


def test_disabled_provider():
    p = SportsipyProvider(Settings(sportsipy_enabled=False), loader=lambda *a: None)
    with pytest.raises(ProviderNotConfigured):
        p.team_games("nba", "Boston Celtics", T)


def test_throttle_budget_and_logs():
    loader, calls = fake_world()
    p, clock = make(loader, sportsipy_max_requests_per_minute=15, sportsipy_games_per_team=5)
    lines = p.team_player_logs("nba", "Boston Celtics", T)
    assert len(calls) == 6                            # 1 schedule + 5 box scores
    assert all(s == pytest.approx(4.0) for s in clock.slept)  # 60/15 seconds between page loads
    tatum = [l for l in lines if l.player == "Jayson Tatum"]
    assert len(tatum) == 5 and all(l.team == "Boston Celtics" for l in tatum)
    assert {l.home for l in tatum} == {True, False}
    p2, _ = make(fake_world()[0], sportsipy_max_requests_per_run=3)
    with pytest.raises(ProviderBudgetExceeded):
        p2.team_player_logs("nba", "Boston Celtics", T)


def test_games_only_before_cutoff_and_errors_wrapped():
    loader, _ = fake_world()
    p, _ = make(loader)
    games = p.team_games("nba", "Boston Celtics", datetime(2026, 10, 25, tzinfo=timezone.utc))
    assert [g["date"] for g in games] == ["2026-10-22", "2026-10-23", "2026-10-24"]   # no leakage, no unplayed
    with pytest.raises(ProviderError):
        p.boxscore_lines("nba", "missing")


def test_prop_settlement_lookup():
    loader, _ = fake_world(star_points=[31] * 12)
    p, _ = make(loader)
    st, val, src = p.player_stat_in_game("nba", "Boston Celtics", datetime(2026, 10, 23, 23, 0, tzinfo=timezone.utc),
                                         "Jayson Tatum", ("points",))
    assert st == "FINAL" and val == 31
    st2, _, _ = p.player_stat_in_game("nba", "Boston Celtics", datetime(2026, 10, 23, 23, 0, tzinfo=timezone.utc),
                                      "Bench Guy", ("points",))
    assert st2 == "DNP"   # zero/blank minutes -> void, not a loss


def test_prop_model_coherence():
    vals = [22, 25, 19, 30, 27, 24, 21, 26, 28, 23, 25, 24]
    o, po = distribution_probability(vals, "over", 24.5)
    u, pu = distribution_probability(vals, "under", 24.5)
    assert po == pu == 0 and o + u == pytest.approx(1.0)
    o2, push = distribution_probability(vals, "over", 25.0)
    u2, _ = distribution_probability(vals, "under", 25.0)
    assert push > 0 and o2 + u2 + push == pytest.approx(1.0)
    assert distribution_probability(vals[:5], "over", 24.5) is None      # too few games
    lo, hi = bootstrap_interval(vals, "over", 24.5)
    assert lo < o < hi


def test_player_values_and_history_report():
    loader, _ = fake_world(star_points=[10, 30] * 6)
    p, _ = make(loader)
    lines = lines_to_dicts(p.team_player_logs("nba", "Boston Celtics", T))
    rows = player_values(lines, "Jayson Tatum", ("points",), "nba")
    assert len(rows) == 12 and player_values(lines, "Bench Guy", ("points",), "nba") == []   # DNP excluded
    rep = prop_history_report(rows, "over", 20.5, True, "Opp 0")
    assert rep["windows"]["all_logged"]["wins"] == 6 and rep["windows"]["all_logged"]["n"] == 12
    assert rep["splits"]["vs_this_opponent"]["n"] == 1


def test_prop_settlement_through_service(repo):
    from services.settlement import SettlementService
    loader, _ = fake_world(star_points=[31] * 12, only_abbr="BOS")
    s = Settings(sportsipy_enabled=True)
    sp = SportsipyProvider(s, None, loader, lambda: 0.0, lambda x: None)
    run = repo.create_run("t", {})
    start = datetime(2026, 10, 23, 23, 0, tzinfo=timezone.utc)
    rid = repo.save_recommendation(run_id=run, kind="single", candidate_key="p1", sport_code="NBA", league="NBA",
                                   event_id="e", event_name="NYK @ BOS", home_team="Boston Celtics",
                                   away_team="New York Knicks", commence_time=start, market="player_points",
                                   selection="Jayson Tatum Over 29.5", side="over", point=29.5, odds_decimal=1.9,
                                   final_prob=0.5, status="QUALIFIED", model_version="t",
                                   report_json={"player": "Jayson Tatum"})
    svc = SettlementService(s, repo, sportsipy=sp)
    assert svc.grade_recommendations(now=start + timedelta(hours=6))["graded"] == 1
    rec = repo.get_recommendation(rid)
    assert rec.result == "win" and "31" in rec.result_source
