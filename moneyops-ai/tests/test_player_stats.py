"""nba_api / nflreadpy player-log sources and the prop model, with FAKE
fetchers shaped like the real responses (LeagueGameLog player mode rows;
nflverse stats_player_week + schedules). No network."""
from datetime import datetime, timedelta, timezone

import pytest

from config import Settings
from data.player_stats_provider import (NbaApiSource, NflReadSource, PlayerStatsService, game_date_et,
                                        lines_to_dicts, nba_season, nfl_season)
from data.provider_base import ProviderError, ProviderNotConfigured
from models.props import bootstrap_interval, distribution_probability, player_values, prop_history_report

T = datetime(2026, 11, 20, 23, 0, tzinfo=timezone.utc)


def nba_rows(points, start=datetime(2026, 10, 22), player="Jayson Tatum", team="BOS", team_name="Boston Celtics"):
    rows = []
    for i, pts in enumerate(points):
        home = i % 2 == 0
        rows.append({"SEASON_ID": "22026", "PLAYER_ID": 1628369, "PLAYER_NAME": player, "TEAM_ABBREVIATION": team,
                     "TEAM_NAME": team_name, "GAME_ID": f"00226{i:05d}",
                     "GAME_DATE": (start + timedelta(days=2 * i)).strftime("%Y-%m-%d"),
                     "MATCHUP": f"{team} vs. NYK" if home else f"{team} @ NYK", "WL": "W", "MIN": 36,
                     "PTS": pts, "REB": 8, "AST": 5, "FG3M": 3, "BLK": 1, "STL": 1, "TOV": 2})
    return rows


def nba_source(rows_by_season, **kw):
    calls = []

    def fetch(season, season_type):
        calls.append(season)
        if season not in rows_by_season:
            raise RuntimeError("HTTP 403")
        return rows_by_season[season]
    return NbaApiSource(Settings(**kw), None, fetch), calls


def test_season_helpers():
    assert nba_season(datetime(2026, 10, 20)) == "2026-27" and nba_season(datetime(2027, 2, 1)) == "2026-27"
    assert nfl_season(datetime(2027, 1, 10)) == 2026 and nfl_season(datetime(2026, 9, 10)) == 2026
    assert game_date_et(datetime(2026, 10, 22, 2, 30, tzinfo=timezone.utc)) == "2026-10-21"   # 10:30pm ET


def test_nba_logs_no_leakage_and_previous_season_fallback():
    rows = nba_rows([20 + i for i in range(12)])
    src, calls = nba_source({"2026-27": rows})        # previous season unavailable -> tolerated
    lines = src.team_player_logs("Boston Celtics", datetime(2026, 11, 5, 23, 0, tzinfo=timezone.utc))
    assert calls == ["2025-26", "2026-27"]
    assert lines and all(l.date < "2026-11-05" for l in lines)
    assert {l.home for l in lines} == {True, False} and lines[0].opponent == "New York Knicks"
    assert lines[0].stats["points"] == 20 and lines[0].minutes == 36
    src2, _ = nba_source({})
    with pytest.raises(ProviderError):                 # current season unavailable -> reported
        src2.team_player_logs("Boston Celtics", T)


def test_traded_player_follows_latest_team():
    old = nba_rows([10] * 5, start=datetime(2025, 11, 1), player="Moved Guy", team="NYK", team_name="New York Knicks")
    new = nba_rows([15] * 3, start=datetime(2026, 10, 22), player="Moved Guy")
    src, _ = nba_source({"2025-26": old, "2026-27": new})
    bos = src.team_player_logs("Boston Celtics", T)
    assert len([l for l in bos if l.player == "Moved Guy"]) == 8         # history travels with the player
    nyk = src.team_player_logs("New York Knicks", T)
    assert not [l for l in nyk if l.player == "Moved Guy"]


def test_nba_settlement_lookup():
    rows = nba_rows([31] * 3) + [dict(nba_rows([0])[0], PLAYER_NAME="Bench Guy", MIN=0)]
    src, _ = nba_source({"2026-27": rows})
    day = datetime(2026, 10, 22, 23, 30, tzinfo=timezone.utc)
    assert src.player_stat_in_game("Boston Celtics", day, "Jayson Tatum", ("points",))[:2] == ("FINAL", 31.0)
    assert src.player_stat_in_game("Boston Celtics", day, "Bench Guy", ("points",))[0] == "DNP"
    assert src.player_stat_in_game("Boston Celtics", day, "Not Listed", ("points",))[0] == "DNP"
    later = datetime(2026, 12, 1, 23, 30, tzinfo=timezone.utc)
    assert src.player_stat_in_game("Boston Celtics", later, "Jayson Tatum", ("points",))[0] == "UNAVAILABLE"


def nfl_fakes(final=True):
    days = ["2026-09-10", "2026-09-17", "2026-09-24", "2026-10-01", "2026-10-08", "2026-10-15"]
    sched = [{"game_id": f"2026_{w:02d}_KC_BUF", "season": 2026, "week": w, "gameday": days[w - 1],
              "home_team": "KC" if w % 2 else "BUF", "away_team": "BUF" if w % 2 else "KC",
              "home_score": 24 if final or w < 6 else None, "away_score": 20 if final or w < 6 else None}
             for w in range(1, 7)]
    stats = [{"season": 2026, "week": w, "season_type": "REG", "player_display_name": "Patrick Mahomes",
              "team": "KC", "opponent_team": "BUF", "passing_yards": 250 + 10 * w, "passing_tds": 2,
              "completions": 22, "attempts": 34, "passing_interceptions": 1, "carries": 4, "rushing_yards": 20}
             for w in range(1, 6)]
    return stats, sched


def nfl_source(final=True):
    stats, sched = nfl_fakes(final)
    return NflReadSource(Settings(), None, lambda season: stats if season == 2026 else [],
                         lambda season: sched if season == 2026 else [])


def test_nfl_join_and_columns():
    src = nfl_source()
    lines = src.team_player_logs("Kansas City Chiefs", datetime(2026, 10, 25, 17, 0, tzinfo=timezone.utc))
    assert len(lines) == 5 and lines[0].team == "Kansas City Chiefs" and lines[0].opponent == "Buffalo Bills"
    assert lines[0].stats["passing_yards"] == 260 and lines[0].stats["interceptions_thrown"] == 1
    assert lines[0].home is True and lines[1].home is False
    assert lines[0].date == "2026-09-10"


def test_nfl_settlement_states():
    src = nfl_source(final=False)
    off_day = datetime(2026, 10, 13, 17, 0, tzinfo=timezone.utc)
    assert src.player_stat_in_game("Kansas City Chiefs", off_day, "Patrick Mahomes", ("passing_yards",))[0] == "UNAVAILABLE"
    day5 = datetime(2026, 10, 8, 17, 0, tzinfo=timezone.utc)
    st, val, _ = src.player_stat_in_game("Kansas City Chiefs", day5, "Patrick Mahomes", ("passing_yards",))
    assert (st, val) == ("FINAL", 300.0)
    assert src.player_stat_in_game("Kansas City Chiefs", day5, "Backup QB", ("passing_yards",))[0] == "DNP"
    day6 = datetime(2026, 10, 15, 17, 0, tzinfo=timezone.utc)     # not final yet
    assert src.player_stat_in_game("Kansas City Chiefs", day6, "Patrick Mahomes", ("passing_yards",))[0] == "UNAVAILABLE"


def test_service_routing_and_disabled():
    svc = PlayerStatsService(Settings(), sources={"nba": nba_source({})[0]})
    with pytest.raises(ProviderNotConfigured):
        svc.team_player_logs("nfl", "Kansas City Chiefs", T)
    assert not NbaApiSource(Settings(nba_api_enabled=False), None, lambda *a: []).is_configured()


def test_prop_model_coherence():
    vals = [22, 25, 19, 30, 27, 24, 21, 26, 28, 23, 25, 24]
    o, po = distribution_probability(vals, "over", 24.5)
    u, pu = distribution_probability(vals, "under", 24.5)
    assert po == pu == 0 and o + u == pytest.approx(1.0)
    o2, push = distribution_probability(vals, "over", 25.0)
    u2, _ = distribution_probability(vals, "under", 25.0)
    assert push > 0 and o2 + u2 + push == pytest.approx(1.0)
    assert distribution_probability(vals[:5], "over", 24.5) is None
    lo, hi = bootstrap_interval(vals, "over", 24.5)
    assert lo < o < hi


def test_player_values_and_history_report():
    rows = nba_rows([10, 30] * 6) + [dict(nba_rows([0])[0], PLAYER_NAME="Bench Guy", MIN=0)]
    src, _ = nba_source({"2026-27": rows})
    lines = lines_to_dicts(src.team_player_logs("Boston Celtics", T))
    vals = player_values(lines, "Jayson Tatum", ("points",), "nba")
    assert len(vals) == 12 and player_values(lines, "Bench Guy", ("points",), "nba") == []   # 0 minutes excluded
    rep = prop_history_report(vals, "over", 20.5, True, "New York Knicks")
    assert rep["windows"]["all_logged"]["wins"] == 6 and rep["splits"]["vs_this_opponent"]["n"] == 12


def test_prop_settlement_through_service(repo):
    from services.settlement import SettlementService
    src, _ = nba_source({"2026-27": nba_rows([31] * 3)})
    run = repo.create_run("t", {})
    start = datetime(2026, 10, 22, 23, 30, tzinfo=timezone.utc)
    rid = repo.save_recommendation(run_id=run, kind="single", candidate_key="p1", sport_code="NBA", league="NBA",
                                   event_id="e", event_name="NYK @ BOS", home_team="Boston Celtics",
                                   away_team="New York Knicks", commence_time=start, market="player_points",
                                   selection="Jayson Tatum Over 29.5", side="over", point=29.5, odds_decimal=1.9,
                                   final_prob=0.5, status="QUALIFIED", model_version="t",
                                   report_json={"player": "Jayson Tatum"})
    svc = SettlementService(Settings(), repo, player_stats=PlayerStatsService(Settings(), sources={"nba": src}))
    assert svc.grade_recommendations(now=start + timedelta(hours=6))["graded"] == 1
    rec = repo.get_recommendation(rid)
    assert rec.result == "win" and "31" in rec.result_source
