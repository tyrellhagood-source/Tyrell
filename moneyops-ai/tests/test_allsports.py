"""AllSportsAPI parser/provider tests. Fixtures are HAND-WRITTEN from the
publicly documented envelope; odds/probability field names are unverified
until scripts/probe_allsports.py is run against the real API."""
from datetime import datetime, timezone

import pytest

from config import Settings
from data.allsports_provider import AllSportsProvider, parse_fixture, parse_odds, parse_probabilities
from data.provider_base import ProviderBudgetExceeded, ProviderError, ProviderNotConfigured
from data.results import ResultsService
from data.schemas import FinalScore
from tests.test_data_validation import FakeResp, FakeSession

T = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)


def test_fixture_parsing_and_timezone(load_fixture):
    items = load_fixture("allsports_fixtures.json")["result"]
    g, extra = parse_fixture(items[0], "NBA", "America/New_York", T)
    assert g.start_time == datetime(2026, 10, 21, 23, 30, tzinfo=timezone.utc)  # 19:30 EDT
    assert g.state == "pre" and not g.completed and g.home_score is None and g.source == "allsportsapi"
    f, _ = parse_fixture(items[1], "NBA", "UTC", T)
    assert f.completed and (f.home_score, f.away_score) == (110, 104)
    assert parse_fixture(items[2], "NBA", "UTC", T) is None


@pytest.mark.parametrize("item,completed,score,state", [
    ({"event_status": "Finished", "event_final_result": "2 - 1", "event_ft_result": "2 - 1"}, True, (2, 1), "post"),
    ({"event_status": "After ET", "event_final_result": "3 - 2", "event_ft_result": "2 - 2"}, True, (2, 2), "post"),
    ({"event_status": "After Pen.", "event_final_result": "1 - 1"}, False, None, "post"),   # no 90-min score
    ({"event_status": "Postponed", "event_final_result": "-"}, False, None, "post"),
    ({"event_status": "67", "event_final_result": "1 - 0"}, False, None, "in"),
])
def test_soccer_status_and_90_minute_rule(item, completed, score, state):
    base = {"event_key": 5, "event_date": "2026-10-21", "event_time": "15:00", "event_home_team": "A",
            "event_away_team": "B"}
    g, extra = parse_fixture(base | item, "EPL", "UTC", T, has_draw=True)
    assert g.completed is completed and g.state == state
    assert (None if g.home_score is None else (g.home_score, g.away_score)) == score
    if item["event_status"] == "Postponed":
        assert extra["cancelled"]


def test_tennis_player_fields():
    g, _ = parse_fixture({"event_key": 7, "event_date": "2026-10-21", "event_time": "12:00",
                          "event_first_player": "J. Sinner", "event_second_player": "C. Alcaraz",
                          "event_status": "Finished", "event_final_result": "2 - 1"}, "TENNIS", "UTC", T)
    assert g.home_team == "J. Sinner" and g.away_team == "C. Alcaraz"


def test_odds_parsing_fail_closed(load_fixture):
    items = load_fixture("allsports_fixtures.json")["result"]
    g, _ = parse_fixture(items[0], "NBA", "America/New_York", T)
    evs = parse_odds(load_fixture("allsports_odds.json")["result"], {"1001": g}, "NBA", "basketball_nba", "NBA",
                     False, T)
    assert len(evs) == 1  # match 9999 has no fixture -> skipped
    ev = evs[0]
    books = {p.bookmaker for p in ev.prices}
    assert books == {"as:bet365"}            # Unibet quoted a 3-way market (odd_x) for basketball; NoTime lacks epoch
    assert all(p.provider == "allsportsapi" for p in ev.prices)
    assert {(p.market, p.name, p.point) for p in ev.prices} == {
        ("h2h", "Boston Celtics", None), ("h2h", "New York Knicks", None),
        ("totals", "Over", 221.5), ("totals", "Under", 221.5)}   # asian handicap skipped
    assert parse_odds([], {}, "NBA", "k", "NBA", False, T) == []


def test_soccer_three_way_odds():
    g, _ = parse_fixture({"event_key": 1, "event_date": "2026-10-21", "event_time": "15:00",
                          "event_home_team": "Arsenal", "event_away_team": "Chelsea"}, "EPL", "UTC", T, True)
    evs = parse_odds({"1": [{"odd_bookmakers": "bwin", "odd_epoch": 1792605600, "odd_1": "2.1", "odd_x": "3.4",
                             "odd_2": "3.6"}]}, {"1": g}, "EPL", "soccer_epl", "EPL", True, T)
    assert sorted(p.name for p in evs[0].prices) == ["Arsenal", "Chelsea", "Draw"]


def test_probabilities_parsing():
    out = parse_probabilities([
        {"event_key": 1, "event_HW": "50.0", "event_D": "25.0", "event_AW": "25.0"},
        {"event_key": 2, "event_HW": "abc", "event_D": "25", "event_AW": "25"},
        {"event_key": 3, "event_HW": "10", "event_D": "10", "event_AW": "10"},   # does not sum to ~100
    ])
    assert list(out) == ["1"] and out["1"]["home"] == pytest.approx(0.5)


def provider(responses, **kw):
    s = Settings(allsports_api_key="allsportssecret999", http_max_retries=0, **kw)
    return AllSportsProvider(s, None, FakeSession(responses))


def test_provider_envelope_errors_budget_and_redaction(load_fixture):
    with pytest.raises(ProviderNotConfigured):
        AllSportsProvider(Settings(), None, FakeSession([])).fixtures("NBA", "basketball", T, T, None, False)
    p = provider([FakeResp(200, {"success": 0, "error": "Wrong login credentials apiKey"})])
    with pytest.raises(ProviderError) as e:
        p.fixtures("NBA", "basketball", T, T, None, False)
    assert "allsportssecret999" not in str(e.value)
    p2 = provider([FakeResp(200, load_fixture("allsports_fixtures.json"))], allsports_max_requests_per_run=1)
    games, _ = p2.fixtures("NBA", "basketball", T, T, "766", False)
    assert len(games) == 2
    url, params = p2.session.calls[0]
    assert url.endswith("/basketball/") and params["met"] == "Fixtures" and params["leagueId"] == "766"
    with pytest.raises(ProviderBudgetExceeded):
        p2.fixtures("NBA", "basketball", T, T, "766", False)


def test_league_resolution_requires_unique_match():
    leagues = {"success": 1, "result": [
        {"league_key": 152, "league_name": "Premier League", "country_name": "England"},
        {"league_key": 999, "league_name": "Premier League", "country_name": "Wales"},
        {"league_key": 3, "league_name": "UEFA Champions League", "country_name": "eurocups"}]}
    p = provider([FakeResp(200, leagues)])
    assert p.resolve_league("football", "Premier League", "England") == "152"
    assert p.resolve_league("football", "Premier League", None) is None   # ambiguous -> gap, not a guess
    assert p.resolve_league("football", "UEFA Champions League", None) == "3"
    assert p.resolve_league("football", "Serie A", "Italy") is None


def test_results_cross_check_espn_vs_allsports():
    rs = ResultsService(settings=Settings())
    rs._espn_final = lambda *a: (FinalScore("B", "A", 100, 90, True, "espn", T), "")
    rs._allsports_final = lambda *a: (FinalScore("B", "A", 100, 91, True, "allsportsapi", T), "")
    assert rs.lookup("NBA", "basketball_nba", "e", "B", "A", T).status == "CONFLICT"
    rs._allsports_final = lambda *a: (FinalScore("B", "A", 100, 90, True, "allsportsapi", T), "")
    r = rs.lookup("NBA", "basketball_nba", "e", "B", "A", T)
    assert r.status == "FINAL" and set(r.sources) == {"espn", "allsportsapi"}
    rs._allsports_final = lambda *a: ("NOT_FINAL", "Q4")
    assert rs.lookup("NBA", "basketball_nba", "e", "B", "A", T).status == "NOT_FINAL"
