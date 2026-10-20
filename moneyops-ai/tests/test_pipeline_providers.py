"""Pipeline integration with AllSportsAPI and nba_api TEST DOUBLES."""
from datetime import datetime, timedelta, timezone

import pytest

from agents.base import AUDIT, PROB, STATS, VERIFY
from data.allsports_provider import parse_fixture, parse_odds
from data.odds_provider import parse_odds_response
from data.provider_base import ProviderStatus
from data.player_stats_provider import PlayerStatsService
from services.analysis import AnalysisService
from tests.test_pipeline import NOW, FakeESPN, FakeOdds
from tests.test_player_stats import nba_rows, nba_source


class FakeAllSports:
    def __init__(self, fixtures_body, odds_body, probs=None):
        self.fx, self.od, self.probs = fixtures_body, odds_body, probs or {}
        self.status = ProviderStatus("allsportsapi", configured=True)
        self.requests_this_run = 0

    def is_configured(self):
        return True

    def resolve_league(self, path, name, country):
        return "766"

    def fixtures(self, sport_code, path, start, end, league, has_draw, ttl=None):
        games = [parse_fixture(i, sport_code, "America/New_York", NOW, has_draw) for i in self.fx["result"]]
        games = [g for g in games if g]
        return [g for g, _ in games], {g.game_id: e for g, e in games}

    def odds(self, sport_code, sport_key, league, path, start, end, league_key, has_draw, fixtures):
        return parse_odds(self.od["result"], fixtures, sport_code, sport_key, league, has_draw, NOW)

    def probabilities(self, path, start, end, league):
        return self.probs

    def history(self, *a, **k):
        return []


def _odds_with_fresh_epoch(load_fixture, price_home=1.50, price_away=2.70):
    body = load_fixture("allsports_odds.json")
    epoch = int((NOW - timedelta(minutes=5)).timestamp())
    for rows in body["result"].values():
        for r in rows:
            if "odd_epoch" in r:
                r["odd_epoch"] = epoch
    body["result"]["1001"][0].update({"odd_1": str(price_home), "odd_2": str(price_away)})
    return body


@pytest.fixture
def base(settings, repo, load_fixture):
    s = settings.with_overrides(enabled_sports=("NBA",), bootstrap_samples=6)
    return s, repo, load_fixture


def test_allsports_merge_adds_books_and_second_schedule(base):
    s, repo, fx = base
    svc = AnalysisService(s, repo, FakeOdds(fx("odds_nba.json")), FakeESPN(fx("espn_scoreboard.json"), fx("espn_injuries.json")),
                          FakeAllSports(fx("allsports_fixtures.json"), _odds_with_fresh_epoch(fx),
                                        {"1001": {"home": 0.62, "draw": 0.0, "away": 0.38}}),
                          PlayerStatsService(s, sources={}))
    res = svc.run(now=NOW)
    assert res.status == "COMPLETED"
    ml = next(c for c in res.candidates if c.market == "h2h" and c.side == "home")
    assert "as:bet365" in ml.fair_by_book and len(ml.fair_by_book) == 4
    v = ml.reports[VERIFY]
    assert set(v.data["independent_sources"]) == {"espn", "allsportsapi"}
    cross = [ch for ch in v.data["checks"] if ch["check"] == "cross_provider_odds"]
    assert cross and cross[0]["passed"]
    assert ml.reports[PROB].data["external_prob_allsports"] == pytest.approx(0.62)
    assert "allsportsapi" in ml.reports["agent1_research"].data["sources"]
    row = res.coverage[0]
    assert row["allsports"].startswith("2 fixtures, odds merged into 1 game")


def test_cross_provider_disagreement_rejects(base):
    s, repo, fx = base
    svc = AnalysisService(s, repo, FakeOdds(fx("odds_nba.json")), FakeESPN(fx("espn_scoreboard.json"), fx("espn_injuries.json")),
                          FakeAllSports(fx("allsports_fixtures.json"), _odds_with_fresh_epoch(fx, 2.40, 1.60)),
                          PlayerStatsService(s, sources={}))
    res = svc.run(now=NOW)
    ml = next(c for c in res.candidates if c.market == "h2h" and c.side == "home")
    assert ml.reports[VERIFY].verdict == "REJECTED"
    assert any("cross_provider_odds" in r for r in ml.reports[VERIFY].reasons)


class PropOdds(FakeOdds):
    def __init__(self, fixture, props):
        super().__init__(fixture)
        self.props = props

    def get_event_odds(self, sport_key, event_id, markets):
        return parse_odds_response(self.props, NOW - timedelta(minutes=1))


def _prop_service(s, repo, fx, injuries=None):
    s = s.with_overrides(include_player_props=True)
    prev = nba_rows([24, 27, 21, 30, 26, 19, 28, 25, 23, 29, 22, 26, 31, 24], start=datetime(2026, 2, 1))
    src, _ = nba_source({"2025-26": prev, "2026-27": []})
    sp = PlayerStatsService(s, sources={"nba": src})
    return AnalysisService(s, repo, PropOdds(fx("odds_nba.json"), fx("odds_nba_props.json")),
                           FakeESPN(fx("espn_scoreboard.json"), injuries or fx("espn_injuries.json")),
                           FakeAllSports({"result": []}, {"result": {}}), sp)


def test_player_prop_full_pipeline(base):
    s, repo, fx = base
    res = _prop_service(s, repo, fx).run(now=NOW)
    props = [c for c in res.candidates if c.is_prop]
    assert len(props) == 2
    over = next(c for c in props if c.side == "over")
    assert over.reports[VERIFY].verdict == "VERIFIED", over.reports[VERIFY].reasons
    st = over.reports[STATS]
    assert st.verdict in ("SUPPORTS", "NEUTRAL", "CONTRADICTS") and st.data["primary"]["n"] == 14
    p4 = over.reports[PROB].data
    assert p4["model_prob"] is not None and "player performance distribution" in p4["method"]
    assert p4["final_prob_interval"][0] <= p4["final_prob"] <= p4["final_prob_interval"][1]
    assert over.reports[AUDIT].data["gates"]["market_supported"]
    cov = res.coverage[0]
    assert cov["prop_logs"].startswith("nba_api: ")
    assert any("New York Knicks" in n for n in cov["notes"])     # no Knicks lines -> reported, not invented
    rec = next(r for r in repo.list_recommendations(res.run_id) if r.market == "player_points" and r.side == "over")
    assert rec.report_json["player"] == "Jayson Tatum"


def test_injured_player_prop_rejected(base):
    s, repo, fx = base
    inj = {"injuries": [{"displayName": "Boston Celtics", "injuries": [
        {"status": "Out", "shortComment": "Achilles", "athlete": {"displayName": "Jayson Tatum"}}]}]}
    res = _prop_service(s, repo, fx, inj).run(now=NOW)
    for c in (c for c in res.candidates if c.is_prop):
        assert c.reports[VERIFY].verdict == "REJECTED" and c.status == "REJECTED"
