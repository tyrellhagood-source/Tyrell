"""Integration tests of the full five-agent pipeline with OFFLINE test doubles.

The fake providers below replay hand-written fixtures / synthetic history so
the orchestration logic can be tested without network access. They are test
code only; the application never falls back to them.
"""
from datetime import datetime, timedelta, timezone

import pytest

from agents.base import AUDIT, PROB, STATS, VERIFY
from agents.chief_auditor import NO_QUALIFYING
from config import Settings
from data.odds_provider import parse_odds_response
from data.provider_base import ProviderNotConfigured, ProviderStatus
from data.sports_data_provider import parse_injuries, parse_scoreboard
from services.analysis import AnalysisService
from tests.conftest import synthetic_league

NOW = datetime(2026, 10, 21, 18, 5, tzinfo=timezone.utc)


class FakeOdds:
    def __init__(self, fixture):
        self.fixture = fixture
        self.status = ProviderStatus("the-odds-api", configured=True)
        self.credits_spent_this_run = 0

    def list_sports(self):
        return [{"key": "basketball_nba", "group": "Basketball", "title": "NBA", "active": True,
                 "has_outrights": False}]

    def get_odds(self, key, markets=None, regions=None):
        self.credits_spent_this_run += 3
        return parse_odds_response(self.fixture, NOW - timedelta(minutes=1))


class FakeESPN:
    def __init__(self, scoreboard, injuries):
        self.sb, self.inj = scoreboard, injuries
        self.status = ProviderStatus("espn", configured=True)
        teams, games = synthetic_league(n_teams=10, rounds=12, start=datetime(2026, 4, 1, tzinfo=timezone.utc))
        rename = {teams[0]: "New York Knicks", teams[9]: "Boston Celtics"}
        from data.schemas import HistoricalGame
        self.games = [HistoricalGame(g.game_id, g.date, rename.get(g.home_team, g.home_team),
                                     rename.get(g.away_team, g.away_team), g.home_score, g.away_score, False, "test")
                      for g in games if g.date < NOW]

    def history(self, *a, **k):
        return self.games

    def scoreboard(self, *a, **k):
        return parse_scoreboard(self.sb, "NBA", NOW)

    def injuries(self, path):
        return parse_injuries(self.inj, NOW)


@pytest.fixture
def service(settings, repo, load_fixture):
    s = settings.with_overrides(enabled_sports=("NBA",), bootstrap_samples=6)
    return AnalysisService(s, repo, FakeOdds(load_fixture("odds_nba.json")),
                           FakeESPN(load_fixture("espn_scoreboard.json"), load_fixture("espn_injuries.json")))


def test_full_pipeline_records_every_candidate(service, repo):
    res = service.run(now=NOW)
    assert res.status == "COMPLETED"
    assert len(res.events) == 1
    recs = repo.list_recommendations(run_id=res.run_id, kind="single")
    assert len(recs) == len(res.candidates) > 0
    for c in res.candidates:
        assert set(c.reports) == {"agent1_research", VERIFY, STATS, PROB, AUDIT}
        assert c.status in ("QUALIFIED", "NEEDS_RESEARCH", "REJECTED")
        gates = c.reports[AUDIT].data["gates"]
        if c.status == "QUALIFIED":
            assert all(gates.values())
            assert c.reports[VERIFY].verdict == "VERIFIED" and c.reports[PROB].verdict == "PASS"
    # spreads/totals are quoted by only 2 complete books -> cannot be verified
    for c in res.candidates:
        if c.market in ("spreads", "totals"):
            assert c.reports[VERIFY].verdict != "VERIFIED"
            assert c.status != "QUALIFIED"
    h2h = [c for c in res.candidates if c.market == "h2h"]
    assert all(c.reports[VERIFY].verdict == "VERIFIED" for c in h2h)
    assert res.message.endswith(NO_QUALIFYING) or "qualifying single" in res.message
    assert repo.list_audit(run_id=res.run_id)
    assert repo.latest_run().id == res.run_id
    assert repo.latest_backtests()["NBA"].status in ("OK", "INSUFFICIENT_DATA")


def test_auditor_recalculations_match(service):
    res = service.run(now=NOW)
    for c in res.candidates:
        rc = c.reports[AUDIT].data["recalculations"]
        if "expected_value" in rc:
            assert rc["expected_value"]["agent4"] == pytest.approx(rc["expected_value"]["auditor"], abs=1e-9)
        if "fair_prob" in rc:
            assert rc["fair_prob"]["agent1"] == pytest.approx(rc["fair_prob"]["auditor"], abs=1e-9)


def test_stale_odds_are_rejected(service):
    res = service.run(now=NOW + timedelta(hours=2))
    assert res.candidates
    assert all(c.status == "REJECTED" for c in res.candidates)
    assert all(c.reports["agent1_research"].verdict == "STALE" for c in res.candidates)
    assert res.message == NO_QUALIFYING


def test_missing_independent_schedule_blocks_approval(service):
    service.espn.scoreboard = lambda *a, **k: (_ for _ in ()).throw(
        __import__("data.provider_base", fromlist=["ProviderError"]).ProviderError("espn down"))
    res = service.run(now=NOW)
    assert all(c.reports[VERIFY].verdict in ("INSUFFICIENT_DATA", "REJECTED") for c in res.candidates)
    assert not [c for c in res.candidates if c.status == "QUALIFIED"]
    assert any("independent verification unavailable" in n for row in res.coverage for n in row["notes"])


def test_no_api_key_reports_no_data(settings, repo):
    from data.odds_provider import OddsApiProvider

    s = settings.with_overrides(odds_api_key="")
    svc = AnalysisService(s, repo, OddsApiProvider(s), FakeESPN({}, {}))
    res = svc.run(now=NOW)
    assert res.status == "NO_DATA" and NO_QUALIFYING in res.message and not res.candidates
    assert repo.latest_run(status=None).status == "NO_DATA"


def test_duplicate_and_same_game_correlation(service):
    res = service.run(now=NOW)
    auditor = service.agent5
    cands = res.candidates
    for c in cands:
        c.status = "QUALIFIED"
        c.consensus_score = 0.5
    import copy
    dup = copy.copy(cands[0])
    dup.reports = {k: copy.deepcopy(v) for k, v in cands[0].reports.items()}
    dup.consensus_score = 0.4
    auditor.finalize_portfolio(cands + [dup], 25.0, 0.0)
    assert sum(1 for c in cands if c.status == "QUALIFIED") == 1  # one pick per game
    assert dup.status == "REJECTED" and "duplicate recommendation" in dup.reports[AUDIT].reasons
