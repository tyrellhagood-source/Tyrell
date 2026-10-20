from datetime import datetime, timedelta, timezone

import pytest

from data.results import VerifiedResult
from risk.bankroll import BankrollService
from services.settlement import SettlementService, grade_selection
from tests.test_bankroll import NOW, make_rec


@pytest.mark.parametrize("market,side,point,h,a,has_draw,expected", [
    ("h2h", "home", None, 100, 90, False, "win"),
    ("h2h", "away", None, 100, 90, False, "loss"),
    ("h2h", "home", None, 20, 20, False, "push"),     # NFL tie on 2-way ML
    ("h2h", "home", None, 1, 1, True, "loss"),        # soccer draw
    ("h2h", "draw", None, 1, 1, True, "win"),
    ("spreads", "home", -6.5, 100, 94, False, "loss"),
    ("spreads", "home", -6.0, 100, 94, False, "push"),
    ("spreads", "away", 6.5, 100, 94, False, "win"),
    ("totals", "over", 200.5, 100, 101, False, "win"),
    ("totals", "under", 201, 100, 101, False, "push"),
    ("totals", "under", 210.5, 100, 101, False, "win"),
])
def test_grade_selection(market, side, point, h, a, has_draw, expected):
    assert grade_selection(market, side, point, h, a, has_draw) == expected


class FakeResults:
    def __init__(self, result):
        self.result = result

    def lookup(self, *args):
        return self.result


def test_settle_open_wagers_from_verified_result(repo, settings):
    b = BankrollService(repo, settings)
    rid = make_rec(repo)
    wid = b.approve(b.propose(rid, now=NOW), now=NOW)
    svc = SettlementService(settings, repo)
    svc.results = FakeResults(VerifiedResult("FINAL", 110, 100, ("espn",)))
    out = svc.settle_open_wagers(now=NOW + timedelta(hours=8))
    assert out["settled"] == 1
    w = repo.wagers()[0]
    assert w.result == "win" and w.settlement_source.startswith("final 110-100")
    assert svc.grade_recommendations(now=NOW + timedelta(hours=8))["graded"] == 1
    assert repo.get_recommendation(rid).result == "win"


@pytest.mark.parametrize("status", ["CONFLICT", "UNAVAILABLE", "NOT_FINAL"])
def test_unverified_results_stay_pending(repo, settings, status):
    b = BankrollService(repo, settings)
    b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    svc = SettlementService(settings, repo)
    svc.results = FakeResults(VerifiedResult(status, detail="x"))
    out = svc.settle_open_wagers(now=NOW + timedelta(hours=8))
    assert out["settled"] == 0 and out["pending"] == 1
    assert repo.wagers()[0].status == "OPEN"


def test_postponed_is_void(repo, settings):
    b = BankrollService(repo, settings)
    b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    svc = SettlementService(settings, repo)
    svc.results = FakeResults(VerifiedResult("POSTPONED", sources=("espn",)))
    svc.settle_open_wagers(now=NOW + timedelta(hours=8))
    assert repo.wagers()[0].result == "void" and b.summary()["balance"] == 25.0


def test_results_service_detects_conflicts():
    from data.results import ResultsService
    from data.schemas import FinalScore

    rs = ResultsService()
    rs._espn_final = lambda *a: (FinalScore("B", "A", 100, 90, True, "espn", NOW), "")
    rs._odds_final = lambda *a: FinalScore("B", "A", 100, 91, True, "odds", NOW)
    assert rs.lookup("NBA", "basketball_nba", "e", "B", "A", NOW).status == "CONFLICT"
    rs._odds_final = lambda *a: FinalScore("B", "A", 100, 90, True, "odds", NOW)
    r = rs.lookup("NBA", "basketball_nba", "e", "B", "A", NOW)
    assert r.status == "FINAL" and len(r.sources) == 2
