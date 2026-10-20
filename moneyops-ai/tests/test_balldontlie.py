"""balldontlie (free plan: games only) provider tests with HAND-WRITTEN
payloads following its documented games shape. No network."""
from datetime import datetime, timedelta, timezone

import pytest

from agents.base import VERIFY
from config import Settings
from data.balldontlie_provider import BallDontLieProvider, parse_game
from data.provider_base import ProviderBudgetExceeded, ProviderNotConfigured
from data.results import ResultsService
from data.schemas import FinalScore
from services.analysis import AnalysisService
from tests.test_data_validation import FakeResp, FakeSession
from tests.test_pipeline import NOW, FakeESPN, FakeOdds

T = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)
CELTICS = {"id": 2, "abbreviation": "BOS", "city": "Boston", "name": "Celtics", "full_name": "Boston Celtics"}
KNICKS = {"id": 20, "abbreviation": "NYK", "city": "New York", "name": "Knicks", "full_name": "New York Knicks"}


def game(**kw):
    base = {"id": 9001, "date": "2026-10-21", "season": 2026, "status": "2026-10-21T23:30:00Z", "period": 0,
            "time": None, "postseason": False, "home_team_score": 0, "visitor_team_score": 0,
            "home_team": CELTICS, "visitor_team": KNICKS}
    return base | kw


def test_parse_scheduled_final_live_cancelled_and_date_only():
    g = parse_game(game(), "NBA", T)
    assert g.state == "pre" and g.time_known and g.start_time == datetime(2026, 10, 21, 23, 30, tzinfo=timezone.utc)
    assert g.home_team == "Boston Celtics" and g.away_team == "New York Knicks" and g.home_score is None
    f = parse_game(game(status="Final", home_team_score=112, visitor_team_score=104), "NBA", T)
    assert f.completed and (f.home_score, f.away_score) == (112, 104) and not f.time_known
    live = parse_game(game(status="3rd Qtr", period=3, home_team_score=70, visitor_team_score=66), "NBA", T)
    assert live.state == "in" and not live.completed and live.home_score is None
    pp = parse_game(game(status="Postponed"), "NBA", T)
    assert pp.state == "post" and not pp.completed
    timed = parse_game(game(status="7:30 pm ET", datetime="2026-10-21T23:30:00.000Z"), "NBA", T)
    assert timed.state == "pre" and timed.time_known
    nfl = parse_game(game(status="Scheduled", date="2026-10-25T17:00:00.000Z"), "NFL", T)
    assert nfl.time_known and nfl.start_time.hour == 17
    assert parse_game({"id": 1, "home_team": CELTICS}, "NBA", T) is None


def provider(responses, **kw):
    clock = {"t": 0.0, "slept": []}

    def sleep(s):
        clock["slept"].append(s)
        clock["t"] += s
    s = Settings(balldontlie_api_key="bdl-secret-key-123", http_max_retries=0, **kw)
    return BallDontLieProvider(s, None, FakeSession(responses), lambda: clock["t"], sleep), clock


def test_header_auth_pagination_and_throttle():
    pages = [FakeResp(200, {"data": [game()], "meta": {"next_cursor": 55, "per_page": 100}}),
             FakeResp(200, {"data": [game(id=9002, home_team=KNICKS, visitor_team=CELTICS)], "meta": {}})]
    p, clock = provider(pages)
    games = p.games("NBA", "v1", T, T)
    assert [g.game_id for g in games] == ["bdl:9001", "bdl:9002"]
    url, params = p.session.calls[0]
    assert url == "https://api.balldontlie.io/v1/games" and params["start_date"] == "2026-10-21"
    assert "bdl-secret-key-123" not in str(params) and "cursor" not in params
    assert p.session.calls[1][1]["cursor"] == 55
    assert clock["slept"] == [pytest.approx(12.0)]          # free plan: 5 requests/minute


def test_key_sent_in_authorization_header():
    seen = {}

    class S(FakeSession):
        def get(self, url, params=None, timeout=None, headers=None):
            seen["headers"] = headers
            return super().get(url, params, timeout)
    s = Settings(balldontlie_api_key="bdl-secret-key-123")
    p = BallDontLieProvider(s, None, S([FakeResp(200, {"data": [], "meta": {}})]), lambda: 0.0, lambda x: None)
    p.games("NFL", "nfl/v1", T, T)
    assert seen["headers"] == {"Authorization": "bdl-secret-key-123"}
    assert p.session.calls[0][0] == "https://api.balldontlie.io/nfl/v1/games"


def test_not_configured_and_budget():
    with pytest.raises(ProviderNotConfigured):
        BallDontLieProvider(Settings(), None, FakeSession([])).games("NBA", "v1", T, T)
    p, _ = provider([FakeResp(200, {"data": [], "meta": {}})], balldontlie_max_requests_per_run=1)
    p.games("NBA", "v1", T, T)
    with pytest.raises(ProviderBudgetExceeded):
        p.games("NBA", "v1", T + timedelta(days=3), T + timedelta(days=3))


class FakeBDL:
    def __init__(self, items):
        self.items = items
        self.requests_this_run = 0
        from data.provider_base import ProviderStatus
        self.status = ProviderStatus("balldontlie", configured=True)

    def is_configured(self):
        return True

    def games(self, sport_code, path, start, end, ttl=None):
        return [g for g in (parse_game(i, sport_code, NOW) for i in self.items) if g]

    def history(self, *a, **k):
        return []


def _run(settings, repo, fx, items):
    s = settings.with_overrides(enabled_sports=("NBA",), bootstrap_samples=6)
    svc = AnalysisService(s, repo, FakeOdds(fx("odds_nba.json")),
                          FakeESPN(fx("espn_scoreboard.json"), fx("espn_injuries.json")), balldontlie=FakeBDL(items))
    return svc.run(now=NOW)


def test_third_schedule_source_date_only(settings, repo, load_fixture):
    res = _run(settings, repo, load_fixture, [game(status="7:30 pm ET")])     # date only, same ET day
    ml = next(c for c in res.candidates if c.market == "h2h")
    v = ml.reports[VERIFY]
    assert "balldontlie" in v.data["independent_sources"]
    assert any(ch["check"] == "start_date[balldontlie]" and ch["passed"] for ch in v.data["checks"])
    assert "balldontlie" in res.coverage[0]["independent_schedule"]


def test_balldontlie_wrong_date_or_final_rejects(settings, repo, load_fixture):
    res = _run(settings, repo, load_fixture, [game(date="2026-10-20", status="7:30 pm ET")])
    ml = next(c for c in res.candidates if c.market == "h2h")
    assert ml.reports[VERIFY].verdict == "REJECTED"
    res2 = _run(settings, repo, load_fixture, [game(status="Final", home_team_score=100, visitor_team_score=90)])
    ml2 = next(c for c in res2.candidates if c.market == "h2h")
    assert any("game_status[balldontlie]" in r for r in ml2.reports[VERIFY].reasons)


def test_results_three_way_cross_check():
    rs = ResultsService(settings=Settings())
    rs._espn_final = lambda *a: (FinalScore("B", "A", 100, 90, True, "espn", T), "")
    rs._allsports_final = lambda *a: (None, "")
    rs._balldontlie_final = lambda *a: (FinalScore("B", "A", 100, 90, True, "balldontlie", T), "")
    r = rs.lookup("NBA", "basketball_nba", "e", "B", "A", T)
    assert r.status == "FINAL" and set(r.sources) == {"espn", "balldontlie"}
    rs._balldontlie_final = lambda *a: (FinalScore("B", "A", 99, 90, True, "balldontlie", T), "")
    assert rs.lookup("NBA", "basketball_nba", "e", "B", "A", T).status == "CONFLICT"
