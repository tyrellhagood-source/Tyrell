import logging
from datetime import datetime, timezone

import pytest
import requests

from config import Settings
from data.odds_provider import OddsApiProvider, parse_odds_response, parse_scores_response
from data.provider_base import (ProviderAuthError, ProviderBudgetExceeded, ProviderError, ProviderNotConfigured)
from data.sports_data_provider import parse_injuries, parse_scoreboard
from utils.security import SecretRedactingFilter, redact, register_secret
from utils.teams import match_team, normalize_team

T = datetime(2026, 10, 21, 18, 5, tzinfo=timezone.utc)


def test_parse_odds_skips_malformed(load_fixture):
    events = parse_odds_response(load_fixture("odds_nba.json"), T)
    assert len(events) == 1  # "broken" event skipped
    ev = events[0]
    assert ev.sport_code == "NBA" and ev.home_team == "Boston Celtics" and ev.source == "the-odds-api"
    books = {p.bookmaker for p in ev.prices}
    assert "badbook" not in books  # no timestamp / invalid price -> never used
    assert all(p.last_update is not None for p in ev.prices)
    spread = [p for p in ev.prices if p.market == "spreads" and p.bookmaker == "draftkings"]
    assert {p.point for p in spread} == {-6.5, 6.5}


def test_parse_scores():
    body = [{"id": "1", "completed": True, "home_team": "A", "away_team": "B",
             "scores": [{"name": "A", "score": "3"}, {"name": "B", "score": "1"}]},
            {"id": "2", "completed": False, "home_team": "A", "away_team": "B", "scores": None}]
    out = parse_scores_response(body, T)
    assert list(out) == ["1"] and out["1"].home_score == 3


def test_parse_espn_scoreboard_and_injuries(load_fixture):
    games = parse_scoreboard(load_fixture("espn_scoreboard.json"), "NBA", T)
    assert len(games) == 2
    final = [g for g in games if g.completed][0]
    assert final.home_score == 110 and final.away_score == 104 and final.state == "post"
    inj = parse_injuries(load_fixture("espn_injuries.json"), T)
    assert [i.player for i in inj] == ["Player One", "Player Two"]
    assert parse_scoreboard({}, "NBA", T) == []


def test_team_matching():
    assert normalize_team("Brighton & Hove Albion") == normalize_team("Brighton and Hove Albion")
    assert match_team("LA Clippers", ["Los Angeles Clippers", "Los Angeles Lakers"])[0] == "Los Angeles Clippers"
    assert match_team("Boston Celtics", ["Denver Nuggets"])[0] is None


class FakeResp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def provider(responses, **overrides):
    s = Settings(odds_api_key="supersecretkey123", http_max_retries=2, **overrides)
    return OddsApiProvider(s, cache=None, session=FakeSession(responses))


def test_missing_key_raises_not_configured():
    p = OddsApiProvider(Settings(odds_api_key=""), session=FakeSession([]))
    with pytest.raises(ProviderNotConfigured):
        p.get_odds("basketball_nba")


def test_quota_headers_and_budget(load_fixture, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = provider([FakeResp(200, load_fixture("odds_nba.json"),
                           {"x-requests-remaining": "400", "x-requests-used": "100", "x-requests-last": "3"})],
                 max_credits_per_run=5)
    evs = p.get_odds("basketball_nba")
    assert len(evs) == 1 and p.status.quota_remaining == 400 and p.credits_spent_this_run == 3
    with pytest.raises(ProviderBudgetExceeded):
        p.get_odds("basketball_nba")  # 3 more would exceed 5


def test_reserve_respected():
    p = provider([], min_credits_reserve=25)
    p.status.quota_remaining = 26
    with pytest.raises(ProviderBudgetExceeded):
        p.get_odds("basketball_nba")


def test_retries_then_errors_are_redacted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    p = provider([FakeResp(429), FakeResp(500), FakeResp(503)])
    with pytest.raises(ProviderError) as e:
        p.get_odds("basketball_nba")
    assert "supersecretkey123" not in str(e.value)
    p2 = provider([FakeResp(401)])
    with pytest.raises(ProviderAuthError):
        p2.get_odds("basketball_nba")
    p3 = provider([requests.ConnectionError("proxy 403 apiKey=supersecretkey123")] * 3)
    with pytest.raises(ProviderError) as e3:
        p3.get_odds("basketball_nba")
    assert "supersecretkey123" not in str(e3.value) and "supersecretkey123" not in (p3.status.last_error or "")


def test_log_redaction():
    register_secret("abcdef123456")
    assert "abcdef123456" not in redact("GET https://x/?apiKey=abcdef123456&regions=us")
    assert "zzz999" not in redact("url?apiKey=zzz999")
    rec = logging.LogRecord("x", logging.INFO, "f", 1, "key %s", ("abcdef123456",), None)
    SecretRedactingFilter().filter(rec)
    assert "abcdef123456" not in rec.getMessage()


def test_public_settings_hide_key():
    assert Settings(odds_api_key="abc123456789").public_dict()["odds_api_key"] == "configured"
