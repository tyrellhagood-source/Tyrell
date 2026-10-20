from fastapi.testclient import TestClient


def test_api_smoke(monkeypatch, settings, repo):
    import api.server as server
    monkeypatch.setattr(server, "get_settings", lambda: settings.with_overrides(odds_api_key=""))
    monkeypatch.setattr(server, "get_repo", lambda: repo)
    c = TestClient(server.app)
    h = c.get("/health").json()
    assert h["paper_trading_only"] is True and h["odds_api_key"] == "missing"
    assert "test-key" not in c.get("/settings").text
    assert c.get("/bankroll").json()["balance"] == 25.0
    assert c.get("/recommendations").json()["items"] == []
    run = c.post("/analysis/run").json()
    assert run["status"] == "NO_DATA" and "NO QUALIFYING BETS FOUND." in run["message"]
    assert c.post("/bankroll/reset", json={"confirmation": "nope"}).status_code == 400
    assert c.get("/export/wagers").text.startswith("id,")
