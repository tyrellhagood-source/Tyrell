from datetime import datetime, timedelta, timezone

from data.schemas import Event, OutcomePrice
from data.provider_base import ProviderStatus
from database.repository import Repository

NOW = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)


def test_persistence_across_connections(settings):
    r1 = Repository(settings.database_url)
    r1.init_db()
    run = r1.create_run("v", {"x": 1})
    r1.audit("agent", "act", "msg apiKey=shouldnotleak", run_id=run)
    r1.finish_run(run, "COMPLETED", {"message": "ok"}, [{"sport": "NBA"}], ["err apiKey=leak"])
    r2 = Repository(settings.database_url)
    assert r2.latest_run().summary_json["message"] == "ok"
    assert "shouldnotleak" not in r2.list_audit()[0].message
    assert "leak" not in r2.latest_run().errors_json[0]


def test_closing_price_uses_last_pre_start_snapshot(repo):
    start = NOW + timedelta(hours=1)
    for i, (minutes_before, price) in enumerate([(300, 1.90), (20, 1.80)]):
        run = repo.create_run("v", {})
        retrieved = start - timedelta(minutes=minutes_before)
        ev = Event("e1", "NBA", "basketball_nba", "NBA", start, "B", "A", "the-odds-api", retrieved,
                   [OutcomePrice("h2h", "dk", "DK", "B", price, None, retrieved)])
        repo.save_odds_snapshots(run, [ev])
    assert repo.closing_price("e1", "h2h", "B", None, "dk", start) == 1.80
    assert repo.closing_price("e1", "h2h", "B", None, "dk", start, max_gap_minutes=10) is None


def test_provider_status_accumulates(repo):
    st = ProviderStatus("espn", configured=True, requests_made=3, quota_remaining=None)
    repo.save_provider_status(st)
    repo.save_provider_status(st)
    row = repo.list_provider_status()[0]
    assert row.requests_made == 6
