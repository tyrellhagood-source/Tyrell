from datetime import datetime, timedelta, timezone

from config import Settings
from risk.position_sizing import suggested_stake
from risk.risk_limits import check_new_wager, is_stale

NOW = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)


def test_stale_thresholds():
    s = Settings()
    assert not is_stale(NOW - timedelta(minutes=10), NOW, False, s)
    assert is_stale(NOW - timedelta(minutes=31), NOW, False, s)
    assert is_stale(NOW - timedelta(minutes=3), NOW, True, s)  # live window is 2 min


def test_stake_sizing_caps_and_minimums():
    r = suggested_stake(0.60, 2.0, 25.0, 0.01, 0.25, 0.10, typical_min_bet=1.0)
    assert r["stake"] == 0.25 and r["capped_by"] == "max_stake_pct"
    assert any("typical sportsbook minimum" in n for n in r["notes"])
    assert suggested_stake(0.45, 2.0, 25.0, 0.01, 0.25, 0.10)["stake"] == 0.0
    tiny = suggested_stake(0.505, 2.0, 25.0, 0.01, 0.25, 0.10)
    assert tiny["stake"] == 0.0 and tiny["capped_by"] == "below_min_stake"


def test_check_new_wager_gates():
    s = Settings()
    assert check_new_wager(0.25, 25.0, 0.0, 25.0, s).passed
    assert not check_new_wager(0.30, 25.0, 0.0, 25.0, s).passed           # > 1%
    assert not check_new_wager(0.20, 25.0, 1.2, 26.2, s).passed           # exposure > 5%
    assert check_new_wager(0.12, 25.0, 0.0, 25.0, s, is_parlay=True).passed
    assert not check_new_wager(0.13, 25.0, 0.0, 25.0, s, is_parlay=True).passed
    halt = check_new_wager(0.15, 18.0, 0.0, 25.0, s)                       # 28% drawdown
    assert not halt.passed and "drawdown" in halt.reasons[0]
