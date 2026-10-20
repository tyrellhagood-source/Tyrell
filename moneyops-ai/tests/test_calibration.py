import random
from datetime import timedelta

import pytest

from data.sports_registry import SPORTS
from models.backtesting import chronological_split, walk_forward
from models.calibration import (brier_score, brier_skill_score, calibration_table, expected_calibration_error,
                                log_loss)
from models.performance import summarize
from tests.conftest import synthetic_league


def test_brier_and_log_loss():
    assert brier_score([0.5, 0.5], [1, 0]) == 0.25
    assert brier_score([], []) is None
    assert log_loss([0.5], [1]) == pytest.approx(0.6931, abs=1e-4)
    assert brier_skill_score(0.2, 0.25) == pytest.approx(0.2)


def test_calibration_of_calibrated_vs_overconfident():
    rng = random.Random(0)
    probs = [rng.random() for _ in range(20000)]
    ys = [1 if rng.random() < p else 0 for p in probs]
    assert expected_calibration_error(probs, ys) < 0.02
    over = [min(1, max(0, 0.5 + (p - 0.5) * 1.8)) for p in probs]  # overconfident version
    assert expected_calibration_error(over, ys) > 0.05
    table = calibration_table(probs, ys)
    eighty = [r for r in table if r["bin_low"] == 0.8][0]
    assert abs(eighty["observed_rate"] - eighty["mean_predicted"]) < 0.03


def test_chronological_split_has_no_overlap():
    _, games = synthetic_league(rounds=8)
    tr, va, te = chronological_split(games)
    assert max(g.date for g in tr) <= min(g.date for g in va)
    assert max(g.date for g in va) <= min(g.date for g in te)
    assert len(tr) + len(va) + len(te) == len(games)


def test_walk_forward_backtest_reports_out_of_sample_metrics():
    _, games = synthetic_league(n_teams=12, rounds=12)
    res = walk_forward(SPORTS["NBA"], games)
    assert res["status"] == "OK" and "HYPOTHETICAL" in res["label"]
    assert res["n_predictions"] >= 20
    for p in res["predictions"]:  # every prediction is out-of-sample
        assert p["date"] >= res["test_period"][0]
    assert res["brier"] < res["baseline_brier"]  # synthetic data has real structure
    assert walk_forward(SPORTS["NBA"], games[:30])["status"] == "INSUFFICIENT_DATA"


def test_performance_summary_definitions():
    recs = [
        {"ts": "1", "result": "win", "odds": 2.0, "prob": 0.6},
        {"ts": "2", "result": "loss", "odds": 2.0, "prob": 0.6},
        {"ts": "3", "result": "loss", "odds": 2.0, "prob": 0.6},
        {"ts": "4", "result": "push", "odds": 2.0, "prob": 0.6},
        {"ts": "5", "result": "void", "odds": 2.0, "prob": 0.6},
        {"ts": "6", "result": "pending", "odds": 2.0, "prob": 0.6},
        {"ts": "7", "result": "win", "odds": 3.0, "prob": 0.4, "closing_odds": 2.5},
    ]
    s = summarize(recs)
    assert s["total"] == 7 and s["settled"] == 6 and s["open"] == 1
    assert (s["wins"], s["losses"], s["pushes"], s["voids"]) == (2, 2, 1, 1)
    assert s["win_rate"] == 0.5
    assert s["units"] == pytest.approx(1 - 1 - 1 + 2)
    assert s["yield"] == pytest.approx(1 / 5)
    assert s["max_drawdown_units"] == pytest.approx(2.0)
    assert s["longest_loss_streak"] == 2 and s["longest_win_streak"] == 1
    assert s["clv"] == pytest.approx(3.0 / 2.5 - 1) and s["clv_n"] == 1
    assert s["brier"] == pytest.approx(((0.4) ** 2 + 0.6 ** 2 + 0.6 ** 2 + 0.6 ** 2) / 4)
