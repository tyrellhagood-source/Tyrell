import pytest

from models.expected_value import decimal_to_american
from models.parlay import ParlayLeg, combined_decimal, evaluate_parlay


def leg(key, event, teams, p, d, market="h2h", side="home", point=None, prop=False, sel=None):
    return ParlayLeg(key, event, event, teams, market, side, point, sel or key, d, p, is_prop=prop)


def test_combined_odds():
    assert combined_decimal([2.0, 1.5, 3.0]) == pytest.approx(9.0)


def test_independent_parlay():
    legs = [leg("a", "e1", ("A", "B"), 0.6, 1.9), leg("b", "e2", ("C", "D"), 0.55, 2.0)]
    r = evaluate_parlay(legs, stake=1.0)
    assert r["correlation_status"] == "independent"
    assert r["probability"] == pytest.approx(0.33)
    assert r["combined_decimal"] == pytest.approx(3.8)
    assert r["combined_american"] == pytest.approx(decimal_to_american(3.8))
    assert r["expected_value"] == pytest.approx(0.33 * 3.8 - 1)
    assert r["break_even_probability"] == pytest.approx(1 / 3.8)
    assert r["net_profit_if_win"] == pytest.approx(2.8) and r["max_loss"] == 1.0


def test_same_game_without_joint_model_is_uncertain():
    legs = [leg("a", "e1", ("A", "B"), 0.6, 1.9), leg("b", "e1", ("A", "B"), 0.5, 1.9, "totals", "over", 200.5)]
    r = evaluate_parlay(legs)
    assert r["correlation_status"] == "uncertain"
    assert r["probability"] is None and r["expected_value"] is None
    lo, hi = r["probability_bounds"]
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.5)


def test_same_game_with_joint_model():
    legs = [leg("a", "e1", ("A", "B"), 0.6, 1.9), leg("b", "e1", ("A", "B"), 0.5, 1.9, "totals", "over", 200.5)]
    r = evaluate_parlay(legs, joint_fn=lambda ev, group: 0.34)
    assert r["correlation_status"] == "modeled_joint"
    assert r["probability"] == pytest.approx(0.34)


def test_shared_team_across_events_is_uncertain():
    legs = [leg("a", "e1", ("A", "B"), 0.6, 1.9), leg("b", "e2", ("A", "C"), 0.6, 1.9)]
    assert evaluate_parlay(legs)["correlation_status"] == "uncertain"


def test_duplicate_and_conflicting_legs_rejected():
    l1 = leg("a", "e1", ("A", "B"), 0.6, 1.9)
    with pytest.raises(ValueError):
        evaluate_parlay([l1, l1])
    with pytest.raises(ValueError):
        evaluate_parlay([l1, leg("b", "e1", ("A", "B"), 0.4, 2.2, side="away")])
    with pytest.raises(ValueError):
        evaluate_parlay([l1])


def test_large_payout_does_not_imply_value():
    legs = [leg(str(i), f"e{i}", (f"T{i}", f"U{i}"), 0.25, 3.5) for i in range(3)]
    r = evaluate_parlay(legs)
    assert r["combined_decimal"] == pytest.approx(42.875)
    assert r["expected_value"] < 0
