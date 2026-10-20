import math

import pytest

from models.expected_value import (american_to_decimal, break_even_probability, decimal_to_american,
                                   expected_value, format_american, implied_probability, kelly_fraction,
                                   overround, remove_vig)


@pytest.mark.parametrize("american,decimal", [(100, 2.0), (150, 2.5), (-110, 1.0 + 100 / 110), (-200, 1.5),
                                              (-100, 2.0), (250, 3.5)])
def test_american_to_decimal(american, decimal):
    assert american_to_decimal(american) == pytest.approx(decimal)


@pytest.mark.parametrize("decimal,american", [(2.0, 100), (2.5, 150), (1.5, -200), (1.909090909, -110), (3.5, 250)])
def test_decimal_to_american(decimal, american):
    assert decimal_to_american(decimal) == pytest.approx(american, rel=1e-6)


def test_round_trip():
    for a in (-450, -110, 100, 120, 900):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)
    assert format_american(1.5) == "-200" and format_american(2.5) == "+150"


@pytest.mark.parametrize("bad", [0, 50, -99.9])
def test_invalid_american(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


def test_invalid_decimal():
    with pytest.raises(ValueError):
        implied_probability(1.0)
    with pytest.raises(ValueError):
        decimal_to_american(0.9)


def test_implied_probability_and_break_even():
    assert implied_probability(2.0) == 0.5
    assert break_even_probability(american_to_decimal(-110)) == pytest.approx(110 / 210)


def test_vig_removal_multiplicative():
    d = american_to_decimal(-110)
    assert overround([d, d]) == pytest.approx(2 * 110 / 210 - 1)
    assert remove_vig([d, d]) == pytest.approx([0.5, 0.5])
    p = remove_vig([1.5, 2.7])
    assert sum(p) == pytest.approx(1.0)
    assert p[0] == pytest.approx((1 / 1.5) / (1 / 1.5 + 1 / 2.7))


def test_vig_removal_power_and_three_way():
    p = remove_vig([2.1, 3.4, 3.6], method="power")
    assert sum(p) == pytest.approx(1.0)
    m = remove_vig([2.1, 3.4, 3.6])
    assert p[0] > m[0]  # power method shades longshots harder than multiplicative
    with pytest.raises(ValueError):
        remove_vig([2.0])


def test_expected_value_formula():
    # EV per $1 = p*D - 1
    assert expected_value(0.55, 2.0) == pytest.approx(0.10)
    assert expected_value(0.5, american_to_decimal(-110)) == pytest.approx(0.5 * (1 + 100 / 110) - 1)


def test_high_probability_can_be_negative_ev():
    # 80% to win but priced at -500 (D = 1.2): EV = 0.8*1.2 - 1 = -0.04
    assert expected_value(0.80, american_to_decimal(-500)) == pytest.approx(-0.04)


def test_expected_value_with_push():
    # win 45%, push 10%, loss 45% at evens -> EV 0
    assert expected_value(0.45, 2.0, 0.10) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        expected_value(0.8, 2.0, 0.3)


def test_kelly():
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 2.0) == 0.0
    assert math.isclose(kelly_fraction(0.45, 2.0, 0.10), 0.0, abs_tol=1e-12)
