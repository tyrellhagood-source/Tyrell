"""Odds conversion, implied probability, vig removal, expected value, Kelly.

Conventions
-----------
* Decimal odds D include the stake: a winning $1 bet returns $D.
* American odds A: A >= +100 -> D = 1 + A/100 ; A <= -100 -> D = 1 + 100/|A|.
* Expected profit per $1 staked = p * D - 1 (pushes ignored), or more generally
  p_win * (D - 1) - p_loss when a push probability exists.
"""
from __future__ import annotations

import math
from typing import Sequence


def american_to_decimal(american: float) -> float:
    a = float(american)
    if -100 < a < 100:
        raise ValueError(f"Invalid American odds {american}: must be <= -100 or >= +100")
    return 1.0 + a / 100.0 if a > 0 else 1.0 + 100.0 / abs(a)


def decimal_to_american(decimal: float) -> float:
    d = float(decimal)
    if d <= 1.0:
        raise ValueError(f"Invalid decimal odds {decimal}: must be > 1.0")
    return (d - 1.0) * 100.0 if d >= 2.0 else -100.0 / (d - 1.0)


def format_american(decimal: float) -> str:
    a = decimal_to_american(decimal)
    return f"+{a:.0f}" if a > 0 else f"{a:.0f}"


def implied_probability(decimal: float) -> float:
    if decimal <= 1.0:
        raise ValueError("decimal odds must be > 1.0")
    return 1.0 / decimal


def overround(decimals: Sequence[float]) -> float:
    """Bookmaker margin: sum of implied probabilities minus 1."""
    return sum(implied_probability(d) for d in decimals) - 1.0


def remove_vig(decimals: Sequence[float], method: str = "multiplicative") -> list[float]:
    """Fair (no-vig) probabilities for a complete set of mutually exclusive outcomes.

    multiplicative: p_i = q_i / sum(q).  power: find k with sum(q_i^k) = 1
    (shades longshots more, often better calibrated)."""
    q = [implied_probability(d) for d in decimals]
    if len(q) < 2:
        raise ValueError("need at least two outcomes to remove vig")
    total = sum(q)
    if method == "multiplicative":
        return [x / total for x in q]
    if method == "power":
        lo, hi = 0.5, 3.0
        for _ in range(100):
            k = (lo + hi) / 2
            s = sum(x ** k for x in q)
            if s > 1:
                lo = k
            else:
                hi = k
        k = (lo + hi) / 2
        p = [x ** k for x in q]
        t = sum(p)
        return [x / t for x in p]
    raise ValueError(f"unknown method {method}")


def expected_value(p_win: float, decimal: float, p_push: float = 0.0) -> float:
    """Expected profit per $1 stake."""
    if not 0 <= p_win <= 1 or not 0 <= p_push <= 1 or p_win + p_push > 1 + 1e-12:
        raise ValueError("invalid probabilities")
    p_loss = 1.0 - p_win - p_push
    return p_win * (decimal - 1.0) - p_loss


def break_even_probability(decimal: float) -> float:
    return implied_probability(decimal)


def kelly_fraction(p_win: float, decimal: float, p_push: float = 0.0) -> float:
    """Full-Kelly bankroll fraction (0 if no edge). Push mass is removed by
    conditioning on a decided bet."""
    decided = 1.0 - p_push
    if decided <= 0:
        return 0.0
    p = p_win / decided
    b = decimal - 1.0
    f = (p * b - (1 - p)) / b
    return max(0.0, f) * decided


def profit_if_win(stake: float, decimal: float) -> float:
    return round(stake * (decimal - 1.0), 2)


def log_safe(p: float, eps: float = 1e-12) -> float:
    return math.log(min(max(p, eps), 1 - eps))
