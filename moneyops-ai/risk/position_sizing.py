"""Stake sizing: fractional Kelly, hard-capped at a % of the virtual bankroll.

The cap (default 1% of the *current* bankroll) always wins over Kelly, and
the stake never grows to chase losses: it is a function of the current
balance and the estimated edge only.
"""
from __future__ import annotations

import math

from models.expected_value import kelly_fraction


def floor_cents(x: float) -> float:
    return math.floor(x * 100 + 1e-9) / 100


def suggested_stake(p_win: float, decimal: float, bankroll: float, max_pct: float, kelly_mult: float,
                    min_stake: float, p_push: float = 0.0, typical_min_bet: float | None = None) -> dict:
    full = kelly_fraction(p_win, decimal, p_push)
    if full <= 0 or bankroll <= 0:
        return {"stake": 0.0, "kelly_full": full, "capped_by": "no edge" if full <= 0 else "empty bankroll",
                "notes": []}
    kelly_amt = bankroll * full * kelly_mult
    cap_amt = bankroll * max_pct
    stake = floor_cents(min(kelly_amt, cap_amt))
    capped_by = "max_stake_pct" if cap_amt < kelly_amt else "fractional_kelly"
    notes = []
    if stake < min_stake:
        notes.append(f"Suggested stake ${stake:.2f} is below the configured minimum ${min_stake:.2f}; not staked.")
        stake, capped_by = 0.0, "below_min_stake"
    if typical_min_bet and 0 < stake < typical_min_bet:
        notes.append(f"Paper stake ${stake:.2f} is below a typical sportsbook minimum (~${typical_min_bet:.2f}); "
                     "a real book may not accept this size. Results are tracked in simulation only.")
    return {"stake": stake, "kelly_full": full, "capped_by": capped_by, "notes": notes}
