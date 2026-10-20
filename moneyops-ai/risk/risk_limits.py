"""Risk gates applied before anything can be recommended or accepted."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from utils.timeutil import minutes_between


class RiskLimitError(Exception):
    pass


@dataclass
class RiskCheck:
    passed: bool = True
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> None:
        self.passed = False
        self.reasons.append(reason)


def odds_age_minutes(last_update: datetime, now: datetime) -> float:
    return max(0.0, minutes_between(now, last_update))


def is_stale(last_update: datetime, now: datetime, live: bool, settings) -> bool:
    limit = settings.stale_odds_minutes_live if live else settings.stale_odds_minutes_pregame
    return odds_age_minutes(last_update, now) > limit


def check_new_wager(stake: float, bankroll: float, open_exposure: float, peak_balance: float,
                    settings, is_parlay: bool = False) -> RiskCheck:
    rc = RiskCheck()
    max_pct = settings.parlay_max_stake_pct if is_parlay else settings.max_stake_pct
    if stake <= 0:
        rc.fail("stake must be positive")
    if stake > bankroll + 1e-9:
        rc.fail(f"stake ${stake:.2f} exceeds available virtual balance ${bankroll:.2f}")
    # small tolerance for cent rounding
    if stake > round(bankroll * max_pct, 2) + 0.005:
        rc.fail(f"stake ${stake:.2f} exceeds {max_pct:.1%} of bankroll (${bankroll * max_pct:.2f})")
    total_equity = bankroll + open_exposure
    if open_exposure + stake > total_equity * settings.max_open_exposure_pct + 0.005:
        rc.fail(f"open exposure would be ${open_exposure + stake:.2f}, above "
                f"{settings.max_open_exposure_pct:.0%} of equity (${total_equity * settings.max_open_exposure_pct:.2f})")
    if peak_balance > 0 and (peak_balance - total_equity) / peak_balance >= settings.max_drawdown_halt_pct:
        rc.fail(f"drawdown halt: equity ${total_equity:.2f} is {settings.max_drawdown_halt_pct:.0%}+ below "
                f"peak ${peak_balance:.2f}; new wagers paused")
    return rc
