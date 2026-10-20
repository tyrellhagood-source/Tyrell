"""Parlay pricing with explicit correlation handling.

* Legs on different events with no shared team: assumed independent ->
  P(parlay) = product of leg probabilities (label: "independent").
* Legs on the same event: priced jointly from the event's model score grid
  when available (label: "modeled_joint").
* Anything else (shared team/player across events, props, no joint model):
  the probability is UNCERTAIN. Only Frechet bounds are reported,
  max(0, sum(p) - (n-1)) <= P <= min(p), and the parlay is never recommended.

Pushes are treated as non-wins when computing P(all legs win), which slightly
understates EV (a real push reduces the parlay rather than losing it).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Callable

from models.expected_value import decimal_to_american, expected_value


@dataclass
class ParlayLeg:
    key: str
    event_id: str
    event_name: str
    teams: tuple[str, str]
    market: str
    side: str
    point: float | None
    selection: str
    decimal: float
    p_win: float
    sport_code: str = ""
    is_prop: bool = False
    meta: dict = field(default_factory=dict)


def combined_decimal(decimals) -> float:
    return float(prod(decimals))


def evaluate_parlay(legs: list[ParlayLeg], stake: float = 1.0,
                    joint_fn: Callable[[str, list[ParlayLeg]], float | None] | None = None) -> dict:
    if len(legs) < 2:
        raise ValueError("a parlay needs at least two legs")
    keys = [l.key for l in legs]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate leg in parlay")
    warnings: list[str] = []
    by_event: dict[str, list[ParlayLeg]] = {}
    for l in legs:
        by_event.setdefault(l.event_id, []).append(l)
    uncertain = False
    group_probs = []
    for event_id, group in by_event.items():
        markets = [(g.market, g.point) for g in group]
        if len(group) > 1:
            if len(set(m for m, _ in markets)) < len(markets):
                raise ValueError(f"conflicting/duplicate market legs on the same event {group[0].event_name}")
            warnings.append(f"{len(group)} legs on the same game ({group[0].event_name}) are correlated")
            p = joint_fn(event_id, group) if joint_fn and not any(g.is_prop for g in group) else None
            if p is None:
                uncertain = True
                warnings.append(f"no reliable joint model for {group[0].event_name}")
            else:
                group_probs.append(p)
        else:
            group_probs.append(group[0].p_win)
    team_events: dict[str, set] = {}
    for l in legs:
        for t in l.teams:
            team_events.setdefault(t, set()).add(l.event_id)
    shared = [t for t, evs in team_events.items() if len(evs) > 1]
    if shared:
        uncertain = True
        warnings.append(f"team(s) {', '.join(shared)} appear in more than one leg/event; correlation not estimable")
    players = [l.selection for l in legs if l.is_prop]
    if len(players) != len(set(players)):
        uncertain = True
        warnings.append("the same player appears in multiple prop legs")

    d = combined_decimal([l.decimal for l in legs])
    leg_ps = [l.p_win for l in legs]
    lower = max(0.0, sum(leg_ps) - (len(leg_ps) - 1))
    upper = min(leg_ps)
    if uncertain:
        p, status, ev = None, "uncertain", None
    else:
        p = float(prod(group_probs))
        status = "modeled_joint" if any(len(g) > 1 for g in by_event.values()) else "independent"
        ev = expected_value(min(max(p, 0.0), 1.0), d)
        if status == "independent":
            warnings.append("legs assumed independent (different games, no shared team)")
    weakest = min(legs, key=lambda l: l.p_win)
    fail_reasons = [f"Weakest leg: {weakest.selection} ({weakest.market}) at {weakest.p_win:.1%} estimated",
                    f"All {len(legs)} legs must win; one loss loses the whole parlay"]
    if p is not None:
        fail_reasons.append(f"Estimated {1 - p:.1%} chance at least one leg loses")
    return {
        "legs": legs, "n_legs": len(legs), "combined_decimal": d, "combined_american": decimal_to_american(d),
        "leg_probabilities": leg_ps, "probability": p, "probability_bounds": (lower, upper),
        "correlation_status": status, "break_even_probability": 1.0 / d, "expected_value": ev,
        "stake": stake, "payout_if_win": round(stake * d, 2), "net_profit_if_win": round(stake * (d - 1), 2),
        "max_loss": round(stake, 2), "warnings": warnings, "fail_reasons": fail_reasons,
    }
