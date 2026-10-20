"""Small statistical helpers: Wilson intervals, streaks, hit rates."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def hit_rate(outcomes: Sequence[str]) -> dict:
    """outcomes: 'W' | 'L' | 'P' sequence. Pushes excluded from the rate."""
    w = sum(1 for o in outcomes if o == "W")
    l = sum(1 for o in outcomes if o == "L")
    p = sum(1 for o in outcomes if o == "P")
    n = w + l
    lo, hi = wilson_interval(w, n)
    return {"wins": w, "losses": l, "pushes": p, "n": n,
            "rate": (w / n) if n else None, "ci_low": lo if n else None, "ci_high": hi if n else None}


def streaks(outcomes: Iterable[str]) -> dict:
    """Longest win/loss streaks and current streak, ignoring pushes/voids."""
    longest_w = longest_l = cur = 0
    cur_type = None
    for o in outcomes:
        if o not in ("W", "L"):
            continue
        if o == cur_type:
            cur += 1
        else:
            cur_type, cur = o, 1
        if o == "W":
            longest_w = max(longest_w, cur)
        else:
            longest_l = max(longest_l, cur)
    current = f"{cur_type}{cur}" if cur_type else "-"
    return {"longest_win": longest_w, "longest_loss": longest_l, "current": current}
