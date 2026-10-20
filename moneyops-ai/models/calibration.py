"""Probability scoring: Brier score, log loss, calibration tables, ECE.

Brier = mean((p - y)^2); lower is better, 0.25 = coin flip at p=0.5.
Log loss = -mean(y log p + (1-y) log(1-p)).
Calibration: predictions are bucketed; in a well-calibrated model the
observed frequency in each bucket matches the mean predicted probability.
ECE = sum_b (n_b / N) * |observed_b - predicted_b|.
"""
from __future__ import annotations

import math
from typing import Sequence


def brier_score(probs: Sequence[float], outcomes: Sequence[int]) -> float | None:
    if not probs:
        return None
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs)


def log_loss(probs: Sequence[float], outcomes: Sequence[int], eps: float = 1e-12) -> float | None:
    if not probs:
        return None
    total = 0.0
    for p, y in zip(probs, outcomes):
        p = min(max(p, eps), 1 - eps)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(probs)


def calibration_table(probs: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> list[dict]:
    rows = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        mean_p = sum(probs[i] for i in idx) / len(idx)
        obs = sum(outcomes[i] for i in idx) / len(idx)
        rows.append({"bin_low": lo, "bin_high": hi, "n": len(idx), "mean_predicted": mean_p, "observed_rate": obs})
    return rows


def expected_calibration_error(probs: Sequence[float], outcomes: Sequence[int], bins: int = 10) -> float | None:
    if not probs:
        return None
    table = calibration_table(probs, outcomes, bins)
    n = len(probs)
    return sum(r["n"] / n * abs(r["observed_rate"] - r["mean_predicted"]) for r in table)


def brier_skill_score(model_brier: float, baseline_brier: float) -> float | None:
    if baseline_brier in (None, 0) or model_brier is None:
        return None
    return 1.0 - model_brier / baseline_brier
