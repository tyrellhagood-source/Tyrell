"""Performance metrics for recommendations and paper wagers.

Definitions (shown on the dashboard):
* settled = wins + losses + pushes + voids (open/pending excluded).
* win rate = wins / (wins + losses). Pushes and voids are excluded.
* units: flat 1-unit stake per pick; win = +(D-1), loss = -1, push/void = 0.
* yield = units / (wins + losses + pushes)  (flat-stake return per bet).
* ROI = net profit / total amount staked (actual stakes, voids excluded);
  for flat 1-unit staking ROI equals yield.
* max drawdown = largest peak-to-trough fall of cumulative units.
* Brier / log loss use the probability recorded at recommendation time,
  wins vs losses only.
* CLV = mean(odds_taken / closing_odds - 1) over picks with a reliable
  closing price (last observed pre-start snapshot).
"""
from __future__ import annotations

from collections import defaultdict

from models.calibration import brier_score, calibration_table, expected_calibration_error, log_loss
from models.stats import streaks

PROB_BUCKETS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.01)]
ODDS_BUCKETS = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 1e9)]


def bucket_label(value, buckets) -> str:
    if value is None:
        return "n/a"
    for lo, hi in buckets:
        if lo <= value < hi:
            if hi >= 1e8:
                return f"{lo:g}+"
            return f"{lo:g}-{1.0 if hi == 1.01 else hi:g}"
    return "other"


def summarize(records: list[dict]) -> dict:
    """records: dicts with keys result (win/loss/push/void/pending), odds (decimal),
    prob, stake (optional, default 1), closing_odds (optional), ts (sortable)."""
    records = sorted(records, key=lambda r: str(r.get("ts", "")))
    total = len(records)
    settled = [r for r in records if r.get("result") in ("win", "loss", "push", "void")]
    w = sum(r["result"] == "win" for r in settled)
    l = sum(r["result"] == "loss" for r in settled)
    p = sum(r["result"] == "push" for r in settled)
    v = sum(r["result"] == "void" for r in settled)
    units, cum, peak, max_dd = 0.0, 0.0, 0.0, 0.0
    staked, profit = 0.0, 0.0
    seq = []
    for r in settled:
        d = float(r.get("odds") or 0)
        stake = float(r.get("stake") or 1.0)
        if r["result"] == "win":
            u, prof = d - 1, stake * (d - 1)
            seq.append("W")
        elif r["result"] == "loss":
            u, prof = -1.0, -stake
            seq.append("L")
        else:
            u, prof = 0.0, 0.0
        if r["result"] != "void":
            staked += stake
        units += u
        profit += prof
        cum += u
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    decided = [r for r in settled if r["result"] in ("win", "loss") and r.get("prob") is not None]
    probs = [float(r["prob"]) for r in decided]
    ys = [1 if r["result"] == "win" else 0 for r in decided]
    clv_vals = [float(r["odds"]) / float(r["closing_odds"]) - 1 for r in settled
                if r.get("closing_odds") and r.get("odds")]
    st = streaks(seq)
    bets = w + l + p
    return {
        "total": total, "settled": len(settled), "open": total - len(settled),
        "wins": w, "losses": l, "pushes": p, "voids": v,
        "win_rate": (w / (w + l)) if (w + l) else None,
        "avg_odds": (sum(float(r["odds"]) for r in settled if r.get("odds")) / len(settled)) if settled else None,
        "units": units, "yield": (units / bets) if bets else None,
        "staked": staked, "profit": profit, "roi": (profit / staked) if staked else None,
        "max_drawdown_units": max_dd, "longest_win_streak": st["longest_win"],
        "longest_loss_streak": st["longest_loss"], "current_streak": st["current"],
        "brier": brier_score(probs, ys), "log_loss": log_loss(probs, ys),
        "ece": expected_calibration_error(probs, ys), "calibration": calibration_table(probs, ys),
        "clv": (sum(clv_vals) / len(clv_vals)) if clv_vals else None, "clv_n": len(clv_vals),
    }


def group_summary(records: list[dict], key_fn) -> dict[str, dict]:
    groups: dict[str, list] = defaultdict(list)
    for r in records:
        groups[str(key_fn(r))].append(r)
    return {k: summarize(v) for k, v in sorted(groups.items())}
