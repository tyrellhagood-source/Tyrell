"""Player-prop statistics and probability model.

Data: per-game player lines (sportsipy box scores). Games where the player
did not appear (or logged 0 minutes in the NBA) are excluded - a prop on a
player who does not play is void, not a loss.

Model (documented "player performance distribution"):
  * recency-weighted mean and standard deviation of the stat over the last
    up-to-20 appearances (weights halve every 10 games);
  * stat ~ Normal(mean, sd) discretised to integers (continuity correction),
    so whole-number lines get a push probability;
  * uncertainty: bootstrap over the player's games (resample, recompute).
Requires >= MIN_GAMES appearances. No adjustment for opponent, pace, minutes
changes or injuries to teammates (listed as caveats).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from models.stats import hit_rate, streaks
from utils.teams import normalize_team

MIN_GAMES = 8


def player_values(lines: list[dict], player: str, attrs: tuple, module: str) -> list[dict]:
    """[{date, value, home, opponent, minutes}] sorted by date for appearances."""
    target = normalize_team(player)
    out = []
    for l in lines:
        if normalize_team(l["player"]) != target:
            continue
        if module == "nba" and not l.get("minutes"):
            continue
        vals = [l["stats"].get(a) for a in attrs]
        if any(v is None for v in vals):
            continue
        out.append({"date": l["date"], "value": float(sum(vals)), "home": l["home"], "opponent": l["opponent"],
                    "minutes": l.get("minutes"), "game_id": l["game_id"]})
    dedup = {r["game_id"]: r for r in out}
    return sorted(dedup.values(), key=lambda r: r["date"])


def outcome(value: float, side: str, point: float) -> str:
    if value == point:
        return "P"
    return "W" if (value > point) == (side == "over") else "L"


def _weights(n: int, half_life: float = 10.0) -> np.ndarray:
    ages = np.arange(n)[::-1]          # 0 = most recent
    return 0.5 ** (ages / half_life)


def distribution_probability(values: list[float], side: str, point: float) -> tuple[float, float] | None:
    """(p_win, p_push) from a weighted normal fitted to the last 20 values."""
    v = np.array(values[-20:], dtype=float)
    if len(v) < MIN_GAMES or (point * 2) % 1 != 0:
        return None
    w = _weights(len(v))
    mu = float(np.average(v, weights=w))
    sd = float(np.sqrt(np.average((v - mu) ** 2, weights=w)))
    sd = max(sd * np.sqrt(len(v) / (len(v) - 1)), 0.5)
    if point % 1 == 0:      # whole line: push possible
        p_over = 1 - norm.cdf((point + 0.5 - mu) / sd)
        p_push = norm.cdf((point + 0.5 - mu) / sd) - norm.cdf((point - 0.5 - mu) / sd)
    else:
        p_over, p_push = 1 - norm.cdf((point - mu) / sd), 0.0
    p_under = 1 - p_over - p_push
    return (float(p_over) if side == "over" else float(p_under)), float(p_push)


def bootstrap_probabilities(values: list[float], side: str, point: float, n: int = 200, seed: int = 7) -> list[float]:
    """P(win) recomputed on resampled game logs (order kept for recency weights)."""
    rng = np.random.default_rng(seed)
    v = np.array(values[-20:])
    ps = []
    for _ in range(n):
        sample = list(v[np.sort(rng.integers(0, len(v), len(v)))])
        r = distribution_probability(sample, side, point)
        if r:
            ps.append(r[0])
    return ps


def bootstrap_interval(values: list[float], side: str, point: float, n: int = 200, seed: int = 7):
    ps = bootstrap_probabilities(values, side, point, n, seed)
    if len(ps) < 20:
        return None
    lo, hi = np.percentile(ps, [5, 95])
    return float(lo), float(hi)


def prop_history_report(rows: list[dict], side: str, point: float, home_game: bool, opponent: str) -> dict:
    series = [outcome(r["value"], side, point) for r in rows]
    windows = {}
    for label, k in (("last_5", 5), ("last_10", 10), ("last_20", 20), ("all_logged", len(rows))):
        windows[label] = hit_rate(series[-k:]) | {"n_games": min(k, len(rows))}
    same_venue = [outcome(r["value"], side, point) for r in rows if r["home"] == home_game]
    vs_opp = [outcome(r["value"], side, point) for r in rows if normalize_team(r["opponent"]) == normalize_team(opponent)]
    mins = [r["minutes"] for r in rows if r.get("minutes")]
    return {"windows": windows,
            "splits": {"same_venue_role": hit_rate(same_venue) | {"role": "home" if home_game else "away"},
                       "vs_this_opponent": hit_rate(vs_opp)},
            "workload": {"avg_minutes_last5": float(np.mean(mins[-5:])) if mins else None,
                         "avg_minutes_all": float(np.mean(mins)) if mins else None},
            "recent_values": [r["value"] for r in rows[-10:]],
            "average": float(np.mean([r["value"] for r in rows])) if rows else None,
            "streaks": streaks(series), "last_game": rows[-1]["date"] if rows else None}
