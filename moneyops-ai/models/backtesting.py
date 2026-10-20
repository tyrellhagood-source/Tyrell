"""Time-aware model evaluation (HYPOTHETICAL backtests, kept separate from
paper-trading results).

Split (chronological, never shuffled):
  * first 60% of games  -> training only (warm-up)
  * next 20%            -> validation: choose the regularisation strength
  * final 20%           -> test: walk-forward, refit weekly on all games
                           strictly before each test week, predict that week.

Target: home team wins (final score). Only moneyline-style probabilities are
backtested because free historical *odds* are not available; therefore this
measures probability quality (Brier, log loss, calibration), NOT profitability.
"""
from __future__ import annotations

from datetime import timedelta

from data.schemas import HistoricalGame
from data.sports_registry import SportConfig
from models.calibration import (brier_score, brier_skill_score, calibration_table,
                                expected_calibration_error, log_loss)
from models.probability import ALPHA_GRID, TeamRatingModel
from utils.timeutil import ensure_utc


def _home_win_prob(model: TeamRatingModel, g: HistoricalGame) -> float | None:
    if not (model.is_rated(g.home_team) and model.is_rated(g.away_team)):
        return None
    r = model.probability("h2h", "home", None, g.home_team, g.away_team, g.neutral_site)
    return None if r is None else r[0]


def chronological_split(games: list[HistoricalGame], train=0.6, val=0.2):
    games = sorted(games, key=lambda g: g.date)
    n = len(games)
    a, b = int(n * train), int(n * (train + val))
    return games[:a], games[a:b], games[b:]


def walk_forward(cfg: SportConfig, games: list[HistoricalGame], half_life_days=120.0,
                 min_games_per_team=5, block_days=7) -> dict:
    games = [g for g in sorted(games, key=lambda g: g.date)
             if not (cfg.ties_resolved and g.home_score == g.away_score)]
    if len(games) < 80:
        return {"status": "INSUFFICIENT_DATA", "detail": f"{len(games)} games; need >= 80", "n_games": len(games)}
    train, val, test = chronological_split(games)
    # ---- validation: pick alpha -------------------------------------------
    alpha_scores = {}
    for alpha in ALPHA_GRID[cfg.model]:
        m = TeamRatingModel(cfg, alpha, half_life_days, min_games_per_team).fit(train, val[0].date)
        ps, ys = [], []
        for g in val:
            p = _home_win_prob(m, g)
            if p is not None:
                ps.append(p); ys.append(int(g.home_score > g.away_score))
        alpha_scores[alpha] = log_loss(ps, ys) if ps else float("inf")
    best_alpha = min(alpha_scores, key=alpha_scores.get)
    # ---- test: weekly walk-forward ------------------------------------------
    preds = []
    start = ensure_utc(test[0].date)
    end = ensure_utc(test[-1].date)
    cursor = start
    while cursor <= end:
        block_end = cursor + timedelta(days=block_days)
        block = [g for g in test if cursor <= ensure_utc(g.date) < block_end]
        if block:
            m = TeamRatingModel(cfg, best_alpha, half_life_days, min_games_per_team).fit(games, cursor)
            for g in block:
                p = _home_win_prob(m, g)
                if p is not None:
                    preds.append({"date": g.date.isoformat(), "home": g.home_team, "away": g.away_team,
                                  "p_home": p, "home_won": int(g.home_score > g.away_score)})
        cursor = block_end
    if len(preds) < 20:
        return {"status": "INSUFFICIENT_DATA", "detail": f"only {len(preds)} test predictions", "n_games": len(games)}
    ps = [p["p_home"] for p in preds]
    ys = [p["home_won"] for p in preds]
    base_rate = sum(int(g.home_score > g.away_score) for g in train + val) / len(train + val)
    model_brier = brier_score(ps, ys)
    base_brier = brier_score([base_rate] * len(ys), ys)
    bss = brier_skill_score(model_brier, base_brier)
    ece = expected_calibration_error(ps, ys)
    quality = max(0.0, min(1.0, (bss or 0) / 0.10))
    if ece is not None and ece > 0.08:
        quality *= 0.5
    return {
        "status": "OK", "label": "HYPOTHETICAL BACKTEST (probability quality only, no odds)",
        "sport": cfg.code, "model": cfg.model, "alpha": best_alpha, "alpha_validation_logloss": alpha_scores,
        "n_train": len(train), "n_validation": len(val), "n_test": len(test), "n_predictions": len(preds),
        "train_period": [train[0].date.isoformat(), train[-1].date.isoformat()],
        "validation_period": [val[0].date.isoformat(), val[-1].date.isoformat()],
        "test_period": [test[0].date.isoformat(), test[-1].date.isoformat()],
        "brier": model_brier, "baseline_brier": base_brier, "brier_skill_score": bss,
        "log_loss": log_loss(ps, ys), "baseline_log_loss": log_loss([base_rate] * len(ys), ys),
        "ece": ece, "calibration": calibration_table(ps, ys), "model_quality_score": quality,
        "predictions": preds,
    }
