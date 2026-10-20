"""Team-strength probability models.

Two model families, chosen per sport (see data/sports_registry.py):

* ``gaussian`` (NFL, NCAAF, NBA, NCAAB, NCAAW, WNBA): a ridge regression
  ``points = mu + offence[team] + defence[opponent] + home_adv`` fitted with
  exponential recency weights. Home/away scores are modelled as a bivariate
  normal (covariance from training residuals, inflated for degrees of
  freedom) discretised onto an integer score grid, so spreads/totals get
  correct push probabilities on whole-number lines.
* ``poisson`` (MLB, NHL, soccer): Poisson GLM on the same design (log link);
  home and away goals independent Poisson. For NHL/MLB tied regulation scores
  are resolved by moving half of each tie to "home +1" and half to "away +1"
  (shootout / extra-innings approximation).

Every market (moneyline, spread, total, and same-game combinations) is then
priced by summing the same joint score grid, which also gives exact joint
probabilities for same-game parlays under the model.

Only games strictly before ``as_of`` are used for fitting (no leakage).
Known limitations are documented in README.md (no player-level adjustments,
no Dixon-Coles low-score correction, OT approximations, key numbers in NFL
captured only through the normal approximation).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor, Ridge

from data.schemas import HistoricalGame
from data.sports_registry import SportConfig
from utils.timeutil import ensure_utc

DEFAULT_ALPHA = {"gaussian": 5.0, "poisson": 0.002}
ALPHA_GRID = {"gaussian": [1.0, 5.0, 20.0], "poisson": [0.0005, 0.002, 0.01]}


@dataclass
class ScoreGrid:
    home_scores: np.ndarray   # shape (n,)
    away_scores: np.ndarray   # shape (m,)
    prob: np.ndarray          # shape (n, m), sums to 1
    ties_split: bool          # moneyline ties count half (OT approx) for gaussian

    def _mats(self):
        h = self.home_scores[:, None].astype(float)
        a = self.away_scores[None, :].astype(float)
        return h, a

    def leg_weights(self, market: str, side: str, point: float | None, has_draw: bool):
        """Return (win_weight, push_weight) matrices, or None if unsupported."""
        h, a = self._mats()
        shape = self.prob.shape
        zeros = np.zeros(shape)
        if market == "h2h":
            if side == "draw":
                return (h == a).astype(float) + zeros, zeros
            mine, theirs = (h, a) if side == "home" else (a, h)
            win = (mine > theirs).astype(float) + zeros
            if self.ties_split and not has_draw:
                win = win + 0.5 * (h == a)
            return win, zeros
        if point is None or (point * 2) % 1 != 0:
            return None  # quarter (Asian) lines are not supported
        if market == "spreads":
            margin = (h - a) if side == "home" else (a - h)
            adj = margin + point
            return (adj > 0).astype(float) + zeros, (adj == 0).astype(float) + zeros
        if market == "totals":
            total = h + a + zeros
            if side == "over":
                return (total > point).astype(float), (total == point).astype(float)
            return (total < point).astype(float), (total == point).astype(float)
        return None

    def probability(self, market, side, point, has_draw=False) -> tuple[float, float] | None:
        w = self.leg_weights(market, side, point, has_draw)
        if w is None:
            return None
        win, push = w
        return float((self.prob * win).sum()), float((self.prob * push).sum())

    def joint_probability(self, legs, has_draw=False) -> float | None:
        """P(all legs win). Pushes count as not-winning (conservative)."""
        weight = np.ones(self.prob.shape)
        for market, side, point in legs:
            w = self.leg_weights(market, side, point, has_draw)
            if w is None:
                return None
            weight = weight * w[0]
        return float((self.prob * weight).sum())


class TeamRatingModel:
    def __init__(self, cfg: SportConfig, alpha: float | None = None, half_life_days: float | None = 120.0,
                 min_games_per_team: int = 5):
        if cfg.model not in ("gaussian", "poisson"):
            raise ValueError(f"{cfg.code} has no statistical model")
        self.cfg = cfg
        self.kind = cfg.model
        self.alpha = alpha if alpha is not None else DEFAULT_ALPHA[self.kind]
        self.half_life_days = half_life_days
        self.min_games_per_team = min_games_per_team
        self.fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, games: list[HistoricalGame], as_of: datetime) -> "TeamRatingModel":
        as_of = ensure_utc(as_of)
        train = [g for g in games if ensure_utc(g.date) < as_of]
        if len(train) < 10:
            raise ValueError(f"only {len(train)} completed games before {as_of:%Y-%m-%d}; need >= 10")
        teams = sorted({g.home_team for g in train} | {g.away_team for g in train})
        self.team_index = {t: i for i, t in enumerate(teams)}
        T = len(teams)
        X = np.zeros((2 * len(train), 2 * T + 1))
        y = np.zeros(2 * len(train))
        w = np.ones(2 * len(train))
        self.games_count = {t: 0 for t in teams}
        for k, g in enumerate(train):
            hi, ai = self.team_index[g.home_team], self.team_index[g.away_team]
            home_flag = 0.0 if g.neutral_site else 1.0
            X[2 * k, hi] = 1; X[2 * k, T + ai] = 1; X[2 * k, 2 * T] = home_flag
            X[2 * k + 1, ai] = 1; X[2 * k + 1, T + hi] = 1
            y[2 * k], y[2 * k + 1] = g.home_score, g.away_score
            if self.half_life_days:
                age = (as_of - ensure_utc(g.date)).total_seconds() / 86400
                w[2 * k] = w[2 * k + 1] = 0.5 ** (age / self.half_life_days)
            self.games_count[g.home_team] += 1
            self.games_count[g.away_team] += 1
        if self.kind == "gaussian":
            reg = Ridge(alpha=self.alpha).fit(X, y, sample_weight=w)
        else:
            reg = PoissonRegressor(alpha=self.alpha, max_iter=1000).fit(X, y, sample_weight=w)
        self.coef = reg.coef_
        self.intercept = float(reg.intercept_)
        self.n_teams = T
        self.n_games = len(train)
        self.trained_through = max(ensure_utc(g.date) for g in train)
        self.as_of = as_of
        if self.kind == "gaussian":
            linear = X @ self.coef + self.intercept
            resid = (y - linear).reshape(-1, 2)
            gw = w[::2]
            mean = np.average(resid, axis=0, weights=gw)
            centred = resid - mean
            cov = (centred * gw[:, None]).T @ centred / gw.sum()
            n_obs, n_par = 2 * len(train), 2 * T + 1
            inflation = max(1.0, n_obs / max(1.0, n_obs - n_par))
            self.cov = cov * inflation
        else:
            self.cov = None
        self._train_games = train
        self.fitted = True
        return self

    # ------------------------------------------------------------ queries
    def is_rated(self, team: str) -> bool:
        return self.fitted and self.games_count.get(team, 0) >= self.min_games_per_team

    def team_rating(self, team: str) -> float | None:
        i = self.team_index.get(team)
        if i is None:
            return None
        return float(self.coef[i] - self.coef[self.n_teams + i])

    def expected_scores(self, home: str, away: str, neutral: bool = False) -> tuple[float, float]:
        hi, ai = self.team_index[home], self.team_index[away]
        T = self.n_teams
        hfa = 0.0 if neutral else self.coef[2 * T]
        lin_h = self.intercept + self.coef[hi] + self.coef[T + ai] + hfa
        lin_a = self.intercept + self.coef[ai] + self.coef[T + hi]
        if self.kind == "poisson":
            return float(np.exp(lin_h)), float(np.exp(lin_a))
        return max(0.1, float(lin_h)), max(0.1, float(lin_a))

    def score_grid(self, home: str, away: str, neutral: bool = False) -> ScoreGrid:
        mu_h, mu_a = self.expected_scores(home, away, neutral)
        if self.kind == "poisson":
            n = self.cfg.max_score + 2
            hs, as_ = np.arange(n), np.arange(n)
            p = np.outer(poisson.pmf(hs, mu_h), poisson.pmf(as_, mu_a))
            if self.cfg.tie_adds_one:
                diag = np.diag(p).copy()
                p[np.arange(n), np.arange(n)] = 0
                idx = np.arange(n - 1)
                p[idx + 1, idx] += 0.5 * diag[:-1]
                p[idx, idx + 1] += 0.5 * diag[:-1]
            p /= p.sum()
            return ScoreGrid(hs, as_, p, ties_split=False)
        sd_h, sd_a = np.sqrt(self.cov[0, 0]), np.sqrt(self.cov[1, 1])
        hs = np.arange(max(0, int(mu_h - 5 * sd_h)), int(mu_h + 5 * sd_h) + 2)
        as_ = np.arange(max(0, int(mu_a - 5 * sd_a)), int(mu_a + 5 * sd_a) + 2)
        inv = np.linalg.inv(self.cov)
        dh = hs[:, None] - mu_h
        da = as_[None, :] - mu_a
        q = inv[0, 0] * dh ** 2 + 2 * inv[0, 1] * dh * da + inv[1, 1] * da ** 2
        p = np.exp(-0.5 * q)
        p /= p.sum()
        return ScoreGrid(hs, as_, p, ties_split=self.cfg.ties_resolved)

    def probability(self, market, side, point, home, away, neutral=False):
        return self.score_grid(home, away, neutral).probability(market, side, point, self.cfg.has_draw)

    def joint_probability(self, legs, home, away, neutral=False):
        return self.score_grid(home, away, neutral).joint_probability(legs, self.cfg.has_draw)

    def bootstrap(self, n: int, seed: int) -> list["TeamRatingModel"]:
        """Refit on resampled training games -> parameter uncertainty."""
        rng = np.random.default_rng(seed)
        out = []
        games = self._train_games
        for _ in range(n):
            sample = [games[i] for i in rng.integers(0, len(games), len(games))]
            m = TeamRatingModel(self.cfg, self.alpha, self.half_life_days, self.min_games_per_team)
            try:
                out.append(m.fit(sample, self.as_of))
            except (ValueError, np.linalg.LinAlgError):
                continue
        return out
