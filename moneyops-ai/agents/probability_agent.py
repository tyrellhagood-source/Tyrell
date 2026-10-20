"""Agent 4 - Probability & Risk Engineer.

Probability = documented blend of
  * the sport's team-rating model (models/probability.py) and
  * the multi-book no-vig consensus (the market's own estimate):
      p_final = w * p_model + (1 - w) * p_market        (w = MODEL_WEIGHT, default 0.35)
Shrinking toward the market is deliberate: the model sees no injuries,
lineups or news, so large disagreements are far more often model error than
market error.

Uncertainty: the model is refit on bootstrap resamples of its training
games; the 5th-95th percentile of the resulting probabilities (blended the
same way) gives the interval. EV is reported at the point estimate and at
the conservative (lower) end.
"""
from __future__ import annotations

import numpy as np

from agents.base import PROB, AgentReport, Candidate, SportContext
from models.expected_value import expected_value, implied_probability
from risk.position_sizing import suggested_stake


class ProbabilityAgent:
    name = PROB

    def __init__(self, settings):
        self.s = settings

    def _model_probs(self, c: Candidate, ctx: SportContext):
        model = ctx.model
        if model is None or c.is_prop or c.side not in ("home", "away", "draw", "over", "under"):
            return None, None, "no statistical model for this market"
        home, away = ctx.history_name(c.event.home_team), ctx.history_name(c.event.away_team)
        if not home or not away or not (model.is_rated(home) and model.is_rated(away)):
            return None, None, "one or both teams have too few rated games"
        r = model.probability(c.market, c.side, c.point, home, away, c.neutral_site)
        if r is None:
            return None, None, f"line {c.point} not supported (quarter/Asian line)"
        boots = []
        for m in ctx.bootstrap_models:
            if m.is_rated(home) and m.is_rated(away):
                b = m.probability(c.market, c.side, c.point, home, away, c.neutral_site)
                if b is not None:
                    boots.append(b[0])
        return r, boots, None

    def evaluate(self, c: Candidate, ctx: SportContext, bankroll: float) -> AgentReport:
        s = self.s
        if c.best is None:
            return AgentReport(PROB, "INSUFFICIENT_DATA", 0.0, "No fresh price to evaluate",
                               reasons=["no fresh price"])
        d = c.best.price_decimal
        implied = implied_probability(d)
        model_r, boots, why_no_model = self._model_probs(c, ctx)
        fair = c.fair_prob
        data = {"decimal_odds": d, "market_implied_prob": implied, "fair_prob_no_vig": fair,
                "model_weight": s.model_weight, "method": None}
        reasons, flags = [], []
        if model_r is None:
            data["method"] = "market consensus only"
            reasons.append(f"No independent model estimate: {why_no_model}")
            if fair is None:
                return AgentReport(PROB, "INSUFFICIENT_DATA", 0.0, "No model and no complete market",
                                   reasons=reasons, data=data)
            ev = expected_value(fair, d)
            data |= {"final_prob": fair, "push_prob": 0.0, "expected_value": ev, "ev_per_dollar": ev,
                     "model_prob": None}
            return AgentReport(PROB, "INSUFFICIENT_DATA", 0.0,
                               f"Market-only EV {ev:+.3f}/$ (line shopping); not enough for approval",
                               reasons=reasons, data=data)
        p_model, p_push = model_r
        market = fair if fair is not None else implied
        w = s.model_weight
        decided = 1 - p_push
        p_final = w * p_model + (1 - w) * market * decided
        if boots and len(boots) >= 5:
            lo_m, hi_m = np.percentile(boots, [5, 95])
        else:
            lo_m = hi_m = p_model
            flags.append("bootstrap uncertainty unavailable; interval understates uncertainty")
        p_low = w * lo_m + (1 - w) * market * decided
        p_high = w * hi_m + (1 - w) * market * decided
        ev = expected_value(p_final, d, p_push)
        ev_low = expected_value(min(p_low, 1 - p_push), d, p_push)
        stake = suggested_stake(p_final, d, bankroll, s.max_stake_pct, s.kelly_fraction, s.min_stake, p_push,
                                s.typical_sportsbook_min_bet)
        c.stake = stake["stake"]
        gap = p_model - market * decided
        data |= {"method": f"{ctx.cfg.model} team-rating model blended with no-vig consensus",
                 "model_prob": p_model, "model_prob_interval": [float(lo_m), float(hi_m)],
                 "bootstrap_n": len(boots or []), "push_prob": p_push, "final_prob": p_final,
                 "final_prob_interval": [float(p_low), float(p_high)], "model_minus_market": gap,
                 "expected_value": ev, "expected_value_low": ev_low, "expected_roi_pct": ev * 100,
                 "break_even_prob": implied, "stake": stake, "potential_profit": round(stake["stake"] * (d - 1), 2),
                 "max_loss": stake["stake"]}
        if ctx.model is not None:
            mu = ctx.model.expected_scores(ctx.history_name(c.event.home_team), ctx.history_name(c.event.away_team),
                                           c.neutral_site)
            data["expected_score"] = {"home": round(mu[0], 2), "away": round(mu[1], 2)}
        if p_final >= 0.65 and ev < 0:
            flags.append(f"High win probability ({p_final:.0%}) but negative EV: the price is too short")
        flags.extend(stake["notes"])
        if ev < s.min_ev:
            verdict = "FAIL"
            reasons.append(f"EV {ev:+.3f}/$ below minimum {s.min_ev:+.3f}")
        elif ev_low < s.min_ev_lower:
            verdict = "FAIL"
            reasons.append(f"EV at conservative probability {ev_low:+.3f}/$ below {s.min_ev_lower:+.3f}: edge "
                           "is within model uncertainty")
        elif stake["stake"] <= 0:
            verdict = "FAIL"
            reasons.append(f"no stake after risk sizing ({stake['capped_by']})")
        else:
            verdict = "PASS"
            reasons.append(f"EV {ev:+.3f}/$ (conservative {ev_low:+.3f}) at {d:.2f}; p={p_final:.3f} vs "
                           f"break-even {implied:.3f}")
        score = max(0.0, min(1.0, ev / 0.08)) if verdict == "PASS" else 0.0
        return AgentReport(PROB, verdict, round(score, 3), reasons[-1], reasons=reasons, flags=flags, data=data)
