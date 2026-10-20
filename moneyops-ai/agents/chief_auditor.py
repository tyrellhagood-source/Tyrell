"""Agent 5 - Chief Auditor & Consensus.

* Recomputes the critical numbers independently (exact rational arithmetic,
  separate code path): no-vig consensus from the raw book prices, implied
  probability, and EV. A mismatch is a veto.
* Re-checks freshness at audit time, completeness and agent disagreements.
* Vetoes overconfident estimates (model far from the market) - five agents
  agreeing does not prove anything if they share a data error.
* Builds the weighted consensus score, applies the mandatory gates, removes
  duplicates/correlated exposure, enforces the exposure cap, ranks singles,
  and builds/audits parlays separately.
"""
from __future__ import annotations

import itertools
import statistics
from fractions import Fraction

from agents.base import AUDIT, PROB, RESEARCH, STATS, VERIFY, AgentReport, Candidate, SportContext
from models.parlay import ParlayLeg, evaluate_parlay
from risk.position_sizing import suggested_stake
from risk.risk_limits import is_stale

WEIGHTS = {"data_quality": 0.12, "verification": 0.15, "statistical_evidence": 0.12, "model_quality": 0.10,
           "expected_value": 0.20, "market_price": 0.10, "historical_validation": 0.06, "risk": 0.10,
           "correlation": 0.05}
NO_QUALIFYING = "NO QUALIFYING BETS FOUND."


def _independent_fair(c: Candidate) -> float | None:
    """Re-derive the consensus no-vig probability with Fractions from raw prices."""
    vals = []
    by_book = {}
    for p in c.event.prices:
        if p.market == c.market:
            by_book.setdefault(p.bookmaker, []).append(p)
    for book in c.fair_by_book:
        outs = by_book.get(book, [])
        mine = [o for o in outs if o.name == c.selection and o.point == c.point and o.description == c.description]
        if not mine:
            continue
        me = mine[0]
        if c.market == "h2h":
            group = outs
        elif c.market == "spreads":
            group = [me] + [o for o in outs if o.name != me.name and o.point is not None and me.point is not None
                            and abs(o.point + me.point) < 1e-9]
        else:
            group = [me] + [o for o in outs if o.name != me.name and o.point == me.point
                            and o.description == me.description]
        inv = [1 / Fraction(str(o.price_decimal)) for o in group]
        idx = next(i for i, o in enumerate(group) if o is me)
        vals.append(float(inv[idx] / sum(inv)))
    return statistics.median(vals) if vals else None


def _independent_ev(p_win: float, p_push: float, d: float) -> float:
    pw, pp, dd = Fraction(str(p_win)), Fraction(str(p_push)), Fraction(str(d))
    return float(pw * (dd - 1) - (1 - pw - pp))


class ChiefAuditor:
    name = AUDIT

    def __init__(self, settings):
        self.s = settings

    # ------------------------------------------------------------ single
    def audit(self, c: Candidate, ctx: SportContext, now) -> AgentReport:
        s = self.s
        r1, r2, r3, r4 = (c.reports.get(a) for a in (RESEARCH, VERIFY, STATS, PROB))
        vetoes, flags, disagreements, recalcs = [], [], [], {}
        # independent recalculation
        if c.fair_prob is not None:
            fair2 = _independent_fair(c)
            recalcs["fair_prob"] = {"agent1": c.fair_prob, "auditor": fair2}
            if fair2 is None or abs(fair2 - c.fair_prob) > 1e-6:
                vetoes.append(f"no-vig consensus mismatch (agent1 {c.fair_prob}, auditor {fair2})")
        p4 = r4.data if r4 else {}
        if p4.get("final_prob") is not None and c.best is not None:
            ev2 = _independent_ev(p4["final_prob"], p4.get("push_prob") or 0.0, c.best.price_decimal)
            recalcs["expected_value"] = {"agent4": p4.get("expected_value"), "auditor": ev2}
            if abs(ev2 - p4["expected_value"]) > 1e-9:
                vetoes.append(f"EV mismatch (agent4 {p4['expected_value']}, auditor {ev2})")
            imp2 = float(1 / Fraction(str(c.best.price_decimal)))
            recalcs["implied_prob"] = {"agent4": p4.get("market_implied_prob"), "auditor": imp2}
            if abs(imp2 - p4["market_implied_prob"]) > 1e-12:
                vetoes.append("implied probability mismatch")
        # freshness at audit time
        if c.best is not None and is_stale(c.best.last_update, now, c.is_live, s):
            vetoes.append("odds became stale before audit")
        # overconfidence
        gap = p4.get("model_minus_market")
        if gap is not None and abs(gap) > s.max_model_market_gap:
            vetoes.append(f"model disagrees with the market by {gap * 100:+.1f} pp (> {s.max_model_market_gap * 100:.0f}); "
                          "most likely missing information (injuries, lineups, news), not a market error")
        if p4.get("final_prob") and p4["final_prob"] > 0.85:
            flags.append("very high probability estimate; check for missing information")
        # disagreements between agents
        if r3 and r4:
            if r3.verdict == "CONTRADICTS" and r4.verdict == "PASS":
                disagreements.append("Statistician's history contradicts the Probability Engineer's positive EV")
            if r3.verdict == "SUPPORTS" and r4.verdict == "FAIL":
                disagreements.append("History supports the side but the price offers no value")
        if r1 and r1.verdict == "COMPLETE" and r2 and r2.verdict != "VERIFIED":
            disagreements.append(f"Researcher reported complete data but Fact-Checker returned {r2.verdict}")
        if r1:
            flags.extend(f"agent1: {f}" for f in r1.flags if f.startswith(("CONFLICT", "OUTLIER")))
        # weighted components
        bt = ctx.backtest or {}
        ev = p4.get("expected_value")
        interval = p4.get("final_prob_interval")
        comp = {
            "data_quality": r1.score if r1 else 0.0,
            "verification": r2.score if r2 else 0.0,
            "statistical_evidence": r3.score if r3 else 0.0,
            "model_quality": float(bt.get("model_quality_score") or 0.0),
            "expected_value": max(0.0, min(1.0, (ev or 0) / 0.08)),
            "market_price": 0.0,
            "historical_validation": max(0.0, 1 - (bt.get("ece") if bt.get("ece") is not None else 0.1) / 0.1),
            "risk": max(0.0, 1 - ((interval[1] - interval[0]) / 0.2)) if interval else 0.0,
            "correlation": 1.0,
        }
        if c.best is not None and c.fair_prob is not None:
            comp["market_price"] = max(0.0, min(1.0, (c.fair_prob * c.best.price_decimal - 1) / 0.05)) * \
                min(1.0, len(c.fair_by_book) / 6)
        score = sum(WEIGHTS[k] * v for k, v in comp.items())
        # mandatory gates
        gates = {
            "fresh_complete_price": bool(r1 and r1.verdict != "STALE" and c.best is not None and c.fair_prob is not None),
            "verified": bool(r2 and r2.verdict == "VERIFIED"),
            "statistics_not_insufficient_or_contradicting": bool(r3 and r3.verdict in ("SUPPORTS", "NEUTRAL")),
            "probability_pass": bool(r4 and r4.verdict == "PASS"),
            "no_auditor_veto": not vetoes,
            "pre_game": not c.is_live,
            "team_market": not c.is_prop,
        }
        if all(gates.values()):
            verdict = "QUALIFIED"
        elif (r2 and r2.verdict == "REJECTED") or (r4 and r4.verdict == "FAIL") or vetoes or \
                (r1 and r1.verdict == "STALE") or (ev is not None and ev <= 0) or ev is None:
            verdict = "REJECTED"
        else:
            verdict = "NEEDS_RESEARCH"
        failed = [k for k, ok in gates.items() if not ok]
        reasons = vetoes + [f"gate failed: {g}" for g in failed]
        c.status, c.consensus_score = verdict, round(score, 4)
        summary = {"QUALIFIED": f"All gates passed; consensus score {score:.3f}",
                   "NEEDS_RESEARCH": "Positive EV but not fully verified: " + ", ".join(failed),
                   "REJECTED": "Rejected: " + "; ".join(reasons[:3])}[verdict]
        return AgentReport(AUDIT, verdict, round(score, 4), summary, reasons=reasons, flags=flags, data={
            "components": comp, "weights": WEIGHTS, "gates": gates, "vetoes": vetoes,
            "disagreements": disagreements, "recalculations": recalcs})

    # ---------------------------------------------------------- portfolio
    def finalize_portfolio(self, cands: list[Candidate], bankroll: float, open_exposure: float) -> None:
        """Duplicates, one pick per game, exposure cap (mutates candidates)."""
        s = self.s
        seen, events_taken = set(), {}
        budget = max(0.0, (bankroll + open_exposure) * s.max_open_exposure_pct - open_exposure)
        for c in sorted(cands, key=lambda x: -x.consensus_score):
            rep = c.reports[AUDIT]
            ident = (c.event.event_id, c.market, c.selection, c.point, c.description)
            if ident in seen:
                c.status = rep.verdict = "REJECTED"
                rep.reasons.append("duplicate recommendation")
                continue
            seen.add(ident)
            if c.status != "QUALIFIED":
                continue
            if c.event.event_id in events_taken:
                c.status = rep.verdict = "NEEDS_RESEARCH"
                rep.reasons.append(f"correlated with higher-ranked pick on the same game ({events_taken[c.event.event_id]})")
                rep.data["components"]["correlation"] = 0.0
                continue
            events_taken[c.event.event_id] = c.label
            if c.stake > budget + 1e-9:
                rep.flags.append(f"exposure cap reached: suggested stake ${c.stake:.2f} exceeds remaining "
                                 f"budget ${budget:.2f}; not stakeable now")
                c.stake = 0.0
            budget -= c.stake

    # ------------------------------------------------------------- parlays
    def same_game_factor(self, a: Candidate, b: Candidate, ctx: SportContext, grid_cache: dict) -> float | None:
        """P_model(a and b) / (P_model(a) * P_model(b)) from the event's score grid."""
        if ctx.model is None or a.is_prop or b.is_prop or a.market == b.market:
            return None
        pa = (a.reports.get(PROB) or AgentReport(PROB, "", 0, "")).data.get("model_prob")
        pb = (b.reports.get(PROB) or AgentReport(PROB, "", 0, "")).data.get("model_prob")
        if not pa or not pb:
            return None
        home, away = ctx.history_name(a.event.home_team), ctx.history_name(a.event.away_team)
        key = a.event.event_id
        if key not in grid_cache:
            grid_cache[key] = ctx.model.score_grid(home, away, a.neutral_site)
        joint = grid_cache[key].joint_probability([(a.market, a.side, a.point), (b.market, b.side, b.point)],
                                                  ctx.cfg.has_draw)
        return None if joint is None else joint / (pa * pb)

    @staticmethod
    def to_leg(c: Candidate) -> ParlayLeg:
        p4 = c.reports[PROB].data
        return ParlayLeg(key=c.key, event_id=c.event.event_id, event_name=c.event.name,
                         teams=(c.event.home_team, c.event.away_team), market=c.market, side=c.side, point=c.point,
                         selection=c.label, decimal=c.best.price_decimal, p_win=p4["final_prob"],
                         sport_code=c.cfg.code, is_prop=c.is_prop,
                         meta={"p_low": p4.get("final_prob_interval", [p4["final_prob"]])[0]})

    def build_parlays(self, qualified: list[Candidate], bankroll: float) -> list[dict]:
        s = self.s
        pool = sorted([c for c in qualified if c.status == "QUALIFIED"], key=lambda c: -c.consensus_score)[:8]
        out = []
        for n in range(2, s.max_parlay_legs + 1):
            for combo in itertools.combinations(pool, n):
                legs = [self.to_leg(c) for c in combo]
                try:
                    ev = evaluate_parlay(legs)
                except ValueError:
                    continue
                reasons = []
                if ev["correlation_status"] == "uncertain":
                    reasons.append("probability uncertain (correlation not estimable)")
                p = ev["probability"]
                if p is not None:
                    p_low = 1.0
                    for l in legs:
                        p_low *= l.meta["p_low"]
                    ev["probability_conservative"] = p_low
                    ev["expected_value_conservative"] = p_low * ev["combined_decimal"] - 1
                    if ev["expected_value"] < s.parlay_min_ev:
                        reasons.append(f"EV {ev['expected_value']:+.3f} below parlay minimum {s.parlay_min_ev:+.3f}")
                    stake = suggested_stake(p, ev["combined_decimal"], bankroll, s.parlay_max_stake_pct,
                                            s.kelly_fraction, s.min_stake, 0.0, s.typical_sportsbook_min_bet)
                    ev["stake"] = stake["stake"]
                    ev["payout_if_win"] = round(stake["stake"] * ev["combined_decimal"], 2)
                    ev["net_profit_if_win"] = round(stake["stake"] * (ev["combined_decimal"] - 1), 2)
                    ev["max_loss"] = stake["stake"]
                    ev["warnings"] += stake["notes"]
                    if stake["stake"] <= 0:
                        reasons.append("no stake after risk sizing")
                ev["status"] = "QUALIFIED" if not reasons else "REJECTED"
                ev["reasons"] = reasons
                ev["score"] = round(statistics.mean(c.consensus_score for c in combo) *
                                    (0.9 ** (n - 2)), 4)
                ev["candidates"] = list(combo)
                out.append(ev)
        out.sort(key=lambda e: (e["status"] != "QUALIFIED", -(e.get("expected_value_conservative") or -9)))
        return out[: max(s.max_parlays * 3, s.max_parlays)]
