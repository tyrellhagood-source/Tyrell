"""Read-side helpers shared by the dashboard and the REST API."""
from __future__ import annotations

from agents.base import AGENT_NAMES, ENDORSE
from database.models import Recommendation
from models.parlay import ParlayLeg, evaluate_parlay
from models.performance import ODDS_BUCKETS, PROB_BUCKETS, bucket_label, group_summary, summarize


def rec_record(r: Recommendation) -> dict:
    return {"id": r.id, "ts": r.created_at.isoformat() if r.created_at else "", "result": r.result,
            "odds": r.odds_decimal, "prob": r.final_prob, "stake": 1.0, "closing_odds": r.closing_odds,
            "sport": r.sport_code, "league": r.league, "market": r.market, "kind": r.kind, "status": r.status,
            "model_version": r.model_version, "verdicts": r.verdicts_json or {}}


def performance_report(repo) -> dict:
    """Research performance of RECOMMENDATIONS (flat 1-unit, hypothetical
    tracking of every pick incl. rejected). Paper-wager results are reported
    separately by wager_performance()."""
    recs = [rec_record(r) for r in repo.list_recommendations()]
    consensus = [r for r in recs if r["status"] == "QUALIFIED"]
    out = {
        "all_recommendations": summarize(recs),
        "consensus_picks": summarize(consensus),
        "singles": summarize([r for r in consensus if r["kind"] == "single"]),
        "parlays": summarize([r for r in consensus if r["kind"] == "parlay"]),
        "by_status": group_summary(recs, lambda r: r["status"]),
        "by_sport": group_summary(consensus, lambda r: r["sport"]),
        "by_league": group_summary(consensus, lambda r: r["league"]),
        "by_market": group_summary(consensus, lambda r: r["market"]),
        "by_probability": group_summary(consensus, lambda r: bucket_label(r["prob"], PROB_BUCKETS)),
        "by_odds": group_summary(consensus, lambda r: bucket_label(r["odds"], ODDS_BUCKETS)),
        "by_model_version": group_summary(consensus, lambda r: r["model_version"]),
        "by_agent": {},
    }
    singles = [r for r in recs if r["kind"] == "single"]
    for agent, endorse in ENDORSE.items():
        endorsed = [r for r in singles if (r["verdicts"].get(agent) or {}).get("verdict") == endorse]
        out["by_agent"][AGENT_NAMES[agent]] = summarize(endorsed)
    return out


def wager_performance(repo, account_id: int | None = None) -> dict:
    ws = repo.wagers(account_id)
    recs = [{"ts": w.placed_at.isoformat(), "result": w.result if w.status == "SETTLED" else "pending",
             "odds": w.odds_decimal, "prob": w.est_prob, "stake": w.stake, "closing_odds": w.closing_odds}
            for w in ws]
    return summarize(recs)


def custom_parlay(repo, rec_ids: list[int], stake: float = 1.0) -> dict:
    """Parlay Lab: evaluate any combination of stored single recommendations.
    Two legs on the same game use the model's stored same-game correlation
    factor when available; otherwise the probability is reported as uncertain."""
    recs = [repo.get_recommendation(i) for i in rec_ids]
    if any(r is None or r.kind != "single" for r in recs):
        raise ValueError("all legs must be stored single recommendations")
    if any(r.odds_decimal is None or r.final_prob is None for r in recs):
        raise ValueError("every leg needs a price and a probability estimate")
    legs = [ParlayLeg(key=r.candidate_key, event_id=r.event_id, event_name=r.event_name,
                      teams=(r.home_team, r.away_team), market=r.market, side=r.side, point=r.point,
                      selection=r.selection, decimal=r.odds_decimal, p_win=r.final_prob, sport_code=r.sport_code,
                      is_prop=r.market.startswith("player_")) for r in recs]
    by_key = {r.candidate_key: r for r in recs}

    def joint(event_id, group):
        if len(group) != 2:
            return None
        a, b = by_key[group[0].key], by_key[group[1].key]
        factor = ((a.report_json or {}).get("same_game_factors") or {}).get(b.candidate_key)
        if factor is None:
            return None
        return min(1.0, a.final_prob * b.final_prob * factor)

    res = evaluate_parlay(legs, stake, joint)
    res["legs"] = [{"selection": l.selection, "event": l.event_name, "market": l.market, "decimal": l.decimal,
                    "p_win": l.p_win} for l in legs]
    return res
