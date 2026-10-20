"""Grade recommendations and settle open paper wagers from verified results.

A result is used only when status is FINAL from at least one source and no
source disagrees. Conflicts, missing data and props stay pending and are
reported - never guessed.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from data.results import ResultsService
from data.provider_base import ProviderError
from data.sports_registry import SPORTS
from data.player_stats_provider import PROP_STATS
from database.models import Recommendation
from risk.bankroll import BankrollService
from utils.timeutil import ensure_utc, utcnow

log = logging.getLogger(__name__)


def grade_selection(market: str, side: str, point, home_score: int, away_score: int, has_draw: bool) -> str:
    if market == "totals":
        total = home_score + away_score
        if total == point:
            return "push"
        return "win" if (total > point) == (side == "over") else "loss"
    if side == "draw":
        return "win" if home_score == away_score else "loss"
    mine, theirs = (home_score, away_score) if side == "home" else (away_score, home_score)
    if market == "h2h":
        if mine == theirs:
            return "loss" if has_draw else "push"
        return "win" if mine > theirs else "loss"
    if market == "spreads":
        adj = mine - theirs + point
        return "push" if adj == 0 else ("win" if adj > 0 else "loss")
    raise ValueError(f"cannot auto-grade market {market}")


class SettlementService:
    def __init__(self, settings, repo, espn=None, odds=None, allsports=None, player_stats=None, balldontlie=None):
        self.s, self.repo = settings, repo
        self.results = ResultsService(espn, odds, settings, allsports, balldontlie)
        self.player_stats = player_stats
        self._cache = {}

    def _grade_prop(self, rec: Recommendation) -> tuple[str | None, str]:
        cfg = SPORTS.get(rec.sport_code)
        module = cfg.player_logs if cfg else None
        attrs = PROP_STATS.get(module or "", {}).get(rec.market)
        player = (rec.report_json or {}).get("player")
        if not (module and attrs and player and rec.side in ("over", "under") and rec.point is not None):
            return None, "prop cannot be auto-graded (no box-score mapping); settle manually"
        if self.player_stats is None or not self.player_stats.is_configured(module):
            return None, f"{module} player-stats source disabled; settle props manually"
        statuses = []
        for team in (rec.home_team, rec.away_team):
            try:
                st, value, src = self.player_stats.player_stat_in_game(module, team, ensure_utc(rec.commence_time),
                                                                       player, attrs)
            except ProviderError as exc:
                return None, f"player stats: {exc}"
            if st == "FINAL":
                if value == rec.point:
                    return "push", f"{player} {value:g} via {src}"
                return ("win" if (value > rec.point) == (rec.side == "over") else "loss"), f"{player} {value:g} via {src}"
            statuses.append((st, src))
        if all(st == "DNP" for st, _ in statuses):
            return "void", f"{player} did not play ({'; '.join(s for _, s in statuses)})"
        return None, "; ".join(s for _, s in statuses)

    def _result(self, sport_code, sport_key, event_id, home, away, commence):
        key = event_id
        if key not in self._cache:
            self._cache[key] = self.results.lookup(sport_code, sport_key, event_id, home, away, ensure_utc(commence))
        return self._cache[key]

    def _grade_leg(self, leg: dict, rec_lookup) -> tuple[str | None, str]:
        rec = rec_lookup(leg.get("recommendation_id"))
        if rec is None:
            return None, "leg recommendation missing"
        return self._grade_single(rec)

    def _grade_single(self, rec: Recommendation) -> tuple[str | None, str]:
        if rec.market.startswith("player_"):
            return self._grade_prop(rec)
        cfg = SPORTS.get(rec.sport_code)
        vr = self._result(rec.sport_code, rec.sport_key, rec.event_id, rec.home_team, rec.away_team, rec.commence_time)
        if vr.status == "POSTPONED":
            return "void", "postponed/cancelled per " + ",".join(vr.sources)
        if vr.status != "FINAL":
            return None, f"{vr.status}: {vr.detail}"
        res = grade_selection(rec.market, rec.side, rec.point, vr.home_score, vr.away_score,
                              cfg.has_draw if cfg else False)
        return res, f"final {vr.home_score}-{vr.away_score} via {'+'.join(vr.sources)}"

    def grade_recommendations(self, now=None) -> dict:
        now = now or utcnow()
        done, pending = 0, 0
        recs = {r.id: r for r in self.repo.list_recommendations(pending_only=True)}
        for rec in sorted(recs.values(), key=lambda r: r.kind != "single"):
            if rec.commence_time is None or ensure_utc(rec.commence_time) > now - timedelta(hours=2):
                continue
            if rec.kind == "single":
                res, src = self._grade_single(rec)
                closing = self.repo.closing_price(rec.event_id, rec.market, _outcome_name(rec), rec.point,
                                                  None, ensure_utc(rec.commence_time)) if res else None
            else:
                leg_res = []
                src = []
                for leg in rec.legs_json:
                    r, s_ = self._grade_leg(leg, self.repo.get_recommendation)
                    leg_res.append(r)
                    src.append(s_)
                if any(r == "loss" for r in leg_res):
                    res = "loss"
                elif any(r is None for r in leg_res):
                    res = None
                elif all(r in ("push", "void") for r in leg_res):
                    res = "void"
                else:
                    res = "win"
                src, closing = "; ".join(src), None
            if res is None:
                pending += 1
                continue
            self.repo.grade_recommendation(rec.id, res, src, closing)
            done += 1
        return {"graded": done, "still_pending": pending}

    def settle_open_wagers(self, now=None) -> dict:
        now = now or utcnow()
        bank = BankrollService(self.repo, self.s)
        settled, pending, notes = 0, 0, []
        for w in self.repo.wagers(status="OPEN"):
            rec = self.repo.get_recommendation(w.recommendation_id)
            if rec is None or rec.commence_time is None or ensure_utc(rec.commence_time) > now - timedelta(hours=2):
                pending += 1
                continue
            if w.kind == "single":
                res, src = self._grade_single(rec)
                if res is None:
                    pending += 1
                    notes.append(f"wager {w.id}: {src}")
                    continue
                closing = self.repo.closing_price(rec.event_id, rec.market, _outcome_name(rec), rec.point,
                                                  None, ensure_utc(rec.commence_time))
                bank.settle(w.id, res, src, closing)
            else:
                leg_res, srcs = [], []
                for leg in w.legs_json:
                    r, s_ = self._grade_leg(leg, self.repo.get_recommendation)
                    leg_res.append(r)
                    srcs.append(s_)
                if any(r is None for r in leg_res) and not any(r == "loss" for r in leg_res):
                    pending += 1
                    notes.append(f"parlay {w.id}: legs pending")
                    continue
                leg_res = [r or "loss" for r in leg_res] if any(r == "loss" for r in leg_res) else leg_res
                bank.settle(w.id, "loss", "; ".join(srcs), None, leg_results=leg_res)
            settled += 1
        return {"settled": settled, "pending": pending, "notes": notes}


def _outcome_name(rec: Recommendation) -> str:
    if rec.side == "home":
        return rec.home_team
    if rec.side == "away":
        return rec.away_team
    return {"draw": "Draw", "over": "Over", "under": "Under"}.get(rec.side, rec.selection)
