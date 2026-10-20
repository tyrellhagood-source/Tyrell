"""Agent 2 - Independent Fact-Checker.

Re-derives the critical facts from a source other than the odds feed:
* the game exists on an independent schedule (ESPN) with the same teams,
* the start time agrees, and the game has not already finished/postponed,
* the price is fresh and quoted by enough independent books,
* the best price is not an outlier versus the multi-book consensus,
* historical statistics are dated before the game (no leakage) and recent
  enough to belong to the current season,
* injury information was actually retrieved.

Verdicts: VERIFIED, REJECTED (contradicting evidence) or INSUFFICIENT_DATA
(a critical fact could not be checked). Only VERIFIED can be fully approved.
"""
from __future__ import annotations

from datetime import timedelta

from agents.base import VERIFY, AgentReport, Candidate, SportContext
from data.provider_base import ProviderError
from data.schedules import fetch_schedule, find_scheduled_game
from risk.risk_limits import is_stale
from utils.timeutil import ensure_utc


class VerificationAgent:
    name = VERIFY

    def __init__(self, settings, espn):
        self.s = settings
        self.espn = espn

    def prepare(self, ctx: SportContext, events) -> None:
        """Fetch the independent schedule for this sport (own request, not
        reusing Agent 1's data)."""
        if not events:
            return
        if not ctx.cfg.espn_path or self.espn is None:
            ctx.schedule_error = f"no independent schedule source for {ctx.cfg.code}"
            return
        start = min(e.commence_time for e in events)
        end = max(e.commence_time for e in events)
        try:
            ctx.schedule = fetch_schedule(self.espn, ctx.cfg, start, end)
        except ProviderError as exc:
            ctx.schedule_error = str(exc)

    def verify(self, c: Candidate, ctx: SportContext, now) -> AgentReport:
        s = self.s
        checks, contradictions, unverifiable, flags = [], [], [], []

        def check(name, ok, critical, detail, contradiction=False):
            checks.append({"check": name, "passed": ok, "critical": critical, "detail": detail})
            if not ok and critical:
                (contradictions if contradiction else unverifiable).append(f"{name}: {detail}")
            elif not ok:
                flags.append(f"{name}: {detail}")

        # 1-3: independent schedule
        if ctx.schedule is None:
            check("independent_schedule", False, True, ctx.schedule_error or "schedule not retrieved")
        else:
            game, detail = find_scheduled_game(c.event, ctx.schedule)
            if game is None:
                check("independent_schedule", False, True, "game not found on independent schedule")
            else:
                check("independent_schedule", True, True, f"matched ESPN game {game.game_id}")
                c.neutral_site = game.neutral_site
                check("home_away_orientation", not detail["home_away_swapped"], True,
                      "home/away teams reversed between sources" if detail["home_away_swapped"] else "consistent",
                      contradiction=True)
                diff = detail["start_diff_minutes"]
                check("start_time", diff <= s.max_start_time_diff_minutes, True,
                      f"sources differ by {diff:.0f} min", contradiction=True)
                status = game.status_detail.lower()
                bad = game.completed or "postpon" in status or "cancel" in status or "suspend" in status
                check("game_status", not bad, True, f"ESPN status: {game.status_detail or game.state}",
                      contradiction=True)
                if game.state == "in":
                    flags.append("game in progress per ESPN")
        # 4: freshness
        check("odds_fresh", c.best is not None, True,
              "best price is fresh" if c.best else "no fresh price within the window")
        # 5: multiple independent books
        n_books = len(c.fair_by_book)
        check("multi_book_confirmation", n_books >= s.min_books, True,
              f"{n_books} book(s) quote the full market at this line (need {s.min_books})")
        # 6: price outlier
        if c.best is not None and c.fair_prob is not None:
            gap = c.fair_prob - 1 / c.best.price_decimal
            check("price_not_outlier", gap <= s.outlier_threshold, True,
                  f"best price is {gap * 100:.1f} pp better than consensus", contradiction=True)
        # 7: statistics dated correctly
        if ctx.history:
            latest = max(ensure_utc(g.date) for g in ctx.history)
            check("history_before_game", latest < c.event.commence_time, True,
                  f"latest history game {latest:%Y-%m-%d}", contradiction=True)
            recent = latest > now - timedelta(days=45)
            check("history_current_season", recent, False,
                  "history is current" if recent else f"latest result {latest:%Y-%m-%d}: may be previous season")
        elif ctx.cfg.model is not None:
            check("history_available", False, False, "no historical results retrieved")
        # 8: injuries
        if ctx.cfg.injuries:
            check("injuries_retrieved", ctx.injuries is not None, False,
                  "injury report retrieved" if ctx.injuries is not None else (ctx.injuries_error or "unavailable"))
        # sanity: selection belongs to event
        if c.side in ("home", "away"):
            check("selection_in_event", c.selection in (c.event.home_team, c.event.away_team), True,
                  "selection matches event teams", contradiction=True)
        if c.is_live:
            flags.append("live market: pre-game model does not account for in-game state")

        if contradictions:
            verdict, score = "REJECTED", 0.0
        elif unverifiable:
            verdict, score = "INSUFFICIENT_DATA", 0.3
        else:
            verdict, score = "VERIFIED", max(0.5, 1.0 - 0.1 * len(flags))
        reasons = contradictions + unverifiable
        summary = {"VERIFIED": f"{sum(ch['passed'] for ch in checks)}/{len(checks)} checks passed",
                   "REJECTED": "contradicting evidence: " + "; ".join(contradictions),
                   "INSUFFICIENT_DATA": "could not verify: " + "; ".join(unverifiable)}[verdict]
        return AgentReport(VERIFY, verdict, round(score, 3), summary, reasons=reasons, flags=flags,
                           data={"checks": checks, "independent_source": "espn" if ctx.schedule is not None else None,
                                 "neutral_site": c.neutral_site})

    @staticmethod
    def odds_still_fresh(c: Candidate, now, settings) -> bool:
        return c.best is not None and not is_stale(c.best.last_update, now, c.is_live, settings)
