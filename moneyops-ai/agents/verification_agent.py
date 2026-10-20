"""Agent 2 - Independent Fact-Checker.

Re-derives the critical facts from sources other than the odds feed:
* the game exists on at least one independent schedule (ESPN and/or
  AllSportsAPI) with the same teams; when two sources are available they
  must not contradict each other (status, start time, orientation),
* the start time agrees, and the game has not already finished/postponed,
* the price is fresh and quoted by enough independent books,
* the best price is not an outlier versus the multi-book consensus,
* when two odds PROVIDERS quote the same line, their consensus agrees,
* historical statistics are dated before the game (no leakage) and recent
  enough to belong to the current season,
* injury information was actually retrieved,
* player props: the player's latest game log (nba_api / nflreadpy) is with
  one of the two teams, he is playing recently, and is not listed Out/Doubtful.

Verdicts: VERIFIED, REJECTED (contradicting evidence) or INSUFFICIENT_DATA
(a critical fact could not be checked). Only VERIFIED can be fully approved.
"""
from __future__ import annotations

import statistics
from datetime import timedelta

from agents.base import VERIFY, AgentReport, Candidate, SportContext
from data.provider_base import ProviderError
from data.schedules import fetch_schedule, find_scheduled_game
from data.sports_registry import allsports_path
from risk.risk_limits import is_stale
from utils.teams import normalize_team
from utils.timeutil import ensure_utc, minutes_between

BAD_STATUS = ("postpon", "cancel", "suspend", "abandon")


class VerificationAgent:
    name = VERIFY

    def __init__(self, settings, espn, allsports=None):
        self.s = settings
        self.espn = espn
        self.allsports = allsports

    def prepare(self, ctx: SportContext, events) -> None:
        """Fetch independent schedules for this sport (own requests; the
        AllSportsAPI fixtures call is cached, so reusing it costs nothing)."""
        if not events:
            return
        start = min(e.commence_time for e in events)
        end = max(e.commence_time for e in events)
        if ctx.cfg.espn_path and self.espn is not None and self.s.espn_enabled:
            try:
                ctx.schedules["espn"] = fetch_schedule(self.espn, ctx.cfg, start, end)
            except ProviderError as exc:
                ctx.schedule_errors["espn"] = str(exc)
        path = allsports_path(ctx.cfg, self.s.allsports_path_overrides)
        if path and self.allsports is not None and self.allsports.is_configured():
            try:
                games, _ = self.allsports.fixtures(ctx.cfg.code, path, start - timedelta(days=1),
                                                   end + timedelta(days=1), ctx.league_keys.get("allsportsapi"),
                                                   ctx.cfg.has_draw)
                ctx.schedules["allsportsapi"] = games
            except ProviderError as exc:
                ctx.schedule_errors["allsportsapi"] = str(exc)
        if not ctx.schedules and not ctx.schedule_errors:
            ctx.schedule_errors["none"] = f"no independent schedule source for {ctx.cfg.code}"

    def verify(self, c: Candidate, ctx: SportContext, now) -> AgentReport:
        s = self.s
        checks, contradictions, unverifiable, flags = [], [], [], []

        def check(name, ok, critical, detail, contradiction=False):
            checks.append({"check": name, "passed": ok, "critical": critical, "detail": detail})
            if not ok and critical:
                (contradictions if contradiction else unverifiable).append(f"{name}: {detail}")
            elif not ok:
                flags.append(f"{name}: {detail}")

        # 1-3: independent schedules
        matched = {}
        for source, games in ctx.schedules.items():
            game, detail = find_scheduled_game(c.event, games)
            if game is None:
                flags.append(f"game not found on {source} schedule")
            else:
                matched[source] = (game, detail)
        if not matched:
            why = "; ".join(f"{k}: {v}" for k, v in ctx.schedule_errors.items()) or "game not found on any schedule"
            check("independent_schedule", False, True, why)
        else:
            check("independent_schedule", True, True,
                  "matched " + ", ".join(f"{src} game {g.game_id}" for src, (g, _) in matched.items()))
            for src, (game, detail) in matched.items():
                if src == "espn":
                    c.neutral_site = game.neutral_site
                check(f"home_away_orientation[{src}]", not detail["home_away_swapped"], True,
                      "home/away teams reversed between sources" if detail["home_away_swapped"] else "consistent",
                      contradiction=True)
                diff = detail["start_diff_minutes"]
                check(f"start_time[{src}]", diff <= s.max_start_time_diff_minutes, True,
                      f"odds feed and {src} differ by {diff:.0f} min", contradiction=True)
                status = game.status_detail.lower()
                bad = game.completed or game.state == "post" or any(w in status for w in BAD_STATUS)
                check(f"game_status[{src}]", not bad, True, f"{src} status: {game.status_detail or game.state}",
                      contradiction=True)
                if game.state == "in":
                    flags.append(f"game in progress per {src}")
            if len(matched) >= 2:
                (g1, _), (g2, _) = list(matched.values())[:2]
                gap = abs(minutes_between(g1.start_time, g2.start_time))
                check("schedule_sources_agree", gap <= s.max_start_time_diff_minutes, True,
                      f"independent sources differ by {gap:.0f} min", contradiction=True)
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
        # 6b: cross-provider odds agreement
        by_provider: dict[str, list[float]] = {}
        for v in c.fair_by_book.values():
            by_provider.setdefault(v.get("provider", "the-odds-api"), []).append(v["fair"])
        if len(by_provider) >= 2:
            meds = {k: statistics.median(v) for k, v in by_provider.items()}
            spread = max(meds.values()) - min(meds.values())
            check("cross_provider_odds", spread <= s.outlier_threshold, True,
                  "provider consensus: " + ", ".join(f"{k} {v:.3f}" for k, v in meds.items()), contradiction=True)
        # 7: statistics dated correctly
        if ctx.history:
            latest = max(ensure_utc(g.date) for g in ctx.history)
            check("history_before_game", latest < c.event.commence_time, True,
                  f"latest history game {latest:%Y-%m-%d}", contradiction=True)
            recent = latest > now - timedelta(days=45)
            check("history_current_season", recent, False,
                  "history is current" if recent else f"latest result {latest:%Y-%m-%d}: may be previous season")
        elif ctx.cfg.model is not None and not c.is_prop:
            check("history_available", False, False, "no historical results retrieved")
        # 8: injuries
        if ctx.cfg.injuries:
            check("injuries_retrieved", ctx.injuries is not None, False,
                  "injury report retrieved" if ctx.injuries is not None else (ctx.injuries_error or "unavailable"))
        # 9: player props
        if c.is_prop:
            self._verify_prop(c, ctx, now, check, flags)
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
            verdict, score = "VERIFIED", max(0.5, 1.0 - 0.1 * len(flags) + 0.1 * (len(matched) >= 2))
            score = min(1.0, score)
        reasons = contradictions + unverifiable
        summary = {"VERIFIED": f"{sum(ch['passed'] for ch in checks)}/{len(checks)} checks passed "
                               f"({len(matched)} independent schedule source(s))",
                   "REJECTED": "contradicting evidence: " + "; ".join(contradictions),
                   "INSUFFICIENT_DATA": "could not verify: " + "; ".join(unverifiable)}[verdict]
        return AgentReport(VERIFY, verdict, round(score, 3), summary, reasons=reasons, flags=flags,
                           data={"checks": checks, "independent_sources": sorted(matched),
                                 "neutral_site": c.neutral_site})

    def _verify_prop(self, c: Candidate, ctx: SportContext, now, check, flags) -> None:
        player = normalize_team(c.description or "")
        teams = [t for t in (c.event.home_team, c.event.away_team) if ctx.prop_logs.get(t)]
        if not teams:
            errs = "; ".join(f"{t}: {e}" for t, e in ctx.prop_errors.items()) or "no player game logs retrieved"
            check("player_on_roster", False, True, errs)
            return
        found = [(t, l) for t in teams for l in ctx.prop_logs[t] if normalize_team(l["player"]) == player]
        if not found:
            check("player_on_roster", False, True,
                  f"{c.description} not found in recent game logs of {', '.join(teams)}")
            return
        team = max(found, key=lambda x: x[1]["date"])[0]
        check("player_on_roster", True, True, f"{c.description} plays for {team} ({found[0][1]['source']})")
        last = max(l["date"] for _, l in found)
        team_dates = sorted({l["date"] for l in ctx.prop_logs[team] if l["team"] == team})
        missed = sum(1 for d in team_dates[-3:] if d > last)
        check("player_recently_active", missed == 0, False,
              f"last appearance {last}" + (f"; missed team's last {missed} game(s)" if missed else ""))
        if ctx.injuries is not None:
            for r in ctx.injuries:
                if normalize_team(r.player) == player:
                    st = r.status.lower()
                    if st in ("out", "doubtful", "injured reserve", "suspension"):
                        check("player_available", False, True, f"injury report: {r.status} ({r.detail})",
                              contradiction=True)
                    else:
                        flags.append(f"injury report lists {c.description}: {r.status} ({r.detail})")

    @staticmethod
    def odds_still_fresh(c: Candidate, now, settings) -> bool:
        return c.best is not None and not is_stale(c.best.last_update, now, c.is_live, settings)
