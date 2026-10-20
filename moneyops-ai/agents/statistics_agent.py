"""Agent 3 - Historical Statistician.

For each candidate it replays the *exact* bet (same side, same line) over
the relevant teams' completed games strictly before kick-off and reports
win/loss/push counts, sample sizes, Wilson 95% intervals, last-5/10/20,
season and full-lookback windows, home/away splits, results against
comparable opponents, rest, strength of schedule and streaks.

Important honesty notes carried in every report:
* "Hit rate vs today's line" is NOT a historical against-the-spread record:
  free sources do not provide historical closing lines.
* Past hit rates are descriptive; small samples and streaks are weak
  evidence. The verdict compares the Wilson interval with the break-even
  probability: SUPPORTS only if the lower bound clears it, CONTRADICTS only if
  the upper bound is below it, NEUTRAL otherwise (the usual case).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from agents.base import STATS, AgentReport, Candidate, SportContext
from data.schemas import HistoricalGame
from models.stats import hit_rate, streaks
from utils.timeutil import ensure_utc

CAVEATS = [
    "Historical hit rates describe the past; team strength, rosters and prices change.",
    "Results were replayed against today's line, not the lines that were available at the time.",
    "Win/loss streaks have little predictive value on their own.",
]


def team_games(history: list[HistoricalGame], team: str, before) -> list[HistoricalGame]:
    return [g for g in history if team in (g.home_team, g.away_team) and ensure_utc(g.date) < before]


def current_season(games: list[HistoricalGame], gap_days: int = 60) -> list[HistoricalGame]:
    """Games after the most recent off-season gap (> gap_days without a game)."""
    for i in range(len(games) - 1, 0, -1):
        if (ensure_utc(games[i].date) - ensure_utc(games[i - 1].date)) > timedelta(days=gap_days):
            return games[i:]
    return games


def outcome_for(g: HistoricalGame, market: str, side: str, point, team: str | None, has_draw: bool) -> str:
    if market == "totals":
        total = g.home_score + g.away_score
        if total == point:
            return "P"
        return "W" if (total > point) == (side == "over") else "L"
    if side == "draw":
        return "W" if g.home_score == g.away_score else "L"
    mine = g.home_score if g.home_team == team else g.away_score
    theirs = g.away_score if g.home_team == team else g.home_score
    if market == "h2h":
        if mine == theirs:
            return "L" if has_draw else "P"
        return "W" if mine > theirs else "L"
    adj = mine - theirs + point
    return "P" if adj == 0 else ("W" if adj > 0 else "L")


class StatisticsAgent:
    name = STATS

    def __init__(self, settings):
        self.s = settings

    def analyze(self, c: Candidate, ctx: SportContext) -> AgentReport:
        if c.is_prop:
            return AgentReport(STATS, "INSUFFICIENT_DATA", 0.0, "Player game logs are not integrated yet",
                               reasons=["no player-level history source connected"], data={"caveats": CAVEATS})
        if ctx.cfg.model is None or not ctx.history:
            return AgentReport(STATS, "INSUFFICIENT_DATA", 0.0, f"No historical results for {ctx.cfg.code}",
                               reasons=["historical data source unavailable for this sport"],
                               data={"caveats": CAVEATS})
        home, away = ctx.history_name(c.event.home_team), ctx.history_name(c.event.away_team)
        if not home or not away:
            return AgentReport(STATS, "INSUFFICIENT_DATA", 0.0, "Team names could not be matched to history",
                               reasons=[f"unmatched: {[t for t, m in ((c.event.home_team, home), (c.event.away_team, away)) if not m]}"],
                               data={"caveats": CAVEATS})
        before = c.event.commence_time
        team = {"home": home, "away": away}.get(c.side)
        opp = away if c.side == "home" else home if c.side == "away" else None
        has_draw = ctx.cfg.has_draw

        if team:  # team-specific market
            games = team_games(ctx.history, team, before)
        else:     # totals / draw: both teams' games
            seen = {}
            for g in team_games(ctx.history, home, before) + team_games(ctx.history, away, before):
                seen[g.game_id] = g
            games = sorted(seen.values(), key=lambda g: g.date)
        season = current_season(games) if team else sorted(
            {g.game_id: g for g in current_season(team_games(ctx.history, home, before)) +
             current_season(team_games(ctx.history, away, before))}.values(), key=lambda g: g.date)

        def series(gs):
            return [outcome_for(g, c.market, c.side, c.point, team, has_draw) for g in gs]

        windows = {}
        for label, gs in (("last_5", season[-5:]), ("last_10", season[-10:]), ("last_20", season[-20:]),
                          ("season", season), ("full_lookback", games)):
            windows[label] = hit_rate(series(gs)) | {"n_games": len(gs)}
        splits = {}
        if team:
            role_home = c.side == "home"
            role_games = [g for g in season if (g.home_team == team) == role_home and not g.neutral_site]
            splits["same_venue_role"] = hit_rate(series(role_games)) | {
                "role": "home" if role_home else "away", "n_games": len(role_games)}
        # comparable opponents by model rating tier
        model = ctx.model
        if model is not None and team and opp:
            ratings = {t: model.team_rating(t) for t in model.team_index}
            vals = np.array([v for v in ratings.values() if v is not None])
            if len(vals) >= 6 and ratings.get(opp) is not None:
                lo, hi = np.quantile(vals, [1 / 3, 2 / 3])
                tier = lambda r: 0 if r < lo else (1 if r < hi else 2)
                opp_tier = tier(ratings[opp])
                comp = [g for g in season if ratings.get(g.away_team if g.home_team == team else g.home_team)
                        is not None and tier(ratings[g.away_team if g.home_team == team else g.home_team]) == opp_tier]
                splits["vs_comparable_opponents"] = hit_rate(series(comp)) | {
                    "tier": ["bottom third", "middle third", "top third"][opp_tier], "n_games": len(comp)}
                sos = [ratings.get(g.away_team if g.home_team == team else g.home_team) for g in season]
                sos = [x for x in sos if x is not None]
                splits["strength_of_schedule"] = {"mean_opponent_rating": float(np.mean(sos)) if sos else None,
                                                  "league_mean": float(vals.mean())}
        rest = {}
        for t in (home, away):
            tg = team_games(ctx.history, t, before)
            if tg:
                days = (before - ensure_utc(tg[-1].date)).total_seconds() / 86400
                rest[t] = {"days_since_last_game": round(days, 1), "back_to_back": days < 1.5}
        streak = streaks(series(season))
        record = None
        if team:
            wl = [outcome_for(g, "h2h", "home" if g.home_team == team else "away", None, team, has_draw) for g in season]
            record = {"wins": wl.count("W"), "losses": wl.count("L"), "ties": wl.count("P")}

        primary = windows["season"] if windows["season"]["n"] >= self.s.min_sample else windows["full_lookback"]
        primary_label = "season" if primary is windows["season"] else "full_lookback"
        be = 1 / c.best.price_decimal if c.best else None
        reasons, flags = [], []
        if primary["n"] < self.s.min_sample:
            verdict, score = "INSUFFICIENT_DATA", 0.0
            reasons.append(f"only {primary['n']} decided games (minimum {self.s.min_sample})")
        elif be is None:
            verdict, score = "INSUFFICIENT_DATA", 0.0
            reasons.append("no fresh price to compare against")
        elif primary["ci_low"] > be:
            verdict, score = "SUPPORTS", 0.8
            reasons.append(f"{primary_label} hit rate {primary['rate']:.1%} (95% CI {primary['ci_low']:.1%}-"
                           f"{primary['ci_high']:.1%}, n={primary['n']}) is above break-even {be:.1%}")
        elif primary["ci_high"] < be:
            verdict, score = "CONTRADICTS", 0.2
            reasons.append(f"{primary_label} hit rate {primary['rate']:.1%} (CI upper {primary['ci_high']:.1%}) "
                           f"is below break-even {be:.1%}")
        else:
            verdict, score = "NEUTRAL", 0.5
            reasons.append(f"{primary_label} hit rate {primary['rate']:.1%} (95% CI {primary['ci_low']:.1%}-"
                           f"{primary['ci_high']:.1%}, n={primary['n']}) is statistically indistinguishable "
                           f"from break-even {be:.1%}")
        for t, r in rest.items():
            if r["back_to_back"]:
                flags.append(f"{t} on back-to-back / short rest")
        if windows["last_5"]["n"] >= 5 and windows["last_5"]["rate"] in (0.0, 1.0):
            flags.append("5-game streak present; treat as noise unless supported by larger samples")
        summary = reasons[0] if reasons else verdict
        return AgentReport(STATS, verdict, score, summary, reasons=reasons, flags=flags, data={
            "primary_window": primary_label, "primary": primary, "windows": windows, "splits": splits,
            "rest": rest, "streaks": streak, "team_record_season": record, "break_even": be,
            "caveats": CAVEATS, "leakage_guard": f"only games before {before.isoformat()} used"})
