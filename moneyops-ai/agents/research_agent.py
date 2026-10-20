"""Agent 1 - Live Data Researcher.

Turns provider odds into candidate wagers and documents, for every one:
which books offer it, each price and its timestamp, the best fresh price,
the no-vig consensus fair probability, data freshness, conflicts between
books, and what supporting data (history, injuries) is or is not available.
It never fills gaps: missing items are listed in `missing`.
"""
from __future__ import annotations

import hashlib
import statistics
from collections import defaultdict

from agents.base import RESEARCH, AgentReport, Candidate, SportContext
from data.injuries import injuries_for_teams, significant
from utils.teams import normalize_team
from data.schemas import Event, OutcomePrice
from models.expected_value import overround, remove_vig
from risk.risk_limits import is_stale, odds_age_minutes


def candidate_key(event_id: str, market: str, selection: str, point, description) -> str:
    raw = f"{event_id}|{market}|{selection}|{point}|{description or ''}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _side(event: Event, name: str) -> str | None:
    if name == event.home_team:
        return "home"
    if name == event.away_team:
        return "away"
    low = name.lower()
    if low in ("draw", "tie"):
        return "draw"
    if low in ("over", "under"):
        return low
    return None


def _complement_set(market: str, outcome: OutcomePrice, book_outcomes: list[OutcomePrice], has_draw: bool):
    """Return the full set of mutually-exclusive outcomes from one book that
    includes `outcome`, or None if the book's market is incomplete."""
    if market == "h2h":
        needed = 3 if has_draw else 2
        return book_outcomes if len(book_outcomes) == needed else None
    if market == "spreads":
        opp = [o for o in book_outcomes if o.name != outcome.name and o.point is not None
               and outcome.point is not None and abs(o.point + outcome.point) < 1e-9]
        return [outcome, opp[0]] if len(opp) == 1 else None
    # totals and props: Over/Under at the same point (and same player)
    other = "Under" if outcome.name == "Over" else "Over"
    opp = [o for o in book_outcomes if o.name == other and o.point == outcome.point
           and o.description == outcome.description]
    return [outcome, opp[0]] if len(opp) == 1 else None


class ResearchAgent:
    name = RESEARCH

    def __init__(self, settings):
        self.s = settings

    def build_candidates(self, events: list[Event], contexts: dict[str, SportContext], now) -> list[Candidate]:
        out: list[Candidate] = []
        for ev in events:
            ctx = contexts.get(ev.sport_code)
            cfg = ctx.cfg if ctx else None
            if cfg is None:
                continue
            by_market_book: dict[tuple, list[OutcomePrice]] = defaultdict(list)
            for p in ev.prices:
                by_market_book[(p.market, p.bookmaker)].append(p)
            selections: dict[tuple, list[OutcomePrice]] = defaultdict(list)
            for p in ev.prices:
                selections[(p.market, p.name, p.point, p.description)].append(p)
            for (market, name, point, desc), books in selections.items():
                side = _side(ev, name)
                if side is None:
                    continue
                cand = Candidate(key=candidate_key(ev.event_id, market, name, point, desc), event=ev, cfg=cfg,
                                 market=market, selection=name, side=side, point=point, description=desc,
                                 books=sorted(books, key=lambda b: -b.price_decimal), is_live=ev.is_live(now))
                for b in books:
                    book_set = _complement_set(market, b, by_market_book[(market, b.bookmaker)], cfg.has_draw)
                    if book_set is None:
                        continue
                    fair = remove_vig([o.price_decimal for o in book_set])
                    idx = next(i for i, o in enumerate(book_set) if o is b)
                    cand.fair_by_book[b.bookmaker] = {"fair": fair[idx], "provider": b.provider,
                                                      "overround": overround([o.price_decimal for o in book_set])}
                if cand.fair_by_book:
                    cand.fair_prob = statistics.median(v["fair"] for v in cand.fair_by_book.values())
                cand.reports[RESEARCH] = self._report(cand, ctx, now)
                out.append(cand)
        return out

    def _report(self, c: Candidate, ctx: SportContext, now) -> AgentReport:
        s = self.s
        fresh = [b for b in c.books if not is_stale(b.last_update, now, c.is_live, s)]
        c.best = max(fresh, key=lambda b: b.price_decimal) if fresh else None
        flags, missing = [], []
        if not fresh:
            flags.append("STALE: no bookmaker price is within the freshness window")
        if c.is_live:
            flags.append("LIVE: game in progress - odds are live and change quickly")
        if c.fair_prob is None:
            missing.append("no bookmaker offers the complete market, so a no-vig probability cannot be computed")
        disp = None
        if len(c.fair_by_book) >= 2:
            disp = statistics.pstdev(v["fair"] for v in c.fair_by_book.values())
            if disp > 0.03:
                flags.append(f"CONFLICT: books disagree (no-vig std {disp:.3f})")
        if c.best is not None and c.fair_prob is not None:
            edge = c.fair_prob - 1 / c.best.price_decimal
            if edge > s.outlier_threshold:
                flags.append(f"OUTLIER: best price implies {1 / c.best.price_decimal:.3f} vs consensus fair "
                             f"{c.fair_prob:.3f}; possibly stale or erroneous line")
        hist_teams = {}
        if c.is_prop:
            pass
        elif ctx.cfg.model is None:
            missing.append(f"no statistical model / history source for {ctx.cfg.code}")
        elif not ctx.history:
            missing.append("historical results unavailable")
        else:
            for t in (c.event.home_team, c.event.away_team):
                hist_teams[t] = sum(1 for g in ctx.history if t in (g.home_team, g.away_team))
                if hist_teams[t] == 0:
                    missing.append(f"no historical games found for {t} (name mismatch or new team)")
        if c.is_prop:
            player = normalize_team(c.description or "")
            logs = [l for t in (c.event.home_team, c.event.away_team) for l in ctx.prop_logs.get(t, [])
                    if normalize_team(l["player"]) == player]
            if not ctx.prop_module:
                missing.append(f"no player game-log source for {ctx.cfg.code} props")
            elif not logs:
                missing.append(f"no sportsipy game logs found for {c.description}")
        injuries = {}
        if ctx.injuries is not None:
            inj = injuries_for_teams(ctx.injuries, [c.event.home_team, c.event.away_team])
            injuries = {t: [{"player": r.player, "status": r.status, "detail": r.detail} for r in significant(v)]
                        for t, v in inj.items()}
        elif ctx.cfg.injuries:
            missing.append(f"injury report unavailable ({ctx.injuries_error or 'not fetched'})")
        else:
            missing.append("no injury feed for this sport")
        books_table = [{"book": b.bookmaker_title, "provider": b.provider, "price": b.price_decimal, "point": b.point,
                        "last_update": b.last_update.isoformat(),
                        "age_min": round(odds_age_minutes(b.last_update, now), 1),
                        "fair": c.fair_by_book.get(b.bookmaker, {}).get("fair")} for b in c.books]
        critical_missing = c.best is None or c.fair_prob is None
        verdict = "STALE" if not fresh else ("INCOMPLETE" if critical_missing or missing else "COMPLETE")
        score = 0.0 if critical_missing else max(0.0, 1.0 - 0.15 * len(missing) - 0.1 * len(
            [f for f in flags if not f.startswith("LIVE")]))
        summary = (f"{len(c.books)} book(s) quote {c.label}; best {c.best.price_decimal:.2f} at "
                   f"{c.best.bookmaker_title}" if c.best else f"{len(c.books)} book(s), none fresh")
        if c.fair_prob is not None:
            summary += f"; consensus no-vig {c.fair_prob:.3f}"
        return AgentReport(RESEARCH, verdict, round(score, 3), summary, reasons=[], flags=flags, data={
            "sources": sorted({b.provider for b in c.books} | {g.source for g in ctx.history[:1]}
                              | ({"sportsipy"} if c.is_prop and ctx.prop_logs else set())),
            "odds_retrieved_at": c.event.retrieved_at.isoformat(),
            "books": books_table, "n_books": len(c.books), "n_fresh_books": len(fresh),
            "n_complete_books": len(c.fair_by_book), "fair_prob": c.fair_prob, "fair_dispersion": disp,
            "best_book": c.best.bookmaker_title if c.best else None,
            "best_price": c.best.price_decimal if c.best else None,
            "best_last_update": c.best.last_update.isoformat() if c.best else None,
            "history_games_per_team": hist_teams, "injuries": injuries, "missing": missing,
            "start_time": c.event.commence_time.isoformat(), "live": c.is_live})
