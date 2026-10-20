"""Daily five-agent analysis workflow (orchestrator).

1. Discover in-season sports (free Odds API /sports call).
2. Pull odds per sport within the credit budget (The Odds API), then add
   AllSportsAPI odds/fixtures/probabilities where configured.
3. Pull history (ESPN, AllSportsAPI fallback), fit models, backtest (daily).
4. Pull injuries; Agent 2 fetches independent schedules (ESPN, AllSportsAPI);
   player game logs for props (sportsipy).
5. Agents 1-4 per candidate.  6. Agent 5 audit + portfolio.  7. Parlays.
8. Persist + coverage report.
Nothing is fabricated: any provider failure is recorded as a coverage gap.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from agents.base import AGENT_NAMES, AUDIT, ENDORSE, PROB, RESEARCH, STATS, VERIFY, Candidate, SportContext
from agents.chief_auditor import NO_QUALIFYING, ChiefAuditor
from agents.probability_agent import ProbabilityAgent
from agents.research_agent import ResearchAgent
from agents.statistics_agent import StatisticsAgent
from agents.verification_agent import VerificationAgent
from config import MODEL_VERSION, Settings
from data.allsports_provider import AllSportsProvider
from data.cache import ResponseCache
from data.odds_provider import OddsApiProvider
from data.provider_base import ProviderBudgetExceeded, ProviderError, ProviderNotConfigured
from data.schedules import find_scheduled_game
from data.schemas import Event
from data.sports_data_provider import ESPNProvider
from data.sports_registry import SPORTS, SportConfig, allsports_path, sport_for_odds_key
from data.sportsipy_provider import SportsipyProvider, lines_to_dicts
from database.repository import Repository
from models.backtesting import walk_forward
from models.probability import TeamRatingModel
from risk.bankroll import BankrollService
from utils.teams import match_team
from utils.timeutil import utcnow

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    run_id: int
    status: str
    events: list[Event] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    parlays: list[dict] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    message: str = ""


@dataclass
class Providers:
    odds: object
    espn: object
    allsports: object
    sportsipy: object

    def all(self):
        return [p for p in (self.odds, self.espn, self.allsports, self.sportsipy) if p is not None]


def build_providers(settings: Settings, repo: Repository) -> Providers:
    cache = ResponseCache(repo.Session)
    return Providers(OddsApiProvider(settings, cache), ESPNProvider(settings, cache),
                     AllSportsProvider(settings, cache), SportsipyProvider(settings, cache))


def _configured(provider) -> bool:
    if provider is None:
        return False
    fn = getattr(provider, "is_configured", None)
    return bool(fn()) if callable(fn) else True


class AnalysisService:
    def __init__(self, settings: Settings, repo: Repository, odds=None, espn=None, allsports=None, sportsipy=None):
        self.s, self.repo = settings, repo
        if odds is None or espn is None or allsports is None or sportsipy is None:
            built = build_providers(settings, repo)
            odds, espn = odds or built.odds, espn or built.espn
            allsports = allsports if allsports is not None else built.allsports
            sportsipy = sportsipy if sportsipy is not None else built.sportsipy
        self.odds, self.espn, self.allsports, self.sportsipy = odds, espn, allsports, sportsipy
        self.agent1, self.agent2 = ResearchAgent(settings), VerificationAgent(settings, espn, allsports)
        self.agent3, self.agent4, self.agent5 = StatisticsAgent(settings), ProbabilityAgent(settings), ChiefAuditor(settings)
        self._active: set = set()
        self._league_keys: dict[str, str] = {}
        self._fixtures: dict[str, list] = {}
        self._ext_probs: dict[str, dict] = {}

    # ------------------------------------------------------------ targets
    def _targets(self, coverage: dict) -> list[SportConfig]:
        if not _configured(self.odds):
            if not _configured(self.allsports):
                raise ProviderNotConfigured("No odds provider configured: set ODDS_API_KEY and/or ALLSPORTS_API_KEY.")
            targets = []
            for code in self.s.enabled_sports:
                cfg = SPORTS.get(code)
                if cfg and allsports_path(cfg, self.s.allsports_path_overrides):
                    coverage.setdefault(cfg.code, self._coverage_row(cfg))["notes"].append(
                        "The Odds API not configured: AllSportsAPI odds only")
                    targets.append(cfg)
            return targets
        sports = self.odds.list_sports()
        active = {s["key"]: s for s in sports if s.get("active") and not s.get("has_outrights")}
        self._active = set(active)
        targets = []
        for code in self.s.enabled_sports:
            cfg = SPORTS.get(code)
            if cfg is None:
                continue
            row = coverage.setdefault(cfg.code, self._coverage_row(cfg))
            if cfg.odds_key in active:
                row["in_season"] = True
                targets.append(cfg)
            else:
                row["notes"].append("not in season / no active odds on provider")
        for key, info in active.items():
            if key in {t.odds_key for t in targets}:
                continue
            is_tennis = key.startswith("tennis")
            if (is_tennis and self.s.scan_tennis) or (not is_tennis and self.s.scan_other_sports and
                                                       key not in {c.odds_key for c in SPORTS.values()}):
                cfg = sport_for_odds_key(key, info.get("group", ""), info.get("title", ""))
                coverage.setdefault(cfg.code, self._coverage_row(cfg))["in_season"] = True
                targets.append(cfg)
            elif is_tennis:
                coverage.setdefault("TENNIS", {**self._coverage_row(sport_for_odds_key(key)), "sport": "TENNIS",
                                               "league": "Tennis (all active tournaments)"})["notes"] = [
                    "active on provider; set SCAN_TENNIS=true to scan (odds only, no model)"]
        return targets

    def _coverage_row(self, cfg: SportConfig) -> dict:
        sources = (["espn"] if cfg.espn_path else []) + (
            ["allsportsapi"] if allsports_path(cfg, self.s.allsports_path_overrides) else [])
        return {"sport": cfg.code, "league": cfg.league, "odds_key": cfg.odds_key, "in_season": False,
                "odds_fetched": False, "events": 0, "independent_schedule": ", ".join(sources) or "none",
                "history_games": 0, "model": cfg.model or "none", "injuries": cfg.injuries,
                "player_props": False, "prop_logs": "sportsipy" if cfg.sportsipy else "none",
                "allsports": "not used", "backtest": None, "notes": []}

    # ------------------------------------------------------- AllSportsAPI
    def _allsports_league(self, cfg: SportConfig, path: str, row: dict) -> tuple[bool, str | None]:
        """(ok, league_key). A configured league that cannot be resolved is a
        gap - we never fall back to pulling every league of a sport."""
        if not cfg.allsports_league:
            return True, None
        name, country = cfg.allsports_league
        key = self.allsports.resolve_league(path, name, country)
        if key is None:
            row["notes"].append(f"AllSportsAPI: league '{name}' ({country or 'any'}) not found in met=Leagues")
            return False, None
        self._league_keys[cfg.code] = key
        return True, key

    def _add_allsports(self, cfg: SportConfig, events: list[Event], now, horizon, row: dict) -> None:
        path = allsports_path(cfg, self.s.allsports_path_overrides)
        if not path or not _configured(self.allsports):
            return
        try:
            ok, league = self._allsports_league(cfg, path, row)
            if not ok:
                return
            games, _ = self.allsports.fixtures(cfg.code, path, now - timedelta(hours=6), horizon, league,
                                               cfg.has_draw)
            self._fixtures[cfg.code] = games
            sport_events = [e for e in events if e.sport_code == cfg.code]
            matched_ids = {}
            for ev in sport_events:
                g, _ = find_scheduled_game(ev, games)
                if g is not None:
                    matched_ids[g.game_id] = ev
            added, merged = 0, 0
            if self.s.allsports_use_odds:
                by_key = {g.game_id: g for g in games}
                for as_ev in self.allsports.odds(cfg.code, cfg.odds_key or f"allsports_{path}", cfg.league, path,
                                                 now - timedelta(hours=6), horizon, league, cfg.has_draw, by_key):
                    key = as_ev.event_id.removeprefix("as:")
                    target = matched_ids.get(key)
                    if target is not None:
                        target.prices.extend(as_ev.prices)
                        merged += 1
                    elif now - timedelta(hours=6) <= as_ev.commence_time <= horizon:
                        events.append(as_ev)
                        matched_ids[key] = as_ev
                        added += 1
            probs = {}
            if self.s.allsports_use_probabilities:
                for key, p in self.allsports.probabilities(path, now - timedelta(hours=6), horizon, league).items():
                    if key in matched_ids:
                        probs[matched_ids[key].event_id] = p
            self._ext_probs[cfg.code] = probs
            row["allsports"] = (f"{len(games)} fixtures, odds merged into {merged} game(s), {added} extra game(s), "
                                f"{len(probs)} probability set(s)")
            row["events"] = sum(1 for e in events if e.sport_code == cfg.code)
            row["odds_fetched"] = row["odds_fetched"] or bool(merged or added)
        except ProviderBudgetExceeded as exc:
            row["notes"].append(f"AllSportsAPI: {exc}")
        except ProviderError as exc:
            row["notes"].append(f"AllSportsAPI unavailable: {exc}")

    # ------------------------------------------------------------ contexts
    def _context(self, cfg: SportConfig, events: list[Event], now, row: dict) -> SportContext:
        ctx = SportContext(cfg)
        if cfg.code in self._league_keys:
            ctx.league_keys["allsportsapi"] = self._league_keys[cfg.code]
        ctx.external_probs = self._ext_probs.get(cfg.code, {})
        if cfg.model and cfg.espn_path and self.s.espn_enabled:
            try:
                ctx.history = self.espn.history(cfg.code, cfg.espn_path, now, self.s.history_lookback_days,
                                                cfg.espn_groups)
            except ProviderError as exc:
                row["notes"].append(f"ESPN history unavailable: {exc}")
        path = allsports_path(cfg, self.s.allsports_path_overrides)
        if cfg.model and not ctx.history and path and _configured(self.allsports):
            ok = cfg.code in self._league_keys or not cfg.allsports_league
            if ok:
                try:
                    ctx.history = self.allsports.history(cfg.code, path, now, self.s.history_lookback_days,
                                                         self._league_keys.get(cfg.code), cfg.has_draw)
                    row["notes"].append("history from AllSportsAPI (ESPN unavailable)")
                except ProviderError as exc:
                    row["notes"].append(f"AllSportsAPI history unavailable: {exc}")
        row["history_games"] = len(ctx.history)
        if cfg.model:
            names = sorted({g.home_team for g in ctx.history} | {g.away_team for g in ctx.history})
            for ev in events:
                for t in (ev.home_team, ev.away_team):
                    if t not in ctx.team_map:
                        ctx.team_map[t] = match_team(t, names)[0] if names else None
            if len(ctx.history) >= 10:
                try:
                    ctx.model = TeamRatingModel(cfg, None, self.s.history_half_life_days,
                                                self.s.min_games_per_team).fit(ctx.history, now)
                    ctx.backtest = self._backtest(cfg, ctx, now, row)
                    if ctx.backtest and ctx.backtest.get("alpha"):
                        ctx.model = TeamRatingModel(cfg, ctx.backtest["alpha"], self.s.history_half_life_days,
                                                    self.s.min_games_per_team).fit(ctx.history, now)
                    ctx.bootstrap_models = ctx.model.bootstrap(self.s.bootstrap_samples, self.s.random_seed)
                except (ValueError, ArithmeticError) as exc:
                    row["notes"].append(f"model fit failed: {exc}")
                    ctx.model = None
        if cfg.injuries and cfg.espn_path and self.s.espn_enabled:
            try:
                ctx.injuries = self.espn.injuries(cfg.espn_path)
            except ProviderError as exc:
                ctx.injuries_error = str(exc)
                row["notes"].append(f"injuries unavailable: {exc}")
        self._prop_logs(ctx, events, row)
        return ctx

    def _prop_logs(self, ctx: SportContext, events: list[Event], row: dict) -> None:
        prop_events = [e for e in events if any(p.market.startswith("player_") for p in e.prices)]
        if not prop_events or not ctx.cfg.sportsipy:
            return
        if not _configured(self.sportsipy):
            row["notes"].append("props found but sportsipy disabled (SPORTSIPY_ENABLED=false): no player history")
            return
        ctx.prop_module = ctx.cfg.sportsipy
        for ev in prop_events:
            for team in (ev.home_team, ev.away_team):
                if team in ctx.prop_logs or team in ctx.prop_errors:
                    continue
                try:
                    ctx.prop_logs[team] = lines_to_dicts(
                        self.sportsipy.team_player_logs(ctx.prop_module, team, ev.commence_time))
                except ProviderBudgetExceeded as exc:
                    ctx.prop_errors[team] = str(exc)
                except ProviderError as exc:
                    ctx.prop_errors[team] = str(exc)
        n = sum(len(v) for v in ctx.prop_logs.values())
        row["prop_logs"] = f"sportsipy: {len(ctx.prop_logs)} team(s), {n} player-game lines"
        for team, err in ctx.prop_errors.items():
            row["notes"].append(f"sportsipy {team}: {err}")

    def _backtest(self, cfg, ctx, now, row) -> dict | None:
        latest = self.repo.latest_backtests().get(cfg.code)
        if latest is not None and latest.created_at.date() == now.date() and latest.model_version == MODEL_VERSION:
            row["backtest"] = latest.status
            return latest.metrics_json
        res = walk_forward(cfg, ctx.history, self.s.history_half_life_days, self.s.min_games_per_team)
        self.repo.save_backtest(cfg.code, MODEL_VERSION, res["status"], res)
        row["backtest"] = res["status"]
        return res

    # ----------------------------------------------------------------- run
    def run(self, now=None) -> RunResult:
        now = now or utcnow()
        run_id = self.repo.create_run(MODEL_VERSION, self.s.public_dict())
        result = RunResult(run_id, "RUNNING")
        coverage: dict[str, dict] = {}
        for code in self.s.enabled_sports:
            if code in SPORTS:
                coverage[code] = self._coverage_row(SPORTS[code])
        self.repo.audit("orchestrator", "run_started", f"Analysis run {run_id} started", run_id=run_id)
        try:
            self._run(now, result, coverage)
            result.status = "COMPLETED"
        except ProviderNotConfigured as exc:
            result.status, result.message = "NO_DATA", f"{exc} {NO_QUALIFYING}"
            result.errors.append(str(exc))
        except ProviderError as exc:
            result.status, result.message = "FAILED", f"Data provider failure: {exc}. {NO_QUALIFYING}"
            result.errors.append(str(exc))
        except Exception as exc:  # keep a record of unexpected failures too
            log.exception("analysis run failed")
            result.status, result.message = "FAILED", f"Unexpected error: {exc}"
            result.errors.append(repr(exc))
        finally:
            for p in (self.odds, self.espn, self.allsports, self.sportsipy):
                if p is not None and getattr(p, "status", None) is not None:
                    self.repo.save_provider_status(p.status)
            qualified = [c for c in result.candidates if c.status == "QUALIFIED"]
            q_parlays = [p for p in result.parlays if p["status"] == "QUALIFIED"]
            if result.status == "COMPLETED" and not result.message:
                result.message = (f"{len(qualified)} qualifying single(s), {len(q_parlays)} qualifying parlay(s)"
                                  if qualified else NO_QUALIFYING)
            summary = {"message": result.message, "events": len(result.events),
                       "candidates": len(result.candidates), "qualified": len(qualified),
                       "needs_research": sum(c.status == "NEEDS_RESEARCH" for c in result.candidates),
                       "rejected": sum(c.status == "REJECTED" for c in result.candidates),
                       "parlays_qualified": len(q_parlays),
                       "odds_credits_spent": getattr(self.odds, "credits_spent_this_run", None),
                       "quota_remaining": self.odds.status.quota_remaining,
                       "allsports_requests": getattr(self.allsports, "requests_this_run", None),
                       "sportsipy_requests": getattr(self.sportsipy, "requests_this_run", None),
                       "as_of": now.isoformat()}
            result.coverage = list(coverage.values())
            self.repo.finish_run(run_id, result.status, summary, result.coverage, result.errors)
            self.repo.audit("orchestrator", "run_finished", result.message, summary, run_id=run_id,
                            level="INFO" if result.status == "COMPLETED" else "ERROR")
        return result

    def _run(self, now, result: RunResult, coverage: dict) -> None:
        s = self.s
        targets = self._targets(coverage)
        horizon = now + timedelta(hours=48)
        events: list[Event] = []
        for cfg in targets:
            row = coverage.setdefault(cfg.code, self._coverage_row(cfg))
            if _configured(self.odds) and cfg.odds_key in self._active:
                try:
                    evs = self.odds.get_odds(cfg.odds_key)
                except ProviderBudgetExceeded as exc:
                    row["notes"].append(str(exc))
                    evs = None
                except ProviderError as exc:
                    row["notes"].append(f"odds request failed: {exc}")
                    result.errors.append(f"{cfg.code}: {exc}")
                    evs = None
                if evs is not None:
                    row["odds_fetched"] = True
                    evs = [e for e in evs if now - timedelta(hours=6) <= e.commence_time <= horizon]
                    for e in evs:
                        e.sport_code, e.league = cfg.code, cfg.league
                    row["events"] = len(evs)
                    events.extend(evs)
            self._add_allsports(cfg, events, now, horizon, row)
        if s.include_player_props and _configured(self.odds):
            self._add_props(events, coverage, now)
        result.events = events
        self.repo.save_odds_snapshots(result.run_id, events)

        contexts: dict[str, SportContext] = {}
        for cfg in targets:
            evs = [e for e in events if e.sport_code == cfg.code]
            if not evs:
                continue
            ctx = self._context(cfg, evs, now, coverage[cfg.code])
            self.agent2.prepare(ctx, evs)
            for src, err in ctx.schedule_errors.items():
                coverage[cfg.code]["notes"].append(f"independent verification unavailable ({src}): {err}")
            contexts[cfg.code] = ctx

        bank = BankrollService(self.repo, s)
        summary = bank.summary()
        bankroll, open_exposure = summary["balance"], summary["open_exposure"]

        cands = self.agent1.build_candidates(events, contexts, now)
        for c in cands:
            ctx = contexts[c.cfg.code]
            c.reports[VERIFY] = self.agent2.verify(c, ctx, now)
            c.reports[STATS] = self.agent3.analyze(c, ctx)
            c.reports[PROB] = self.agent4.evaluate(c, ctx, bankroll)
            c.reports[AUDIT] = self.agent5.audit(c, ctx, now)
        self.agent5.finalize_portfolio(cands, bankroll, open_exposure)
        grid_cache: dict = {}
        sgp = {}
        for ev_id in {c.event.event_id for c in cands}:
            ev_cands = [c for c in cands if c.event.event_id == ev_id and c.reports[PROB].data.get("model_prob")]
            for a in ev_cands:
                for b in ev_cands:
                    if a.key < b.key:
                        f = self.agent5.same_game_factor(a, b, contexts[a.cfg.code], grid_cache)
                        if f is not None:
                            sgp.setdefault(a.key, {})[b.key] = f
                            sgp.setdefault(b.key, {})[a.key] = f
        result.parlays = self.agent5.build_parlays(cands, bankroll)
        result.candidates = cands
        self._persist(result, sgp)

    def _add_props(self, events, coverage, now):
        prop_sports = {"NBA", "NFL", "WNBA", "MLB", "NHL", "NCAAF", "NCAAB"}
        chosen = [e for e in sorted(events, key=lambda e: e.commence_time)
                  if e.sport_code in prop_sports and e.commence_time > now
                  and not e.event_id.startswith("as:")][: self.s.max_prop_events_per_run]
        for ev in chosen:
            try:
                extra = self.odds.get_event_odds(ev.sport_key, ev.event_id, self.s.prop_markets)
            except ProviderError as exc:
                coverage[ev.sport_code]["notes"].append(f"props skipped: {exc}")
                break
            for e in extra:
                ev.prices.extend(p for p in e.prices if p.market.startswith("player_"))
            coverage[ev.sport_code]["player_props"] = True

    # ------------------------------------------------------------- persist
    def _persist(self, result: RunResult, sgp: dict) -> None:
        run_id = result.run_id
        key_to_id = {}
        audit_rows = []
        for c in result.candidates:
            p4 = c.reports[PROB].data
            r3 = c.reports[STATS].data.get("primary") or {}
            verdicts = {a: {"name": AGENT_NAMES[a], "verdict": r.verdict, "score": r.score, "summary": r.summary,
                            "endorsed": r.verdict == ENDORSE[a]} for a, r in c.reports.items()}
            rec_id = self.repo.save_recommendation(
                run_id=run_id, kind="single", candidate_key=c.key, sport_code=c.cfg.code, league=c.cfg.league,
                event_id=c.event.event_id, event_name=c.event.name, home_team=c.event.home_team,
                away_team=c.event.away_team, sport_key=c.event.sport_key, commence_time=c.event.commence_time,
                is_live=c.is_live, market=c.market, selection=c.label, side=c.side, point=c.point,
                best_book=c.best.bookmaker_title if c.best else None,
                odds_decimal=c.best.price_decimal if c.best else None,
                odds_last_update=c.best.last_update if c.best else None, odds_retrieved_at=c.event.retrieved_at,
                model_prob=p4.get("model_prob"), market_implied_prob=p4.get("market_implied_prob"),
                fair_prob=c.fair_prob, final_prob=p4.get("final_prob"),
                prob_low=(p4.get("final_prob_interval") or [None, None])[0],
                prob_high=(p4.get("final_prob_interval") or [None, None])[1], push_prob=p4.get("push_prob"),
                expected_value=p4.get("expected_value"), expected_value_low=p4.get("expected_value_low"),
                hist_rate=r3.get("rate"), hist_n=r3.get("n"), suggested_stake=c.stake if c.status == "QUALIFIED" else 0.0,
                consensus_score=c.consensus_score, status=c.status, verdicts_json=verdicts,
                report_json={a: r.to_dict() for a, r in c.reports.items()} | {"same_game_factors": sgp.get(c.key, {}),
                                                                              "player": c.description},
                model_version=MODEL_VERSION)
            key_to_id[c.key] = rec_id
            for a, r in c.reports.items():
                audit_rows.append({"run_id": run_id, "agent": a, "action": f"verdict:{r.verdict}",
                                   "candidate_key": c.key, "message": f"{c.event.name} | {c.market} {c.label}: {r.summary}",
                                   "payload_json": {"reasons": r.reasons, "flags": r.flags},
                                   "level": "WARNING" if r.verdict in ("REJECTED", "FAIL", "STALE") else "INFO"})
        for p in result.parlays:
            legs = [{"key": l.key, "recommendation_id": key_to_id.get(l.key), "event_id": l.event_id,
                     "event_name": l.event_name, "market": l.market, "side": l.side, "point": l.point,
                     "selection": l.selection, "decimal": l.decimal, "p_win": l.p_win, "teams": list(l.teams),
                     "sport_code": l.sport_code} for l in p["legs"]]
            report = {k: v for k, v in p.items() if k not in ("legs", "candidates")}
            report["probability_bounds"] = list(p["probability_bounds"])
            self.repo.save_recommendation(
                run_id=run_id, kind="parlay", candidate_key="P:" + "+".join(sorted(l.key for l in p["legs"])),
                sport_code=",".join(sorted({l.sport_code for l in p["legs"]})), league="multi",
                event_id=",".join(l.event_id for l in p["legs"]), event_name=" | ".join(l.event_name for l in p["legs"]),
                commence_time=min(c.event.commence_time for c in p["candidates"]), market="parlay",
                selection=" + ".join(l.selection for l in p["legs"]), odds_decimal=p["combined_decimal"],
                odds_retrieved_at=min(c.event.retrieved_at for c in p["candidates"]),
                final_prob=p["probability"], prob_low=p.get("probability_conservative"),
                expected_value=p["expected_value"], expected_value_low=p.get("expected_value_conservative"),
                suggested_stake=p.get("stake", 0.0) if p["status"] == "QUALIFIED" else 0.0,
                consensus_score=p["score"], status=p["status"], verdicts_json={}, report_json=report,
                legs_json=legs, model_version=MODEL_VERSION)
            audit_rows.append({"run_id": run_id, "agent": AUDIT, "action": f"parlay:{p['status']}",
                               "message": f"{p['n_legs']}-leg parlay @ {p['combined_decimal']:.2f}: "
                                          f"{'; '.join(p['reasons']) or 'qualified'}", "payload_json": {}})
        self.repo.audit_many(audit_rows)
