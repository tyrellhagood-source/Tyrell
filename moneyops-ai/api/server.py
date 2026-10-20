"""FastAPI backend. Run: uvicorn api.server:app --port 8000

Paper trading only: no endpoint moves real money or talks to a sportsbook.
API keys never leave the server (settings are returned masked).
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from config import APP_VERSION, MODEL_VERSION, get_settings
from risk.bankroll import BankrollError, BankrollService
from risk.risk_limits import RiskLimitError
from services.analysis import AnalysisService
from services.queries import custom_parlay, performance_report, wager_performance
from services.runtime import get_repo
from services.settlement import SettlementService

app = FastAPI(title="MoneyOps AI", version=APP_VERSION,
              description="Five-agent sports betting RESEARCH system with a $25 VIRTUAL paper bankroll.")


def _rec(r) -> dict:
    return {c.name: getattr(r, c.name) for c in r.__table__.columns}


class SlipIn(BaseModel):
    recommendation_id: int
    stake: float | None = None


class SettleIn(BaseModel):
    result: str
    note: str


class ResetIn(BaseModel):
    confirmation: str


class ParlayIn(BaseModel):
    recommendation_ids: list[int]
    stake: float = 1.0


@app.get("/health")
def health():
    s = get_settings()
    return {"status": "ok", "version": APP_VERSION, "model_version": MODEL_VERSION,
            "odds_api_key": "configured" if s.odds_api_key else "missing", "paper_trading_only": True}


@app.get("/settings")
def settings():
    return get_settings().public_dict()


@app.post("/analysis/run")
def run_analysis():
    res = AnalysisService(get_settings(), get_repo()).run()
    return {"run_id": res.run_id, "status": res.status, "message": res.message, "errors": res.errors,
            "coverage": res.coverage}


@app.get("/runs")
def runs(limit: int = 20):
    return [_rec(r) for r in get_repo().list_runs(limit)]


@app.get("/recommendations")
def recommendations(run_id: int | None = None, status: str | None = None, kind: str | None = None, limit: int = 200):
    repo = get_repo()
    if run_id is None:
        latest = repo.latest_run()
        run_id = latest.id if latest else None
    if run_id is None:
        return {"message": "No completed analysis run yet.", "items": []}
    items = [_rec(r) for r in repo.list_recommendations(run_id, status, kind, limit=limit)]
    if status == "QUALIFIED" and not items:
        return {"message": "NO QUALIFYING BETS FOUND.", "items": []}
    return {"run_id": run_id, "items": items}


@app.post("/parlays/evaluate")
def parlay(body: ParlayIn):
    try:
        res = custom_parlay(get_repo(), body.recommendation_ids, body.stake)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    res["probability_bounds"] = list(res["probability_bounds"])
    return res


@app.get("/bankroll")
def bankroll():
    repo = get_repo()
    b = BankrollService(repo, get_settings())
    s = b.summary()
    return s | {"performance": {k: v for k, v in wager_performance(repo, s["account_id"]).items()
                                if k != "calibration"}}


@app.get("/wagers")
def wagers(status: str | None = None):
    return [_rec(w) for w in get_repo().wagers(status=status)]


@app.post("/slips")
def propose(body: SlipIn):
    try:
        return {"slip_id": BankrollService(get_repo(), get_settings()).propose(body.recommendation_id, body.stake)}
    except BankrollError as exc:
        raise HTTPException(400, str(exc))


@app.post("/slips/{slip_id}/approve")
def approve(slip_id: int):
    try:
        return {"wager_id": BankrollService(get_repo(), get_settings()).approve(slip_id)}
    except (BankrollError, RiskLimitError) as exc:
        raise HTTPException(400, str(exc))


@app.post("/slips/{slip_id}/decline")
def decline(slip_id: int):
    try:
        BankrollService(get_repo(), get_settings()).decline(slip_id)
        return {"ok": True}
    except BankrollError as exc:
        raise HTTPException(400, str(exc))


@app.post("/wagers/settle-verified")
def settle_verified():
    svc = AnalysisService(get_settings(), get_repo())
    st = SettlementService(get_settings(), get_repo(), svc.espn, svc.odds, svc.allsports, svc.player_stats)
    return {"wagers": st.settle_open_wagers(), "recommendations": st.grade_recommendations()}


@app.post("/wagers/{wager_id}/settle-manual")
def settle_manual(wager_id: int, body: SettleIn):
    if not body.note.strip():
        raise HTTPException(400, "a note describing the verified source is required")
    try:
        return BankrollService(get_repo(), get_settings()).settle(wager_id, body.result,
                                                                  f"MANUAL (unverified by system): {body.note}")
    except BankrollError as exc:
        raise HTTPException(400, str(exc))


@app.post("/bankroll/reset")
def reset(body: ResetIn):
    try:
        return {"account_id": BankrollService(get_repo(), get_settings()).reset(body.confirmation)}
    except BankrollError as exc:
        raise HTTPException(400, str(exc))


@app.get("/export/{what}", response_class=PlainTextResponse)
def export(what: str):
    try:
        return BankrollService(get_repo(), get_settings()).export_csv(what)
    except BankrollError as exc:
        raise HTTPException(400, str(exc))


@app.get("/performance")
def performance():
    rep = performance_report(get_repo())
    return rep


@app.get("/providers")
def providers():
    return [_rec(p) for p in get_repo().list_provider_status()]


@app.get("/audit")
def audit(run_id: int | None = None, agent: str | None = None, limit: int = 200):
    return [_rec(a) for a in get_repo().list_audit(run_id, agent, limit)]
