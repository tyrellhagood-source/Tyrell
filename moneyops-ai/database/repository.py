"""Data-access layer. All writes go through here."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from database.models import (Account, AnalysisRun, AuditLog, BacktestRun, Base, BetSlip, LedgerEntry,
                             OddsSnapshot, PaperWager, ProviderStatusRow, Recommendation, SchemaVersion)
from utils.security import redact
from utils.timeutil import ensure_utc, utcnow

SCHEMA_VERSION = 1


def make_engine(database_url: str):
    kwargs = {"connect_args": {"check_same_thread": False}} if database_url.startswith("sqlite") else {}
    return create_engine(database_url, future=True, **kwargs)


class Repository:
    def __init__(self, database_url: str):
        self.engine = make_engine(database_url)
        self.Session = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    # ------------------------------------------------------------- schema
    def init_db(self) -> None:
        Base.metadata.create_all(self.engine)
        with self.session() as s:
            if s.get(SchemaVersion, SCHEMA_VERSION) is None:
                s.add(SchemaVersion(version=SCHEMA_VERSION))

    @contextmanager
    def session(self):
        s = self.Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    # --------------------------------------------------------------- runs
    def create_run(self, model_version: str, settings_json: dict) -> int:
        with self.session() as s:
            run = AnalysisRun(model_version=model_version, settings_json=settings_json)
            s.add(run)
            s.flush()
            return run.id

    def finish_run(self, run_id: int, status: str, summary: dict, coverage: list, errors: list) -> None:
        with self.session() as s:
            run = s.get(AnalysisRun, run_id)
            run.status, run.finished_at = status, utcnow()
            run.summary_json, run.coverage_json = summary, coverage
            run.errors_json = [redact(e) for e in errors]

    def latest_run(self, status: str | None = "COMPLETED") -> AnalysisRun | None:
        with self.session() as s:
            q = select(AnalysisRun).order_by(AnalysisRun.id.desc())
            if status:
                q = q.where(AnalysisRun.status == status)
            return s.scalars(q.limit(1)).first()

    def list_runs(self, limit: int = 50) -> list[AnalysisRun]:
        with self.session() as s:
            return list(s.scalars(select(AnalysisRun).order_by(AnalysisRun.id.desc()).limit(limit)))

    # --------------------------------------------------------------- odds
    def save_odds_snapshots(self, run_id: int, events) -> int:
        n = 0
        with self.session() as s:
            for ev in events:
                for p in ev.prices:
                    s.add(OddsSnapshot(run_id=run_id, event_id=ev.event_id, sport_code=ev.sport_code,
                                       sport_key=ev.sport_key, commence_time=ev.commence_time,
                                       home_team=ev.home_team, away_team=ev.away_team, market=p.market,
                                       bookmaker=p.bookmaker, outcome=p.name, description=p.description,
                                       point=p.point, price_decimal=p.price_decimal, last_update=p.last_update,
                                       retrieved_at=ev.retrieved_at, source=ev.source))
                    n += 1
        return n

    def snapshots_for_run(self, run_id: int) -> list[OddsSnapshot]:
        with self.session() as s:
            return list(s.scalars(select(OddsSnapshot).where(OddsSnapshot.run_id == run_id)))

    def closing_price(self, event_id: str, market: str, outcome: str, point: float | None, bookmaker: str | None,
                      commence_time: datetime, max_gap_minutes: float = 60) -> float | None:
        """Last observed pre-start price (same book, else best across books
        in the final snapshot). Returns None unless the snapshot was taken
        within max_gap_minutes of the start (otherwise not a reliable close)."""
        with self.session() as s:
            q = (select(OddsSnapshot).where(OddsSnapshot.event_id == event_id, OddsSnapshot.market == market,
                                            OddsSnapshot.outcome == outcome,
                                            OddsSnapshot.last_update < commence_time)
                 .order_by(OddsSnapshot.retrieved_at.desc()))
            rows = [r for r in s.scalars(q) if (r.point == point)]
        if not rows:
            return None
        latest = max(ensure_utc(r.retrieved_at) for r in rows)
        if (ensure_utc(commence_time) - latest).total_seconds() / 60 > max_gap_minutes:
            return None
        final = [r for r in rows if ensure_utc(r.retrieved_at) == latest]
        same_book = [r for r in final if r.bookmaker == bookmaker]
        if same_book:
            return same_book[0].price_decimal
        return max(r.price_decimal for r in final)

    # ---------------------------------------------------- recommendations
    def save_recommendation(self, **fields) -> int:
        with self.session() as s:
            rec = Recommendation(**fields)
            s.add(rec)
            s.flush()
            return rec.id

    def get_recommendation(self, rec_id: int) -> Recommendation | None:
        with self.session() as s:
            return s.get(Recommendation, rec_id)

    def list_recommendations(self, run_id: int | None = None, status: str | None = None, kind: str | None = None,
                             pending_only: bool = False, limit: int | None = None) -> list[Recommendation]:
        with self.session() as s:
            q = select(Recommendation).order_by(Recommendation.consensus_score.desc(), Recommendation.id)
            if run_id is not None:
                q = q.where(Recommendation.run_id == run_id)
            if status:
                q = q.where(Recommendation.status == status)
            if kind:
                q = q.where(Recommendation.kind == kind)
            if pending_only:
                q = q.where(Recommendation.result == "pending")
            if limit:
                q = q.limit(limit)
            return list(s.scalars(q))

    def grade_recommendation(self, rec_id: int, result: str, source: str, closing_odds: float | None) -> None:
        with self.session() as s:
            rec = s.get(Recommendation, rec_id)
            if rec.result != "pending":
                return
            rec.result, rec.result_source, rec.settled_at = result, source, utcnow()
            if closing_odds:
                rec.closing_odds = closing_odds

    # -------------------------------------------------------------- audit
    def audit(self, agent: str, action: str, message: str, payload: dict | None = None, run_id: int | None = None,
              candidate_key: str | None = None, level: str = "INFO") -> None:
        with self.session() as s:
            s.add(AuditLog(run_id=run_id, agent=agent, action=action, message=redact(message),
                           payload_json=payload or {}, candidate_key=candidate_key, level=level))

    def audit_many(self, rows: list[dict]) -> None:
        with self.session() as s:
            for r in rows:
                r = dict(r)
                r["message"] = redact(r.get("message", ""))
                s.add(AuditLog(**r))

    def list_audit(self, run_id: int | None = None, agent: str | None = None, limit: int = 500) -> list[AuditLog]:
        with self.session() as s:
            q = select(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
            if run_id is not None:
                q = q.where(AuditLog.run_id == run_id)
            if agent:
                q = q.where(AuditLog.agent == agent)
            return list(s.scalars(q))

    # ---------------------------------------------------------- providers
    def save_provider_status(self, status) -> None:
        with self.session() as s:
            row = s.get(ProviderStatusRow, status.name) or ProviderStatusRow(name=status.name)
            row.updated_at = utcnow()
            row.configured = status.configured
            row.last_success = status.last_success or row.last_success
            if status.last_error:
                row.last_error, row.last_error_at = redact(status.last_error), status.last_error_at
            row.requests_made = (row.requests_made or 0) + status.requests_made
            row.cache_hits = (row.cache_hits or 0) + status.cache_hits
            if status.quota_remaining is not None:
                row.quota_remaining = status.quota_remaining
            if status.quota_used is not None:
                row.quota_used = status.quota_used
            if status.last_request_cost is not None:
                row.last_request_cost = status.last_request_cost
            s.merge(row)

    def list_provider_status(self) -> list[ProviderStatusRow]:
        with self.session() as s:
            return list(s.scalars(select(ProviderStatusRow)))

    # ----------------------------------------------------------- backtests
    def save_backtest(self, sport_code: str, model_version: str, status: str, metrics: dict) -> int:
        with self.session() as s:
            row = BacktestRun(sport_code=sport_code, model_version=model_version, status=status, metrics_json=metrics)
            s.add(row)
            s.flush()
            return row.id

    def latest_backtests(self) -> dict[str, BacktestRun]:
        with self.session() as s:
            rows = list(s.scalars(select(BacktestRun).order_by(BacktestRun.id.desc())))
        out = {}
        for r in rows:
            out.setdefault(r.sport_code, r)
        return out

    # ------------------------------------------------- accounts / ledger
    def active_account(self, starting_balance: float) -> Account:
        with self.session() as s:
            acct = s.scalars(select(Account).where(Account.status == "ACTIVE").order_by(Account.id.desc())).first()
            if acct is None:
                acct = Account(starting_balance=starting_balance)
                s.add(acct)
                s.flush()
                s.add(LedgerEntry(account_id=acct.id, entry_type="VIRTUAL_DEPOSIT", amount=starting_balance,
                                  balance_after=starting_balance, note="Simulated starting balance (not real money)"))
            return acct

    def archive_account(self, account_id: int) -> None:
        with self.session() as s:
            acct = s.get(Account, account_id)
            acct.status, acct.archived_at = "ARCHIVED", utcnow()

    def list_accounts(self) -> list[Account]:
        with self.session() as s:
            return list(s.scalars(select(Account).order_by(Account.id)))

    def ledger(self, account_id: int) -> list[LedgerEntry]:
        with self.session() as s:
            return list(s.scalars(select(LedgerEntry).where(LedgerEntry.account_id == account_id)
                                  .order_by(LedgerEntry.id)))

    def balance(self, account_id: int) -> float:
        entries = self.ledger(account_id)
        return round(sum(e.amount for e in entries), 2)

    def slips(self, account_id: int, status: str | None = None) -> list[BetSlip]:
        with self.session() as s:
            q = select(BetSlip).where(BetSlip.account_id == account_id).order_by(BetSlip.id.desc())
            if status:
                q = q.where(BetSlip.status == status)
            return list(s.scalars(q))

    def wagers(self, account_id: int | None = None, status: str | None = None) -> list[PaperWager]:
        with self.session() as s:
            q = select(PaperWager).order_by(PaperWager.id)
            if account_id is not None:
                q = q.where(PaperWager.account_id == account_id)
            if status:
                q = q.where(PaperWager.status == status)
            return list(s.scalars(q))
