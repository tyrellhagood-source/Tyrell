"""SQLAlchemy ORM models.

Integrity rules enforced at the ORM layer:
* LedgerEntry rows are append-only (no update, no delete).
* PaperWager acceptance fields are immutable; only settlement fields may be
  written, and only once (OPEN -> SETTLED). Wagers cannot be deleted.
* Recommendation rows (including rejected and losing picks) cannot be deleted.
"""
from __future__ import annotations

from datetime import datetime

from datetime import timezone

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, TypeDecorator, event,
                        inspect)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from utils.timeutil import utcnow


class UTCDateTime(TypeDecorator):
    """Stores UTC; always returns timezone-aware UTC datetimes (SQLite drops tzinfo)."""
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


class Base(DeclarativeBase):
    pass


class ImmutableRecordError(Exception):
    pass


def _dt():
    return mapped_column(UTCDateTime(), default=utcnow)


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = _dt()


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = _dt()
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="RUNNING")
    model_version: Mapped[str] = mapped_column(String(64))
    settings_json: Mapped[dict] = mapped_column(JSON, default=dict)
    summary_json: Mapped[dict] = mapped_column(JSON, default=dict)
    coverage_json: Mapped[list] = mapped_column(JSON, default=list)
    errors_json: Mapped[list] = mapped_column(JSON, default=list)


class OddsSnapshot(Base):
    __tablename__ = "odds_snapshots"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    event_id: Mapped[str] = mapped_column(String(64), index=True)
    sport_code: Mapped[str] = mapped_column(String(64))
    sport_key: Mapped[str] = mapped_column(String(64))
    commence_time: Mapped[datetime] = mapped_column(UTCDateTime())
    home_team: Mapped[str] = mapped_column(String(128))
    away_team: Mapped[str] = mapped_column(String(128))
    market: Mapped[str] = mapped_column(String(64))
    bookmaker: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(128))
    description: Mapped[str | None] = mapped_column(String(128), nullable=True)
    point: Mapped[float | None] = mapped_column(Float, nullable=True)
    price_decimal: Mapped[float] = mapped_column(Float)
    last_update: Mapped[datetime] = mapped_column(UTCDateTime())
    retrieved_at: Mapped[datetime] = mapped_column(UTCDateTime())
    source: Mapped[str] = mapped_column(String(64))


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    created_at: Mapped[datetime] = _dt()
    kind: Mapped[str] = mapped_column(String(16))           # single | parlay
    candidate_key: Mapped[str] = mapped_column(String(128), index=True)
    sport_code: Mapped[str] = mapped_column(String(64))
    league: Mapped[str] = mapped_column(String(128))
    event_id: Mapped[str] = mapped_column(String(256))
    event_name: Mapped[str] = mapped_column(String(512))
    home_team: Mapped[str | None] = mapped_column(String(128), nullable=True)
    away_team: Mapped[str | None] = mapped_column(String(128), nullable=True)
    sport_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    commence_time: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)
    market: Mapped[str] = mapped_column(String(64))
    selection: Mapped[str] = mapped_column(String(512))
    side: Mapped[str | None] = mapped_column(String(16), nullable=True)
    point: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_book: Mapped[str | None] = mapped_column(String(64), nullable=True)
    odds_decimal: Mapped[float | None] = mapped_column(Float, nullable=True)
    odds_last_update: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    odds_retrieved_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    model_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_implied_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    fair_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    prob_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    push_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_value_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    hist_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    hist_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    suggested_stake: Mapped[float] = mapped_column(Float, default=0.0)
    consensus_score: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), index=True)   # QUALIFIED | NEEDS_RESEARCH | REJECTED
    verdicts_json: Mapped[dict] = mapped_column(JSON, default=dict)
    report_json: Mapped[dict] = mapped_column(JSON, default=dict)
    legs_json: Mapped[list] = mapped_column(JSON, default=list)
    model_version: Mapped[str] = mapped_column(String(64))
    result: Mapped[str] = mapped_column(String(16), default="pending")
    result_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)


class Account(Base):
    __tablename__ = "accounts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = _dt()
    starting_balance: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="ACTIVE")
    archived_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    note: Mapped[str] = mapped_column(String(256), default="Virtual paper-trading account (no real money)")


class BetSlip(Base):
    """A PROPOSED paper wager. Never counted in the bankroll until APPROVED."""
    __tablename__ = "bet_slips"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"))
    recommendation_id: Mapped[int] = mapped_column(ForeignKey("recommendations.id"))
    created_at: Mapped[datetime] = _dt()
    stake: Mapped[float] = mapped_column(Float)
    odds_decimal: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="PROPOSED")  # PROPOSED | APPROVED | DECLINED
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    note: Mapped[str | None] = mapped_column(String(512), nullable=True)


class PaperWager(Base):
    __tablename__ = "paper_wagers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    slip_id: Mapped[int | None] = mapped_column(ForeignKey("bet_slips.id"), nullable=True)
    recommendation_id: Mapped[int] = mapped_column(ForeignKey("recommendations.id"))
    placed_at: Mapped[datetime] = _dt()
    kind: Mapped[str] = mapped_column(String(16))
    sport_code: Mapped[str] = mapped_column(String(64))
    event_name: Mapped[str] = mapped_column(String(512))
    market: Mapped[str] = mapped_column(String(64))
    selection: Mapped[str] = mapped_column(String(512))
    odds_decimal: Mapped[float] = mapped_column(Float)
    odds_source: Mapped[str] = mapped_column(String(128))
    odds_timestamp: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    stake: Mapped[float] = mapped_column(Float)
    est_prob: Mapped[float | None] = mapped_column(Float, nullable=True)
    expected_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    bankroll_before: Mapped[float] = mapped_column(Float)
    legs_json: Mapped[list] = mapped_column(JSON, default=list)
    # settlement fields (written once)
    status: Mapped[str] = mapped_column(String(16), default="OPEN")  # OPEN | SETTLED
    result: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payout: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    bankroll_after: Mapped[float | None] = mapped_column(Float, nullable=True)
    closing_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    settlement_source: Mapped[str | None] = mapped_column(String(256), nullable=True)


SETTLEMENT_FIELDS = {"status", "result", "payout", "profit", "bankroll_after", "closing_odds", "settled_at",
                     "settlement_source"}


class LedgerEntry(Base):
    __tablename__ = "ledger"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.id"), index=True)
    wager_id: Mapped[int | None] = mapped_column(ForeignKey("paper_wagers.id"), nullable=True)
    ts: Mapped[datetime] = _dt()
    entry_type: Mapped[str] = mapped_column(String(32))  # VIRTUAL_DEPOSIT | STAKE | PAYOUT | REFUND
    amount: Mapped[float] = mapped_column(Float)
    balance_after: Mapped[float] = mapped_column(Float)
    note: Mapped[str] = mapped_column(String(512), default="")


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    ts: Mapped[datetime] = _dt()
    agent: Mapped[str] = mapped_column(String(64), index=True)
    level: Mapped[str] = mapped_column(String(16), default="INFO")
    action: Mapped[str] = mapped_column(String(64))
    candidate_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    message: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)


class ProviderStatusRow(Base):
    __tablename__ = "provider_status"
    name: Mapped[str] = mapped_column(String(64), primary_key=True)
    updated_at: Mapped[datetime] = _dt()
    configured: Mapped[bool] = mapped_column(Boolean, default=False)
    last_success: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    last_error_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    requests_made: Mapped[int] = mapped_column(Integer, default=0)
    cache_hits: Mapped[int] = mapped_column(Integer, default=0)
    quota_remaining: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quota_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_request_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)


class ApiCache(Base):
    __tablename__ = "api_cache"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64))
    url: Mapped[str] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text)
    headers: Mapped[str] = mapped_column(Text, default="{}")
    fetched_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime())


class BacktestRun(Base):
    """HYPOTHETICAL model evaluation. Never mixed with paper-trading results."""
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = _dt()
    sport_code: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)


# --------------------------------------------------------------------------
# Integrity enforcement
# --------------------------------------------------------------------------
@event.listens_for(LedgerEntry, "before_update")
def _ledger_no_update(mapper, connection, target):
    raise ImmutableRecordError("Ledger entries are append-only")


@event.listens_for(LedgerEntry, "before_delete")
def _ledger_no_delete(mapper, connection, target):
    raise ImmutableRecordError("Ledger entries cannot be deleted")


@event.listens_for(PaperWager, "before_delete")
def _wager_no_delete(mapper, connection, target):
    raise ImmutableRecordError("Accepted paper wagers cannot be deleted")


@event.listens_for(PaperWager, "before_update")
def _wager_settlement_only(mapper, connection, target):
    state = inspect(target)
    for attr in state.attrs:
        hist = attr.load_history()
        if not hist.has_changes():
            continue
        if attr.key not in SETTLEMENT_FIELDS:
            raise ImmutableRecordError(f"PaperWager.{attr.key} is immutable after acceptance")
    status_hist = state.attrs.status.load_history()
    previous = (status_hist.deleted or status_hist.unchanged or [None])[0]
    if previous == "SETTLED":
        raise ImmutableRecordError("Wager already settled; settlement is final")


@event.listens_for(Recommendation, "before_delete")
def _rec_no_delete(mapper, connection, target):
    raise ImmutableRecordError("Recommendations (including losing and rejected picks) are never deleted")
