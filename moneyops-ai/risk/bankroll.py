"""The $25 VIRTUAL bankroll (paper trading only).

No real money is ever involved: there is no payment, bank or sportsbook
integration anywhere in this code base. The ledger is append-only, and the
balance is always the sum of ledger entries.

Lifecycle: Recommendation -> BetSlip (PROPOSED) -> APPROVED creates a
PaperWager (OPEN) + STAKE ledger debit -> settlement (WIN/LOSS/PUSH/VOID)
writes PAYOUT/REFUND credit once. Proposed or declined slips never touch
the balance.
"""
from __future__ import annotations

import csv
import io
from datetime import timedelta

from database.models import BetSlip, LedgerEntry, PaperWager, Recommendation
from risk.risk_limits import RiskLimitError, check_new_wager
from utils.timeutil import ensure_utc, utcnow

RESET_PHRASE = "RESET PAPER ACCOUNT"
RESULTS = ("win", "loss", "push", "void")


class BankrollError(Exception):
    pass


class BankrollService:
    def __init__(self, repo, settings):
        self.repo, self.s = repo, settings

    # -------------------------------------------------------------- views
    def account(self):
        return self.repo.active_account(self.s.starting_bankroll)

    def equity_curve(self, account_id: int) -> list[dict]:
        cash, open_stakes, curve = 0.0, {}, []
        for e in self.repo.ledger(account_id):
            cash += e.amount
            if e.entry_type == "STAKE":
                open_stakes[e.wager_id] = -e.amount
            elif e.entry_type in ("PAYOUT", "REFUND", "SETTLE_LOSS"):
                open_stakes.pop(e.wager_id, None)
            curve.append({"ts": e.ts, "cash": round(cash, 2), "equity": round(cash + sum(open_stakes.values()), 2),
                          "type": e.entry_type})
        return curve

    def summary(self) -> dict:
        acct = self.account()
        balance = self.repo.balance(acct.id)
        open_w = self.repo.wagers(acct.id, "OPEN")
        exposure = round(sum(w.stake for w in open_w), 2)
        curve = self.equity_curve(acct.id)
        peak, max_dd = acct.starting_balance, 0.0
        for pt in curve:
            peak = max(peak, pt["equity"])
            max_dd = max(max_dd, (peak - pt["equity"]) / peak if peak else 0)
        settled = self.repo.wagers(acct.id, "SETTLED")
        return {"account_id": acct.id, "starting_balance": acct.starting_balance, "balance": balance,
                "open_exposure": exposure, "equity": round(balance + exposure, 2), "peak_equity": round(peak, 2),
                "max_drawdown_pct": max_dd, "open_wagers": len(open_w), "settled_wagers": len(settled),
                "realized_profit": round(sum(w.profit or 0 for w in settled), 2),
                "label": "SIMULATED PAPER BALANCE - NOT REAL MONEY"}

    # ------------------------------------------------------------ actions
    def propose(self, recommendation_id: int, stake: float | None = None, now=None) -> int:
        now = now or utcnow()
        rec: Recommendation = self.repo.get_recommendation(recommendation_id)
        if rec is None:
            raise BankrollError("recommendation not found")
        if rec.odds_decimal is None:
            raise BankrollError("recommendation has no price")
        if rec.commence_time and ensure_utc(rec.commence_time) <= now:
            raise BankrollError("event has already started; cannot propose a pre-game paper wager")
        summary = self.summary()
        cap = self.s.parlay_max_stake_pct if rec.kind == "parlay" else self.s.max_stake_pct
        max_allowed = round(summary["balance"] * cap, 2)
        stake = round(stake if stake is not None else min(rec.suggested_stake or 0.0, max_allowed), 2)
        if stake <= 0:
            raise BankrollError("no positive stake for this recommendation (not qualified or below minimum)")
        with self.repo.session() as s:
            slip = BetSlip(account_id=summary["account_id"], recommendation_id=rec.id, stake=stake,
                           odds_decimal=rec.odds_decimal, status="PROPOSED")
            s.add(slip)
            s.flush()
            slip_id = slip.id
        self.repo.audit("bankroll", "slip_proposed", f"Slip {slip_id}: ${stake:.2f} on {rec.selection} @ "
                        f"{rec.odds_decimal:.2f}", {"recommendation_id": rec.id})
        return slip_id

    def decline(self, slip_id: int, note: str = "") -> None:
        with self.repo.session() as s:
            slip = s.get(BetSlip, slip_id)
            if slip is None or slip.status != "PROPOSED":
                raise BankrollError("slip not found or already decided")
            slip.status, slip.decided_at, slip.note = "DECLINED", utcnow(), note
        self.repo.audit("bankroll", "slip_declined", f"Slip {slip_id} declined", {"note": note})

    def approve(self, slip_id: int, now=None) -> int:
        now = now or utcnow()
        acct = self.account()
        with self.repo.session() as s:
            slip = s.get(BetSlip, slip_id)
            if slip is None or slip.status != "PROPOSED":
                raise BankrollError("slip not found or already decided")
            if slip.account_id != acct.id:
                raise BankrollError("slip belongs to an archived account")
            rec = s.get(Recommendation, slip.recommendation_id)
            if rec.commence_time and ensure_utc(rec.commence_time) <= now:
                raise BankrollError("event already started; the quoted pre-game price is no longer available")
            if rec.odds_retrieved_at and now - ensure_utc(rec.odds_retrieved_at) > timedelta(
                    minutes=self.s.stale_odds_minutes_pregame):
                raise BankrollError("odds on this recommendation are stale; re-run analysis to refresh before approving")
            summary = self.summary()
            rc = check_new_wager(slip.stake, summary["balance"], summary["open_exposure"], summary["peak_equity"],
                                 self.s, rec.kind == "parlay")
            if not rc.passed:
                raise RiskLimitError("; ".join(rc.reasons))
            before = summary["balance"]
            wager = PaperWager(account_id=acct.id, slip_id=slip.id, recommendation_id=rec.id, kind=rec.kind,
                               sport_code=rec.sport_code, event_name=rec.event_name, market=rec.market,
                               selection=rec.selection, odds_decimal=slip.odds_decimal,
                               odds_source=rec.best_book or ("multiple books" if rec.kind == "parlay" else "unknown"),
                               odds_timestamp=rec.odds_last_update or rec.odds_retrieved_at, stake=slip.stake,
                               est_prob=rec.final_prob, expected_value=rec.expected_value, bankroll_before=before,
                               legs_json=rec.legs_json or [])
            s.add(wager)
            s.flush()
            s.add(LedgerEntry(account_id=acct.id, wager_id=wager.id, entry_type="STAKE", amount=-slip.stake,
                              balance_after=round(before - slip.stake, 2),
                              note=f"Paper stake on {rec.selection} @ {slip.odds_decimal:.2f}"))
            slip.status, slip.decided_at = "APPROVED", now
            wager_id = wager.id
        self.repo.audit("bankroll", "wager_accepted", f"Paper wager {wager_id} accepted (${slip.stake:.2f})",
                        {"slip_id": slip_id})
        return wager_id

    def settle(self, wager_id: int, result: str, source: str, closing_odds: float | None = None,
               leg_results: list[str] | None = None) -> dict:
        """Settle once. For parlays pass leg_results; pushed/void legs are
        removed and the payout recomputed from the remaining legs."""
        if result not in RESULTS and leg_results is None:
            raise BankrollError(f"result must be one of {RESULTS}")
        with self.repo.session() as s:
            w = s.get(PaperWager, wager_id)
            if w is None:
                raise BankrollError("wager not found")
            if w.status != "OPEN":
                raise BankrollError("wager already settled")
            payout_decimal = w.odds_decimal
            if w.kind == "parlay" and leg_results is not None:
                result, payout_decimal = parlay_outcome(w.legs_json, leg_results)
            if result == "win":
                payout = round(w.stake * payout_decimal, 2)
                entry_type = "PAYOUT"
            elif result in ("push", "void"):
                payout, entry_type = w.stake, "REFUND"
            else:
                payout, entry_type = 0.0, "SETTLE_LOSS"
            cash = round(sum(e.amount for e in s.query(LedgerEntry).filter_by(account_id=w.account_id)), 2)
            after = round(cash + payout, 2)
            s.add(LedgerEntry(account_id=w.account_id, wager_id=w.id, entry_type=entry_type, amount=payout,
                              balance_after=after, note=f"Settled {result.upper()} via {source}"))
            w.status, w.result, w.payout = "SETTLED", result, payout
            w.profit = round(payout - w.stake, 2)
            w.bankroll_after, w.settled_at, w.settlement_source = after, utcnow(), source
            w.closing_odds = closing_odds
            out = {"wager_id": w.id, "result": result, "payout": payout, "profit": w.profit, "bankroll_after": after}
        self.repo.audit("bankroll", "wager_settled", f"Wager {wager_id} settled {result} ({source})", out)
        return out

    def reset(self, confirmation: str) -> int:
        if confirmation.strip() != RESET_PHRASE:
            raise BankrollError(f'type "{RESET_PHRASE}" to confirm the reset')
        acct = self.account()
        if self.repo.wagers(acct.id, "OPEN"):
            raise BankrollError("settle or wait for open wagers before resetting")
        self.repo.archive_account(acct.id)
        new = self.account()
        self.repo.audit("bankroll", "account_reset", f"Account {acct.id} archived; new account {new.id} with "
                        f"${self.s.starting_bankroll:.2f} virtual balance")
        return new.id

    # ------------------------------------------------------------- export
    def export_csv(self, what: str = "wagers") -> str:
        buf = io.StringIO()
        if what == "wagers":
            rows = self.repo.wagers()
            cols = ["id", "account_id", "placed_at", "kind", "sport_code", "event_name", "market", "selection",
                    "odds_decimal", "odds_source", "odds_timestamp", "stake", "est_prob", "expected_value",
                    "bankroll_before", "status", "result", "payout", "profit", "bankroll_after", "closing_odds",
                    "settled_at", "settlement_source"]
        elif what == "ledger":
            rows = [e for a in self.repo.list_accounts() for e in self.repo.ledger(a.id)]
            cols = ["id", "account_id", "wager_id", "ts", "entry_type", "amount", "balance_after", "note"]
        elif what == "recommendations":
            rows = self.repo.list_recommendations()
            cols = ["id", "run_id", "created_at", "kind", "status", "sport_code", "league", "event_name",
                    "commence_time", "market", "selection", "best_book", "odds_decimal", "odds_last_update",
                    "model_prob", "fair_prob", "final_prob", "prob_low", "prob_high", "expected_value",
                    "hist_rate", "hist_n", "suggested_stake", "consensus_score", "model_version", "result",
                    "result_source", "closing_odds"]
        else:
            raise BankrollError("unknown export")
        writer = csv.writer(buf)
        writer.writerow(cols)
        for r in rows:
            writer.writerow([getattr(r, c) for c in cols])
        return buf.getvalue()


def parlay_outcome(legs: list[dict], leg_results: list[str]) -> tuple[str, float]:
    if len(legs) != len(leg_results):
        raise BankrollError("leg_results must match the number of legs")
    if any(r == "loss" for r in leg_results):
        return "loss", 0.0
    if any(r not in RESULTS for r in leg_results):
        raise BankrollError("invalid leg result")
    remaining = [l for l, r in zip(legs, leg_results) if r == "win"]
    if not remaining:
        return "void", 1.0
    d = 1.0
    for l in remaining:
        d *= float(l["decimal"])
    return "win", d

