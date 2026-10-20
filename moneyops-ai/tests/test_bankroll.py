from datetime import datetime, timedelta, timezone

import pytest

from database.models import ImmutableRecordError, LedgerEntry, PaperWager, Recommendation
from risk.bankroll import RESET_PHRASE, BankrollError, BankrollService, parlay_outcome
from risk.position_sizing import suggested_stake
from risk.risk_limits import RiskLimitError

NOW = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)


def make_rec(repo, run_id=None, odds=2.0, stake=0.25, kind="single", start=NOW + timedelta(hours=3), legs=None):
    if run_id is None:
        run_id = repo.create_run("test", {})
    return repo.save_recommendation(run_id=run_id, kind=kind, candidate_key=f"k{odds}{stake}{kind}",
                                    sport_code="NBA", league="NBA", event_id="e1", event_name="A @ B",
                                    home_team="B", away_team="A", commence_time=start, market="h2h",
                                    selection="B", side="home", odds_decimal=odds, final_prob=0.55,
                                    expected_value=0.1, suggested_stake=stake, status="QUALIFIED",
                                    odds_retrieved_at=NOW - timedelta(minutes=5), model_version="t",
                                    legs_json=legs or [])


def test_starting_balance_is_25_virtual(repo, settings):
    b = BankrollService(repo, settings)
    s = b.summary()
    assert s["balance"] == 25.0 and s["equity"] == 25.0 and "NOT REAL MONEY" in s["label"]
    assert settings.max_stake_pct * 25 == 0.25


def test_proposal_does_not_touch_balance(repo, settings):
    b = BankrollService(repo, settings)
    slip = b.propose(make_rec(repo), now=NOW)
    assert b.summary()["balance"] == 25.0
    b.decline(slip)
    assert b.summary()["balance"] == 25.0 and b.summary()["open_wagers"] == 0


def test_approve_and_settle_win(repo, settings):
    b = BankrollService(repo, settings)
    wid = b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    s = b.summary()
    assert s["balance"] == 24.75 and s["open_exposure"] == 0.25 and s["equity"] == 25.0
    out = b.settle(wid, "win", "test")
    assert out["payout"] == 0.50 and out["profit"] == 0.25 and out["bankroll_after"] == 25.25
    w = repo.wagers()[0]
    assert w.bankroll_before == 25.0 and w.bankroll_after == 25.25


@pytest.mark.parametrize("result,expected", [("loss", 24.75), ("push", 25.0), ("void", 25.0)])
def test_settle_loss_push_void(repo, settings, result, expected):
    b = BankrollService(repo, settings)
    wid = b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    b.settle(wid, result, "test")
    assert b.summary()["balance"] == expected
    with pytest.raises(BankrollError):
        b.settle(wid, "win", "again")  # settlement is final


def test_stake_capped_at_one_percent(repo, settings):
    b = BankrollService(repo, settings)
    slip = b.propose(make_rec(repo, stake=5.0), now=NOW)  # recommendation asks for too much
    assert repo.slips(b.account().id)[0].stake == 0.25
    with pytest.raises(RiskLimitError):
        b.approve(b.propose(make_rec(repo, odds=2.1), stake=1.00, now=NOW), now=NOW)


def test_cannot_approve_started_or_stale(repo, settings):
    b = BankrollService(repo, settings)
    slip = b.propose(make_rec(repo), now=NOW)
    with pytest.raises(BankrollError):
        b.approve(slip, now=NOW + timedelta(hours=4))  # after start
    with pytest.raises(BankrollError):
        b.approve(slip, now=NOW + timedelta(minutes=40))  # odds stale
    with pytest.raises(BankrollError):
        b.propose(make_rec(repo, start=NOW - timedelta(minutes=1)), now=NOW)


def test_exposure_limit(repo, settings):
    s = settings.with_overrides(max_open_exposure_pct=0.015)
    b = BankrollService(repo, s)
    b.approve(b.propose(make_rec(repo, odds=2.0), now=NOW), now=NOW)
    with pytest.raises(RiskLimitError):
        b.approve(b.propose(make_rec(repo, odds=2.2), now=NOW), now=NOW)


def test_ledger_and_wager_immutable(repo, settings):
    b = BankrollService(repo, settings)
    wid = b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    with pytest.raises(ImmutableRecordError):
        with repo.session() as s:
            s.get(PaperWager, wid).stake = 100.0
    with pytest.raises(ImmutableRecordError):
        with repo.session() as s:
            s.delete(s.get(PaperWager, wid))
    with pytest.raises(ImmutableRecordError):
        with repo.session() as s:
            s.query(LedgerEntry).first().amount = 1000
    with pytest.raises(ImmutableRecordError):
        with repo.session() as s:
            s.delete(s.query(Recommendation).first())
    b.settle(wid, "loss", "t")
    with pytest.raises(ImmutableRecordError):
        with repo.session() as s:
            s.get(PaperWager, wid).result = "win"


def test_reset_requires_confirmation_and_keeps_history(repo, settings):
    b = BankrollService(repo, settings)
    wid = b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    with pytest.raises(BankrollError):
        b.reset(RESET_PHRASE)  # open wager
    b.settle(wid, "loss", "t")
    with pytest.raises(BankrollError):
        b.reset("yes")
    new_id = b.reset(RESET_PHRASE)
    assert b.summary()["balance"] == 25.0 and b.summary()["account_id"] == new_id
    assert len(repo.wagers()) == 1 and len(repo.list_accounts()) == 2


def test_parlay_settlement_with_push_leg():
    legs = [{"decimal": 2.0}, {"decimal": 1.5}, {"decimal": 3.0}]
    assert parlay_outcome(legs, ["win", "push", "win"]) == ("win", 6.0)
    assert parlay_outcome(legs, ["win", "loss", "push"]) == ("loss", 0.0)
    assert parlay_outcome(legs, ["push", "void", "push"]) == ("void", 1.0)


def test_csv_export(repo, settings):
    b = BankrollService(repo, settings)
    b.approve(b.propose(make_rec(repo), now=NOW), now=NOW)
    csv_text = b.export_csv("wagers")
    assert csv_text.splitlines()[0].startswith("id,account_id") and len(csv_text.splitlines()) == 2
    assert "VIRTUAL_DEPOSIT" in b.export_csv("ledger")
    assert "QUALIFIED" in b.export_csv("recommendations")


def test_no_stake_increase_to_chase_losses():
    after_loss = suggested_stake(0.55, 2.0, 20.0, 0.01, 0.25, 0.10)
    before = suggested_stake(0.55, 2.0, 25.0, 0.01, 0.25, 0.10)
    assert after_loss["stake"] < before["stake"]
