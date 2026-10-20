import pandas as pd
import streamlit as st

from dashboard.charts import line_chart
from dashboard.components import md_money, money, odds_str, paper_banner, pct, repo, settings
from risk.bankroll import RESET_PHRASE, BankrollError, BankrollService
from risk.risk_limits import RiskLimitError
from services.analysis import build_providers
from services.queries import wager_performance
from services.settlement import SettlementService
from utils.timeutil import to_local

st.title("💵 Virtual Bankroll")
st.error("SIMULATED $25 PAPER BALANCE. Not real money. This app never accesses bank or sportsbook accounts, "
         "never deposits, withdraws or places real wagers.")
s, r = settings(), repo()
bank = BankrollService(r, s)
summ = bank.summary()
c = st.columns(6)
c[0].metric("Cash balance", money(summ["balance"]))
c[1].metric("Open exposure", money(summ["open_exposure"]))
c[2].metric("Equity", money(summ["equity"]), f"{summ['equity'] - summ['starting_balance']:+.2f} vs start")
c[3].metric("Peak equity", money(summ["peak_equity"]))
c[4].metric("Max drawdown", pct(summ["max_drawdown_pct"]))
c[5].metric("Per-wager limit", money(summ["balance"] * s.max_stake_pct), f"{s.max_stake_pct:.0%} of balance",
            delta_color="off")
st.caption(f"Typical sportsbook minimums are around \\${s.typical_sportsbook_min_bet:.2f}; at a {s.max_stake_pct:.0%} limit on {md_money(s.starting_bankroll)} "
           f"the paper stake ({md_money(s.starting_bankroll * s.max_stake_pct)}) is below what most real books accept. This is a research simulation.")

st.subheader("Proposed slips (not yet counted in the balance)")
slips = r.slips(summ["account_id"], "PROPOSED")
if not slips:
    st.caption("No proposed slips. Create them from Top Consensus Picks or Parlay Lab.")
for sl in slips:
    rec = r.get_recommendation(sl.recommendation_id)
    cc = st.columns([5, 1, 1])
    cc[0].write(f"Slip #{sl.id}: **{rec.selection}** ({rec.market}) - {rec.event_name} @ {odds_str(sl.odds_decimal)}"
                f" - stake {md_money(sl.stake)} - est. p {pct(rec.final_prob)} - EV {rec.expected_value or 0:+.3f} per \\$1")
    if cc[1].button("Approve", key=f"a{sl.id}"):
        try:
            st.success(f"Paper wager #{bank.approve(sl.id)} recorded.")
            st.rerun()
        except (BankrollError, RiskLimitError) as exc:
            st.error(str(exc))
    if cc[2].button("Decline", key=f"d{sl.id}"):
        bank.decline(sl.id)
        st.rerun()

def wager_df(ws):
    return pd.DataFrame([{"id": w.id, "placed": to_local(w.placed_at, s.timezone), "game": w.event_name,
                          "market": w.market, "selection": w.selection, "odds": odds_str(w.odds_decimal),
                          "odds source": w.odds_source, "odds time": to_local(w.odds_timestamp, s.timezone),
                          "stake": money(w.stake), "est. p": pct(w.est_prob),
                          "EV/$": None if w.expected_value is None else round(w.expected_value, 3),
                          "bankroll before": money(w.bankroll_before), "result": w.result or "open",
                          "P/L": money(w.profit), "bankroll after": money(w.bankroll_after),
                          "closing odds": w.closing_odds, "settled via": w.settlement_source} for w in ws])

st.subheader("Open paper wagers")
open_w = r.wagers(summ["account_id"], "OPEN")
if open_w:
    st.dataframe(wager_df(open_w), hide_index=True, use_container_width=True)
else:
    st.caption("None.")
cc = st.columns(2)
if cc[0].button("Settle from verified final results"):
    prov = build_providers(s, r)
    st_svc = SettlementService(s, r, prov.espn, prov.odds, prov.allsports, prov.sportsipy)
    st.write(st_svc.settle_open_wagers(), st_svc.grade_recommendations())
with cc[1].popover("Manual settlement"):
    st.caption("Only for markets the system cannot verify (e.g. props). Recorded as MANUAL.")
    if open_w:
        wid = st.selectbox("Wager", [w.id for w in open_w])
        res = st.selectbox("Result", ["win", "loss", "push", "void"])
        note = st.text_input("Verified source (required), e.g. league box score URL")
        if st.button("Settle manually") and note.strip():
            bank.settle(wid, res, f"MANUAL (unverified by system): {note}")
            st.rerun()

st.subheader("Settled paper wagers")
settled = r.wagers(summ["account_id"], "SETTLED")
if settled:
    st.dataframe(wager_df(settled), hide_index=True, use_container_width=True)
else:
    st.caption("None yet.")
perf = wager_performance(r, summ["account_id"])
st.caption(f"Settled {perf['settled']} | W-L-P-V {perf['wins']}-{perf['losses']}-{perf['pushes']}-{perf['voids']} | "
           f"profit {md_money(perf['profit'])} on {md_money(perf['staked'])} staked | ROI {pct(perf['roi'])}")

curve = bank.equity_curve(summ["account_id"])
if len(curve) > 1:
    st.plotly_chart(line_chart(list(range(len(curve))), [p["equity"] for p in curve], "Equity by ledger entry",
                               "Ledger entry #", "Equity ($)"), use_container_width=True)
st.subheader("Transaction ledger (append-only)")
st.dataframe(pd.DataFrame([{"ts": to_local(e.ts, s.timezone), "type": e.entry_type, "amount": money(e.amount),
                            "balance after": money(e.balance_after), "wager": e.wager_id, "note": e.note}
                           for e in r.ledger(summ["account_id"])]), hide_index=True, use_container_width=True)

st.subheader("Export")
cc = st.columns(3)
for col, what in zip(cc, ("wagers", "ledger", "recommendations")):
    col.download_button(f"Download {what}.csv", bank.export_csv(what), f"moneyops_{what}.csv", "text/csv")

with st.expander("Reset simulated account"):
    st.write("Archives the current paper account (history is kept) and opens a new one with "
             f"\\${s.starting_bankroll:.2f}. Requires no open wagers.")
    conf = st.text_input(f'Type "{RESET_PHRASE}" to confirm')
    if st.button("Reset paper account"):
        try:
            st.success(f"New paper account #{bank.reset(conf)}")
        except BankrollError as exc:
            st.error(str(exc))
