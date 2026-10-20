from collections import defaultdict

import pandas as pd
import streamlit as st

from dashboard.components import money, paper_banner, pct, recs_df, repo, run_header, settings
from risk.bankroll import BankrollService

st.title("🛡️ Risk Center")
paper_banner()
s, r = settings(), repo()
summ = BankrollService(r, s).summary()
c = st.columns(4)
c[0].metric("Max drawdown (equity)", pct(summ["max_drawdown_pct"]))
c[1].metric("Open exposure", money(summ["open_exposure"]),
            f"{summ['open_exposure'] / summ['equity']:.1%} of equity" if summ["equity"] else "", delta_color="off")
c[2].metric("Exposure cap", money(summ["equity"] * s.max_open_exposure_pct), f"{s.max_open_exposure_pct:.0%}",
            delta_color="off")
halted = summ["peak_equity"] and (summ["peak_equity"] - summ["equity"]) / summ["peak_equity"] >= s.max_drawdown_halt_pct
c[3].metric("Drawdown halt", "ACTIVE ⛔" if halted else "not triggered", f"at {s.max_drawdown_halt_pct:.0%}",
            delta_color="off")

st.subheader("Configured limits")
limits = {"Max stake per single": f"{s.max_stake_pct:.1%} of balance", "Max stake per parlay": f"{s.parlay_max_stake_pct:.1%}",
          "Kelly multiplier (before cap)": s.kelly_fraction, "Max open exposure": f"{s.max_open_exposure_pct:.0%} of equity",
          "Drawdown halt": f"{s.max_drawdown_halt_pct:.0%} below peak", "Minimum EV": s.min_ev,
          "Minimum conservative EV": s.min_ev_lower, "Minimum books confirming a line": s.min_books,
          "Minimum historical sample": s.min_sample, "Max model-vs-market gap": f"{s.max_model_market_gap:.0%}",
          "Stale odds (pre-game / live)": f"{s.stale_odds_minutes_pregame:g} / {s.stale_odds_minutes_live:g} min",
          "Odds API credits per run": s.max_credits_per_run}
st.dataframe(pd.DataFrame(limits.items(), columns=["limit", "value"]), hide_index=True)

st.subheader("Correlated open positions")
by_event = defaultdict(list)
for w in r.wagers(summ["account_id"], "OPEN"):
    rec = r.get_recommendation(w.recommendation_id)
    for ev in (rec.event_id.split(",") if rec else []):
        by_event[ev].append(f"#{w.id} {w.selection}")
corr = {k: v for k, v in by_event.items() if len(v) > 1}
st.write(corr if corr else "No two open wagers share a game.")

run = run_header()
if run:
    st.subheader("Rejected recommendations (latest run) and why")
    rej = r.list_recommendations(run.id, "REJECTED", "single")
    if rej:
        df = recs_df(rej)
        df["reasons"] = ["; ".join((x.report_json.get("agent5_auditor", {}) or {}).get("reasons", [])[:4]) for x in rej]
        st.dataframe(df[["id", "game", "market", "selection", "odds", "EV/$", "reasons"]], hide_index=True,
                     use_container_width=True)
    else:
        st.caption("None.")
