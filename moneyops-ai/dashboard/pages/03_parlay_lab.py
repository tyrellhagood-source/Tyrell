import pandas as pd
import streamlit as st

from dashboard.components import md_money, money, odds_str, paper_banner, pct, repo, run_header
from models.expected_value import format_american
from risk.bankroll import BankrollError, BankrollService
from dashboard.components import settings
from services.queries import custom_parlay

st.title("🧩 Parlay Lab")
paper_banner()
st.caption("Parlays are ranked separately from singles. Independent legs multiply; same-game legs use the "
           "model's joint score distribution; anything else is shown as UNCERTAIN with bounds only.")
run = run_header()
if run is None:
    st.stop()
parlays = repo().list_recommendations(run.id, kind="parlay")
q = [p for p in parlays if p.status == "QUALIFIED"]
st.subheader("System parlays")
if not q:
    st.info("NO QUALIFYING PARLAYS FOUND. Parlays are only built from qualified single legs.")
rows = []
for p in parlays:
    rep = p.report_json or {}
    rows.append({"id": p.id, "status": p.status, "legs": p.selection, "combined odds": odds_str(p.odds_decimal),
                 "p (est.)": pct(p.final_prob), "p conservative": pct(p.prob_low),
                 "break-even": pct(1 / p.odds_decimal), "EV/$": None if p.expected_value is None else round(p.expected_value, 3),
                 "paper stake": money(p.suggested_stake), "correlation": rep.get("correlation_status"),
                 "why rejected": "; ".join(rep.get("reasons", []))})
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
for p in q:
    rep = p.report_json or {}
    with st.expander(f"{p.selection} @ {odds_str(p.odds_decimal)}"):
        st.dataframe(pd.DataFrame([{"leg": l["selection"], "game": l["event_name"], "market": l["market"],
                                    "odds": odds_str(l["decimal"]), "p": pct(l["p_win"])} for l in p.legs_json]),
                     hide_index=True)
        st.markdown(f"Payout if win {md_money(rep.get('payout_if_win'))} | net profit {md_money(rep.get('net_profit_if_win'))} "
                    f"| max loss {md_money(rep.get('max_loss'))}")
        st.markdown("**Warnings:** " + "; ".join(rep.get("warnings", [])))
        st.markdown("**How it fails:** " + "; ".join(rep.get("fail_reasons", [])))
        if st.button("Create paper bet slip", key=f"ps{p.id}"):
            try:
                st.success(f"Slip #{BankrollService(repo(), settings()).propose(p.id)} proposed.")
            except BankrollError as exc:
                st.error(str(exc))

st.subheader("Build your own")
singles = [r for r in repo().list_recommendations(run.id, kind="single") if r.final_prob and r.odds_decimal]
options = {f"#{r.id} {r.event_name} | {r.market} {r.selection} @ {r.odds_decimal:.2f} [{r.status}]": r.id
           for r in singles}
chosen = st.multiselect("Pick 2+ legs (any status - the lab will tell you what is wrong)", list(options))
stake = st.number_input("Hypothetical stake ($)", 0.0, 100.0, 1.0, 0.25)
if len(chosen) >= 2:
    try:
        res = custom_parlay(repo(), [options[c] for c in chosen], stake)
    except ValueError as exc:
        st.error(str(exc))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Combined odds", f"{res['combined_decimal']:.2f}", format_american(res["combined_decimal"]))
        if res["probability"] is None:
            lo, hi = res["probability_bounds"]
            c2.metric("Win probability", "UNCERTAIN", f"bounds {pct(lo)}-{pct(hi)}", delta_color="off")
            c3.metric("Expected value", "not estimable")
        else:
            c2.metric("Win probability (est.)", pct(res["probability"]), res["correlation_status"], delta_color="off")
            c3.metric("EV per $1", f"{res['expected_value']:+.3f}")
        c4.metric("Break-even probability", pct(res["break_even_probability"]))
        st.markdown(f"Payout {md_money(res['payout_if_win'])} | net profit {md_money(res['net_profit_if_win'])} | "
                    f"max loss {md_money(res['max_loss'])}")
        for w in res["warnings"] + res["fail_reasons"]:
            st.write("- " + w)
