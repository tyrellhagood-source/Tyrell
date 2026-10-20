import streamlit as st

from dashboard.components import (md_money, STATUS_ICON, VERDICT_ICON, money, odds_str, paper_banner, pct, recs_df, repo,
                                  run_header, settings)
from risk.bankroll import BankrollError, BankrollService
from utils.timeutil import to_local

st.title("🏆 Top Consensus Picks")
paper_banner()
run = run_header()
if run is None:
    st.stop()
s = settings()
qualified = repo().list_recommendations(run.id, "QUALIFIED", "single")
if not qualified:
    st.header("NO QUALIFYING BETS FOUND.")
    st.write("No candidate passed every mandatory data, verification, statistical, value and risk check. "
             "That is an expected and acceptable outcome.")
else:
    st.dataframe(recs_df(qualified), hide_index=True, use_container_width=True)

st.subheader("Positive-EV but needs more research")
nr = [r for r in repo().list_recommendations(run.id, "NEEDS_RESEARCH", "single") if (r.expected_value or 0) > 0]
if nr:
    st.dataframe(recs_df(nr), hide_index=True, use_container_width=True)
else:
    st.caption("None.")

st.subheader("Pick details")
for r in qualified + nr[:10]:
    rep = r.report_json or {}
    a1, a2, a3, a4, a5 = (rep.get(k, {}) for k in ("agent1_research", "agent2_verification", "agent3_statistics",
                                                  "agent4_probability", "agent5_auditor"))
    with st.expander(f"{STATUS_ICON.get(r.status)} | {r.sport_code} | {r.event_name} | {r.market} {r.selection} "
                     f"@ {odds_str(r.odds_decimal)}"):
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Sport / league:** {r.sport_code} / {r.league}  \n**Game:** {r.event_name}  \n"
                    f"**Start:** {to_local(r.commence_time, s.timezone)}  \n**Market:** {r.market}  \n"
                    f"**Selection:** {r.selection}  \n**Sportsbook:** {r.best_book}  \n"
                    f"**Odds:** {odds_str(r.odds_decimal)}  \n"
                    f"**Odds timestamp:** {to_local(r.odds_last_update, s.timezone)}")
        d4 = a4.get("data", {})
        c2.markdown(f"**Model probability (estimate):** {pct(r.model_prob)}  \n"
                    f"**Market implied (with vig):** {pct(r.market_implied_prob)}  \n"
                    f"**Market no-vig consensus:** {pct(r.fair_prob)}  \n"
                    f"**Final blended estimate:** {pct(r.final_prob)} (interval {pct(r.prob_low)}-{pct(r.prob_high)})  \n"
                    f"**Push probability:** {pct(r.push_prob)}  \n"
                    f"**Expected value:** {r.expected_value:+.3f} per \\$1 (conservative {r.expected_value_low or 0:+.3f})  \n"
                    f"**Method:** {d4.get('method')}")
        stake = r.suggested_stake
        c3.markdown(f"**Historical hit rate vs this line:** {pct(r.hist_rate)} (n={r.hist_n})  \n"
                    f"**95% CI:** {pct((a3.get('data', {}).get('primary') or {}).get('ci_low'))}-"
                    f"{pct((a3.get('data', {}).get('primary') or {}).get('ci_high'))}  \n"
                    f"**Consensus score:** {r.consensus_score:.3f}  \n**Suggested paper stake:** {md_money(stake)}  \n"
                    f"**Profit if win:** {md_money(stake * (r.odds_decimal - 1) if r.odds_decimal else None)}  \n"
                    f"**Max loss:** {md_money(stake)}")
        st.markdown("**Agent verdicts:** " + " | ".join(
            f"{VERDICT_ICON.get(v['verdict'], '')} {v['name'].split(' - ')[1]}: {v['verdict']}"
            for v in (r.verdicts_json or {}).values()))
        inj = a1.get("data", {}).get("injuries") or {}
        st.markdown("**Injuries (ESPN, significant statuses):** " + ("; ".join(
            f"{t}: " + (", ".join(f"{i['player']} ({i['status']})" for i in v) or "none listed")
            for t, v in inj.items()) or "not available"))
        pro = [a4.get("summary", "")] + a3.get("reasons", [])
        con = a3.get("data", {}).get("caveats", []) + a4.get("flags", []) + a5.get("flags", []) + \
            a5.get("data", {}).get("disagreements", []) + a2.get("flags", [])
        cc1, cc2 = st.columns(2)
        cc1.markdown("**Why it could win**\n" + "\n".join(f"- {x}" for x in pro if x))
        cc2.markdown("**Why it could lose**\n" + "\n".join(f"- {x}" for x in con if x))
        if r.status == "QUALIFIED" and stake > 0:
            if st.button("Create paper bet slip", key=f"slip{r.id}"):
                try:
                    sid = BankrollService(repo(), s).propose(r.id)
                    st.success(f"Slip #{sid} proposed. Approve or decline it on the Virtual Bankroll page.")
                except BankrollError as exc:
                    st.error(str(exc))
