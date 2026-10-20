import streamlit as st

from dashboard.charts import line_chart
from dashboard.components import paper_banner, perf_table, repo
from models.performance import summarize
from services.queries import performance_report, rec_record

st.title("📚 Historical Results")
paper_banner()
st.caption("Every recommendation is recorded and graded, including rejected and losing picks; nothing is deleted. "
           "Flat 1-unit staking per pick. Win rate = W/(W+L). Yield = units/(W+L+P). Max DD = largest peak-to-"
           "trough fall in cumulative units. CLV uses the last pre-start snapshot only when it is within 60 min "
           "of start. These are research records, separate from paper-wager results (Virtual Bankroll page).")
perf = performance_report(repo())
c = st.columns(6)
s = perf["consensus_picks"]
c[0].metric("Consensus picks", s["total"])
c[1].metric("Settled", s["settled"])
c[2].metric("W-L-P-V", f"{s['wins']}-{s['losses']}-{s['pushes']}-{s['voids']}")
c[3].metric("Win rate", "n/a" if s["win_rate"] is None else f"{s['win_rate']:.1%}")
c[4].metric("Units", f"{s['units']:+.2f}")
c[5].metric("Yield", "n/a" if s["yield"] is None else f"{s['yield']:+.1%}")

recs = sorted([rec_record(r) for r in repo().list_recommendations() if r.status == "QUALIFIED"
               and r.result in ("win", "loss", "push", "void")], key=lambda r: r["ts"])
if recs:
    cum, xs, ys = 0.0, [], []
    for i, r in enumerate(recs, 1):
        cum += (r["odds"] - 1) if r["result"] == "win" else (-1 if r["result"] == "loss" else 0)
        xs.append(i)
        ys.append(cum)
    st.plotly_chart(line_chart(xs, ys, "Cumulative units - consensus picks", "Settled pick #", "Units"),
                    use_container_width=True)
else:
    st.info("No settled consensus picks yet. Results appear after games finish and settlement runs.")

for title, key in (("Overall", None), ("By agent (picks each agent endorsed)", "by_agent"),
                   ("By recommendation status", "by_status"), ("By sport", "by_sport"), ("By league", "by_league"),
                   ("By market", "by_market"), ("By probability range", "by_probability"),
                   ("By odds range (decimal)", "by_odds"), ("By model version", "by_model_version")):
    st.subheader(title)
    groups = {"All recommendations": perf["all_recommendations"], "Consensus picks": perf["consensus_picks"],
              "Consensus singles": perf["singles"], "Consensus parlays": perf["parlays"]} if key is None else perf[key]
    if groups:
        st.dataframe(perf_table(groups), hide_index=True, use_container_width=True)
    else:
        st.caption("No data yet.")
