import json

import streamlit as st

from agents.base import AGENT_NAMES
from dashboard.charts import bar_chart
from dashboard.components import STATUS_ICON, VERDICT_ICON, paper_banner, repo, run_header

st.title("🤖 Five-Bot Control Room")
paper_banner()
run = run_header()
if run is None:
    st.stop()
recs = repo().list_recommendations(run.id, kind="single")
if not recs:
    st.info("No candidates in this run.")
    st.stop()
status_filter = st.multiselect("Status", ["QUALIFIED", "NEEDS_RESEARCH", "REJECTED"],
                               default=["QUALIFIED", "NEEDS_RESEARCH", "REJECTED"])
recs = [r for r in recs if r.status in status_filter]
if not recs:
    st.stop()
labels = {f"{STATUS_ICON.get(r.status)} | {r.event_name} | {r.market} {r.selection}": r for r in recs}
r = labels[st.selectbox("Candidate", list(labels))]

disag = (r.report_json.get("agent5_auditor", {}).get("data", {}) or {}).get("disagreements", [])
if disag:
    st.warning("**Agent disagreements:**\n" + "\n".join(f"- {d}" for d in disag))
cols = st.columns(5)
for col, (key, name) in zip(cols, AGENT_NAMES.items()):
    rep = r.report_json.get(key, {})
    col.markdown(f"**{name.split(' - ')[1]}**  \n{VERDICT_ICON.get(rep.get('verdict'), '')} "
                 f"`{rep.get('verdict', 'n/a')}`  \nscore {rep.get('score', 0):.2f}")

for key, name in AGENT_NAMES.items():
    rep = r.report_json.get(key, {})
    with st.expander(f"{name}: {rep.get('verdict')} - {rep.get('summary', '')}", expanded=key == "agent5_auditor"):
        if rep.get("reasons"):
            st.markdown("**Reasons**\n" + "\n".join(f"- {x}" for x in rep["reasons"]))
        if rep.get("flags"):
            st.markdown("**Flags**\n" + "\n".join(f"- {x}" for x in rep["flags"]))
        data = rep.get("data", {})
        if key == "agent1_research" and data.get("books"):
            st.markdown("**Sources & prices (every quote with its timestamp)**")
            st.dataframe(data["books"], hide_index=True, use_container_width=True)
        if key == "agent2_verification" and data.get("checks"):
            st.dataframe(data["checks"], hide_index=True, use_container_width=True)
        if key == "agent5_auditor" and data.get("components"):
            comp = data["components"]
            st.plotly_chart(bar_chart(list(comp), [comp[k] * data["weights"][k] for k in comp],
                                      "Weighted score contributions", "weighted points"), use_container_width=True)
            st.markdown("**Mandatory gates**")
            st.json(data.get("gates"))
            st.markdown("**Independent recalculations**")
            st.json(data.get("recalculations"))
        with st.popover("Raw structured output"):
            st.code(json.dumps(data, indent=2, default=str)[:20000], language="json")
