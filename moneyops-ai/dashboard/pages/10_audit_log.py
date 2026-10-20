import pandas as pd
import streamlit as st

from agents.base import AGENT_NAMES
from dashboard.components import paper_banner, repo, settings
from utils.timeutil import to_local

st.title("🧾 Audit Log")
paper_banner()
s = settings()
runs = repo().list_runs(100)
run_opts = {"All runs": None} | {f"Run #{r.id} ({r.status}, {to_local(r.started_at, s.timezone)})": r.id for r in runs}
c1, c2, c3 = st.columns(3)
run_id = run_opts[c1.selectbox("Run", list(run_opts))]
agents = {"All": None, "Orchestrator": "orchestrator", "Bankroll": "bankroll"} | {v: k for k, v in AGENT_NAMES.items()}
agent = agents[c2.selectbox("Agent", list(agents))]
level = c3.selectbox("Level", ["All", "INFO", "WARNING", "ERROR"])
logs = repo().list_audit(run_id, agent, 2000)
df = pd.DataFrame([{"ts": to_local(a.ts, s.timezone), "run": a.run_id, "agent": AGENT_NAMES.get(a.agent, a.agent),
                    "level": a.level, "action": a.action, "message": a.message,
                    "detail": "; ".join((a.payload_json or {}).get("reasons", []) + (a.payload_json or {}).get("flags", []))}
                   for a in logs])
if level != "All" and not df.empty:
    df = df[df["level"] == level]
st.dataframe(df, hide_index=True, use_container_width=True, height=600)
st.subheader("Run history")
st.dataframe(pd.DataFrame([{"run": r.id, "status": r.status, "started": to_local(r.started_at, s.timezone),
                            "finished": to_local(r.finished_at, s.timezone), "model": r.model_version,
                            "summary": (r.summary_json or {}).get("message"), "errors": "; ".join(r.errors_json or [])}
                           for r in runs]), hide_index=True, use_container_width=True)
