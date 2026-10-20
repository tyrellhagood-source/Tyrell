import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.components import paper_banner, repo, run_header, settings
from utils.timeutil import to_local

st.title("🔌 Data Sources")
paper_banner()
s = settings()
st.markdown(f"**The Odds API key:** {'✅ configured' if s.odds_api_key else '❌ missing'} (value never displayed)  \n"
            f"**ESPN public API:** {'enabled' if s.espn_enabled else 'disabled'} (unofficial, no key; personal research use)")
if st.button("Test provider connectivity (1 free Odds API call + 1 ESPN call)"):
    root = Path(__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, str(root / "scripts" / "check_providers.py")], capture_output=True,
                         text=True, cwd=root, timeout=120)
    st.code(out.stdout + out.stderr[-2000:])

rows = []
for p in repo().list_provider_status():
    rows.append({"provider": p.name, "configured": p.configured, "last success": to_local(p.last_success, s.timezone),
                 "last error": p.last_error, "error at": to_local(p.last_error_at, s.timezone),
                 "requests (total)": p.requests_made, "cache hits": p.cache_hits,
                 "quota remaining": p.quota_remaining, "quota used": p.quota_used, "last cost": p.last_request_cost})
st.subheader("Provider status")
if rows:
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
else:
    st.caption("No calls yet.")

run = run_header()
if run:
    st.subheader("Coverage in latest run")
    cov = pd.DataFrame(run.coverage_json or [])
    if not cov.empty:
        cov["notes"] = cov["notes"].apply(lambda n: "; ".join(n))
        st.dataframe(cov, hide_index=True, use_container_width=True)

st.subheader("What each source provides")
st.dataframe(pd.DataFrame([
    {"source": "The Odds API v4", "role": "Live odds: moneyline, spreads, totals, player props; scores (optional)",
     "cost": "Free 500 credits/mo; paid from ~$30/mo (verify on provider site)", "used by": "Agent 1, settlement backup"},
    {"source": "ESPN site API (unofficial)", "role": "Schedules/status, final scores, season history, injuries",
     "cost": "Free, no key, no SLA, not licensed for commercial use", "used by": "Agents 2, 3, 4; settlement"},
    {"source": "SportsDataIO / Sportradar (not connected)", "role": "Licensed stats, injuries, player logs, odds",
     "cost": "Paid / quote-based", "used by": "Add via data/provider_base.py for commercial use"},
]), hide_index=True, use_container_width=True)
