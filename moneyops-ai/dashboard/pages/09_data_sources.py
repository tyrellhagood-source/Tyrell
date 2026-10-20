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
            f"**AllSportsAPI key:** {'✅ configured' if s.allsports_api_key else '➖ not set (optional)'}  \n"
            f"**balldontlie key:** {'✅ configured' if s.balldontlie_api_key else '➖ not set (optional)'}"
            f" (free plan: games only, {s.balldontlie_requests_per_minute} requests/min)  \n"
            f"**ESPN public API:** {'enabled' if s.espn_enabled else 'disabled'} (unofficial, no key; personal research use)  \n"
            f"**nba_api (stats.nba.com, NBA player logs):** {'enabled' if s.nba_api_enabled else 'disabled'}  \n"
            f"**nflreadpy (nflverse, NFL player stats):** {'enabled' if s.nflreadpy_enabled else 'disabled'}")
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
    {"source": "AllSportsAPI (apiv2.allsportsapi.com)",
     "role": "2nd independent schedule + final results, extra bookmakers (cross-provider price check), "
             "external probabilities, history fallback; football/basketball/tennis/cricket",
     "cost": "Paid plans, 14-day free trial; odds/probability field names unverified until probe_allsports.py runs",
     "used by": "Agents 1, 2, 4; settlement"},
    {"source": "balldontlie (free plan)",
     "role": "Games: 3rd independent schedule/status check, 3rd final-score source, backup season results "
             "(NBA, NFL, MLB, NCAAB, EPL)",
     "cost": "Free plan: games only, 5 requests/min per sport; stats/injuries need All-Star ($9.99), odds need GOAT",
     "used by": "Agent 2; settlement; team model fallback"},
    {"source": "nba_api (swar/nba_api 1.11.x -> stats.nba.com)",
     "role": "NBA player game logs for props (one LeagueGameLog request per season); prop settlement",
     "cost": "Free, no key; maintained 2026; unofficial client of the NBA's public stats site",
     "used by": "Agents 2, 3, 4 (props); settlement"},
    {"source": "nflreadpy (nflverse)",
     "role": "NFL weekly player stats + schedules for props; prop settlement",
     "cost": "Free, no key; maintained 2026; data published by nflverse after games (may lag ~a day)",
     "used by": "Agents 2, 3, 4 (props); settlement"},
    {"source": "SportsDataIO / Sportradar (not connected)", "role": "Licensed stats, injuries, player logs, odds",
     "cost": "Paid / quote-based", "used by": "Add via data/provider_base.py for commercial use"},
]), hide_index=True, use_container_width=True)
