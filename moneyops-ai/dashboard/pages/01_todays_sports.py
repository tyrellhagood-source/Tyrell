import pandas as pd
import streamlit as st

from dashboard.components import paper_banner, repo, run_header, settings
from models.expected_value import format_american
from risk.risk_limits import odds_age_minutes
from services.analysis import AnalysisService
from utils.timeutil import to_local, utcnow

st.title("🗓️ Today's Sports")
paper_banner()
s = settings()

c1, c2 = st.columns([1, 3])
with c1:
    if st.button("▶ Run analysis now", type="primary", disabled=not s.odds_api_key):
        with st.spinner("Five agents at work: fetching odds, history, schedules, injuries..."):
            res = AnalysisService(s, repo()).run()
        st.toast(res.message)
        st.rerun()
with c2:
    if not s.odds_api_key:
        st.error("ODDS_API_KEY is not set. Live odds are unavailable, so no games can be analysed. "
                 "Add the key to `.env` (see README).")
    else:
        st.caption(f"Each run spends up to {s.max_credits_per_run} Odds API credits "
                   f"({len(s.odds_markets)} markets x {len(s.odds_regions)} region(s) per in-season sport).")

run = run_header()
if run is None:
    st.stop()

st.subheader("Sport coverage")
cov = pd.DataFrame(run.coverage_json or [])
if not cov.empty:
    cov["notes"] = cov["notes"].apply(lambda n: "; ".join(n))
    st.dataframe(cov, hide_index=True, use_container_width=True)
    st.caption("A sport without an independent schedule source or model can show odds but can never be "
               "fully QUALIFIED. Gaps are listed, never filled with guesses.")

snaps = repo().snapshots_for_run(run.id)
if not snaps:
    st.info("No odds were retrieved in this run.")
    st.stop()
now = utcnow()
games = {}
for sn in snaps:
    g = games.setdefault(sn.event_id, {"sport": sn.sport_code, "game": f"{sn.away_team} @ {sn.home_team}",
                                       "start": sn.commence_time, "home": sn.home_team, "away": sn.away_team,
                                       "books": set(), "newest": sn.last_update, "ml": {}})
    g["books"].add(sn.bookmaker)
    g["newest"] = max(g["newest"], sn.last_update)
    if sn.market == "h2h":
        g["ml"][sn.outcome] = max(g["ml"].get(sn.outcome, 0), sn.price_decimal)
rows = []
for g in sorted(games.values(), key=lambda g: g["start"]):
    live = g["start"] <= now
    rows.append({"sport": g["sport"], "game": g["game"], "start": to_local(g["start"], s.timezone),
                 "status": "🔴 LIVE odds" if live else "upcoming",
                 "best away ML": format_american(g["ml"][g["away"]]) if g["ml"].get(g["away"]) else "n/a",
                 "best home ML": format_american(g["ml"][g["home"]]) if g["ml"].get(g["home"]) else "n/a",
                 "draw": format_american(g["ml"]["Draw"]) if g["ml"].get("Draw") else "",
                 "books": len(g["books"]),
                 "freshest quote age (min, now)": round(odds_age_minutes(g["newest"], now), 1)})
df = pd.DataFrame(rows)
sports = sorted(df["sport"].unique())
pick = st.multiselect("Filter sports", sports, default=sports)
today_only = st.toggle("Only games starting today (local time)", value=False)
view = df[df["sport"].isin(pick)]
if today_only:
    today = to_local(now, s.timezone)[:10]
    view = view[view["start"].str.startswith(today)]
st.dataframe(view, hide_index=True, use_container_width=True)
st.caption(f"Odds retrieved at {to_local(snaps[0].retrieved_at, s.timezone)} from {snaps[0].source}. "
           "Quote age keeps growing after retrieval; re-run before acting on any price.")
