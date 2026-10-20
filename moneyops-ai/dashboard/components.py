"""Shared Streamlit helpers."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from config import get_settings
from models.expected_value import format_american
from services.runtime import get_repo
from utils.timeutil import to_local

STATUS_ICON = {"QUALIFIED": "✅ QUALIFIED", "NEEDS_RESEARCH": "🔎 NEEDS RESEARCH", "REJECTED": "⛔ REJECTED"}
VERDICT_ICON = {"COMPLETE": "✅", "VERIFIED": "✅", "SUPPORTS": "✅", "PASS": "✅", "QUALIFIED": "✅",
                "NEUTRAL": "➖", "INCOMPLETE": "⚠️", "INSUFFICIENT_DATA": "⚠️", "NEEDS_RESEARCH": "🔎",
                "STALE": "⛔", "REJECTED": "⛔", "FAIL": "⛔", "CONTRADICTS": "⛔"}


def settings():
    return get_settings()


def repo():
    return get_repo()


def paper_banner():
    st.caption("🧪 Research & **paper trading only**. No real money, no sportsbook connection. "
               "Estimates are not guarantees; past results do not predict future results.")


def latest_run():
    return repo().latest_run(status=None)


def run_header():
    run = latest_run()
    if run is None:
        st.info("No analysis has been run yet. Use **Run analysis now** on *Today's Sports*.")
        return None
    s = settings()
    msg = (run.summary_json or {}).get("message", "")
    st.markdown(f"**Latest run #{run.id}** - {run.status} - started {to_local(run.started_at, s.timezone)}"
                f" - model `{run.model_version}`")
    if run.status != "COMPLETED":
        st.error(msg or "Run did not complete.")
    elif "NO QUALIFYING" in msg:
        st.warning(msg)
    elif msg:
        st.success(msg)
    for e in run.errors_json or []:
        st.error(e)
    return run


def pct(x, digits=1):
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def money(x):
    return "n/a" if x is None else f"${x:,.2f}"


def md_money(x):
    """money() escaped for st.markdown/caption, where $...$ would render as LaTeX."""
    return money(x).replace("$", "\\$")


def odds_str(d):
    return "n/a" if not d else f"{d:.2f} ({format_american(d)})"


def recs_df(recs) -> pd.DataFrame:
    tz = settings().timezone
    rows = []
    for r in recs:
        rows.append({"id": r.id, "status": STATUS_ICON.get(r.status, r.status), "score": round(r.consensus_score, 3),
                     "sport": r.sport_code, "game": r.event_name, "start": to_local(r.commence_time, tz),
                     "market": r.market, "selection": r.selection, "book": r.best_book,
                     "odds": odds_str(r.odds_decimal), "model p": pct(r.model_prob), "market p (no-vig)": pct(r.fair_prob),
                     "final p": pct(r.final_prob), "EV/$": None if r.expected_value is None else round(r.expected_value, 3),
                     "hist": "n/a" if r.hist_n is None else f"{pct(r.hist_rate)} (n={r.hist_n})",
                     "paper stake": money(r.suggested_stake), "result": r.result})
    return pd.DataFrame(rows)


def perf_table(groups: dict) -> pd.DataFrame:
    rows = []
    for k, s in groups.items():
        rows.append({"group": k, "recs": s["total"], "settled": s["settled"], "W": s["wins"], "L": s["losses"],
                     "P": s["pushes"], "V": s["voids"], "win rate": pct(s["win_rate"]),
                     "avg odds": None if s["avg_odds"] is None else round(s["avg_odds"], 2),
                     "units": round(s["units"], 2), "yield": pct(s["yield"]),
                     "max DD (u)": round(s["max_drawdown_units"], 2), "best streak": s["longest_win_streak"],
                     "worst streak": s["longest_loss_streak"],
                     "Brier": None if s["brier"] is None else round(s["brier"], 4),
                     "log loss": None if s["log_loss"] is None else round(s["log_loss"], 4),
                     "ECE": None if s["ece"] is None else round(s["ece"], 4),
                     "CLV": pct(s["clv"]) + (f" (n={s['clv_n']})" if s["clv_n"] else "")})
    return pd.DataFrame(rows)
