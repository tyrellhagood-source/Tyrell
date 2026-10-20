"""MoneyOps AI dashboard. Run: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="MoneyOps AI", page_icon="📊", layout="wide")

PAGES = [
    st.Page("dashboard/pages/01_todays_sports.py", title="Today's Sports", icon="🗓️", url_path="today", default=True),
    st.Page("dashboard/pages/02_top_picks.py", title="Top Consensus Picks", icon="🏆", url_path="top-picks"),
    st.Page("dashboard/pages/03_parlay_lab.py", title="Parlay Lab", icon="🧩", url_path="parlay-lab"),
    st.Page("dashboard/pages/04_control_room.py", title="Five-Bot Control Room", icon="🤖", url_path="control-room"),
    st.Page("dashboard/pages/05_probability.py", title="Probability Analytics", icon="📈", url_path="probability"),
    st.Page("dashboard/pages/06_history.py", title="Historical Results", icon="📚", url_path="history"),
    st.Page("dashboard/pages/07_bankroll.py", title="Virtual Bankroll", icon="💵", url_path="bankroll"),
    st.Page("dashboard/pages/08_risk.py", title="Risk Center", icon="🛡️", url_path="risk"),
    st.Page("dashboard/pages/09_data_sources.py", title="Data Sources", icon="🔌", url_path="data-sources"),
    st.Page("dashboard/pages/10_audit_log.py", title="Audit Log", icon="🧾", url_path="audit-log"),
]
st.navigation(PAGES).run()
