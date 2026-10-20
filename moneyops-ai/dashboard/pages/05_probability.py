import pandas as pd
import streamlit as st

from dashboard.charts import calibration_chart, histogram, interval_chart, model_vs_market
from dashboard.components import paper_banner, pct, repo, run_header
from services.queries import performance_report

st.title("📈 Probability Analytics")
paper_banner()
run = run_header()
if run is not None:
    recs = [r for r in repo().list_recommendations(run.id, kind="single") if r.final_prob is not None]
    if recs:
        mm = [{"market": r.fair_prob, "model": r.model_prob, "label": f"{r.event_name} {r.market} {r.selection}"}
              for r in recs if r.model_prob is not None and r.fair_prob is not None]
        c1, c2 = st.columns(2)
        if mm:
            c1.plotly_chart(model_vs_market(mm), use_container_width=True)
        c2.plotly_chart(histogram([r.expected_value for r in recs if r.expected_value is not None],
                                  "Distribution of estimated EV per $1 (all candidates)", "EV per $1"),
                        use_container_width=True)
        top = sorted([r for r in recs if r.prob_low is not None and r.status != "REJECTED"],
                     key=lambda r: -r.consensus_score)
        if top:
            st.plotly_chart(interval_chart([{"label": f"{r.selection} ({r.market})", "p": r.final_prob,
                                             "lo": r.prob_low, "hi": r.prob_high, "be": 1 / r.odds_decimal}
                                            for r in top]), use_container_width=True)
        st.caption("Points on the diagonal mean the model agrees with the market. Large gaps are vetoed by the "
                   "Chief Auditor - they usually mean the model lacks information (injuries, lineups).")

st.header("Model evaluation - HYPOTHETICAL backtests")
st.caption("Walk-forward, time-ordered: train 60% -> validate 20% (choose regularisation) -> test 20% (refit weekly, "
           "predict only future games). Measures probability quality only; no historical odds, so no profit claim.")
bts = repo().latest_backtests()
if not bts:
    st.info("No backtests yet (they run automatically during analysis for sports with history).")
for code, bt in sorted(bts.items()):
    m = bt.metrics_json
    with st.expander(f"{code}: {bt.status} ({bt.created_at:%Y-%m-%d})", expanded=False):
        if bt.status != "OK":
            st.write(m.get("detail"))
            continue
        c = st.columns(5)
        c[0].metric("Test predictions", m["n_predictions"])
        c[1].metric("Brier (model)", f"{m['brier']:.4f}", f"baseline {m['baseline_brier']:.4f}", delta_color="off")
        c[2].metric("Brier skill", pct(m["brier_skill_score"]))
        c[3].metric("Log loss", f"{m['log_loss']:.4f}", f"baseline {m['baseline_log_loss']:.4f}", delta_color="off")
        c[4].metric("Calibration error (ECE)", pct(m["ece"], 2))
        st.plotly_chart(calibration_chart(m["calibration"], f"{code} home-win calibration (test set)"),
                        use_container_width=True)
        st.caption(f"Train {m['train_period'][0][:10]}..{m['train_period'][1][:10]} | validation "
                   f"{m['validation_period'][0][:10]}..{m['validation_period'][1][:10]} | test "
                   f"{m['test_period'][0][:10]}..{m['test_period'][1][:10]} | alpha {m['alpha']}")

st.header("Live calibration of recorded recommendations")
perf = performance_report(repo())
cons = perf["consensus_picks"]
allp = perf["all_recommendations"]
for label, s in (("Consensus picks", cons), ("All recommendations (incl. rejected)", allp)):
    if s["calibration"]:
        st.plotly_chart(calibration_chart(s["calibration"], f"{label}: predicted vs realised"),
                        use_container_width=True)
        st.caption(f"Brier {s['brier']:.4f} | log loss {s['log_loss']:.4f} | ECE {pct(s['ece'], 2)} | "
                   f"n={s['wins'] + s['losses']} decided. An 80% estimate should win ~80% of the time over a "
                   "large sample; small samples are noisy.")
    else:
        st.caption(f"{label}: no settled results yet.")
