"""Daily research workflow: settle finished games, run the five agents,
print the ranked report. Use --loop to repeat every REFRESH_INTERVAL_MINUTES.

    python scripts/run_daily_analysis.py            # one run
    python scripts/run_daily_analysis.py --loop     # keep refreshing
    python scripts/run_daily_analysis.py --settle-only
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from models.expected_value import format_american  # noqa: E402
from services.analysis import AnalysisService  # noqa: E402
from services.runtime import get_repo  # noqa: E402
from services.settlement import SettlementService  # noqa: E402
from utils.timeutil import to_local  # noqa: E402


def once(settle_only: bool = False) -> None:
    s, repo = get_settings(), get_repo()
    svc = AnalysisService(s, repo)
    st = SettlementService(s, repo, svc.espn, svc.odds)
    print("Settlement:", st.settle_open_wagers(), "| grading:", st.grade_recommendations())
    if settle_only:
        return
    res = svc.run()
    print(f"\nRun {res.run_id}: {res.status} - {res.message}")
    for e in res.errors:
        print("  ERROR:", e)
    print("\nCoverage:")
    for row in res.coverage:
        print(f"  {row['sport']:<10} in_season={row['in_season']!s:<5} events={row['events']:<3} "
              f"history={row['history_games']:<5} model={row['model']:<8} {'; '.join(row['notes'])[:120]}")
    singles = sorted([c for c in res.candidates if c.status == "QUALIFIED"], key=lambda c: -c.consensus_score)
    print("\nTOP CONSENSUS SINGLES" if singles else "\nNO QUALIFYING BETS FOUND.")
    for c in singles[:15]:
        p4 = c.reports["agent4_probability"].data
        print(f"  [{c.consensus_score:.3f}] {c.cfg.code} {c.event.name} {to_local(c.event.commence_time, s.timezone)} | "
              f"{c.market} {c.label} @ {c.best.price_decimal:.2f} ({format_american(c.best.price_decimal)}) "
              f"{c.best.bookmaker_title} | p={p4['final_prob']:.3f} EV={p4['expected_value']:+.3f} "
              f"stake=${c.stake:.2f} (paper)")
    research = [c for c in res.candidates if c.status == "NEEDS_RESEARCH"]
    print(f"\nNeeds research: {len(research)} | Rejected: {sum(c.status == 'REJECTED' for c in res.candidates)}")
    parlays = [p for p in res.parlays if p["status"] == "QUALIFIED"]
    if parlays:
        print("\nPARLAY CANDIDATES (ranked separately):")
        for p in parlays[: s.max_parlays]:
            print(f"  {' + '.join(l.selection for l in p['legs'])} @ {p['combined_decimal']:.2f} "
                  f"p={p['probability']:.3f} EV={p['expected_value']:+.3f} stake=${p['stake']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--settle-only", action="store_true")
    args = ap.parse_args()
    while True:
        once(args.settle_only)
        if not args.loop:
            break
        mins = get_settings().refresh_interval_minutes
        print(f"\nSleeping {mins} min (respecting API credit budget)...")
        time.sleep(mins * 60)
