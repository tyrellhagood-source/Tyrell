"""Real-data NFL research report from nflverse (no API key needed).

Uses nflreadpy to download real 2025+2026 schedules/results and the betting
lines nflverse publishes in its schedule file, then:
  * fits the MoneyOps gaussian team-rating model on games played before now,
  * runs the time-ordered walk-forward backtest on those real games,
  * prices each upcoming game of the next week: model probability, market
    no-vig probability, the documented 35/65 blend, and EV at the listed price.

Limits (printed in the output): nflverse does not say which sportsbook the
lines come from or when they were taken, so freshness cannot be verified and
nothing in this report is a QUALIFIED pick - it is a research view.

    python scripts/nfl_week_report.py [--json out.json]
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.player_stats_provider import NFL_NAME, nfl_season  # noqa: E402
from data.schemas import HistoricalGame  # noqa: E402
from data.sports_registry import SPORTS  # noqa: E402
from models.backtesting import walk_forward  # noqa: E402
from models.expected_value import american_to_decimal, expected_value, remove_vig  # noqa: E402
from models.probability import TeamRatingModel  # noqa: E402

ET = ZoneInfo("America/New_York")
W_MODEL = 0.35
MAX_GAP = 0.10


def kickoff(row) -> datetime:
    t = row.get("gametime") or "13:00"
    return datetime.strptime(f"{row['gameday']} {t}", "%Y-%m-%d %H:%M").replace(tzinfo=ET).astimezone(timezone.utc)


def load(now: datetime):
    import nflreadpy
    season = nfl_season(now)
    rows = nflreadpy.load_schedules([season - 1, season]).to_dicts()
    rows = [r for r in rows if r.get("gameday")]
    games = [HistoricalGame(r["game_id"], kickoff(r), r["home_team"], r["away_team"], int(r["home_score"]),
                            int(r["away_score"]), r.get("location") == "Neutral", "nflverse")
             for r in rows if r.get("home_score") is not None and kickoff(r) < now]
    upcoming = sorted([r for r in rows if r.get("home_score") is None and kickoff(r) > now], key=kickoff)
    return games, upcoming


def price(model_p, push, market_p, odds_am):
    """Blend + EV exactly as Agent 4 does. Returns None when inputs are missing."""
    if model_p is None or market_p is None or odds_am is None:
        return None
    d = american_to_decimal(odds_am)
    final = W_MODEL * model_p + (1 - W_MODEL) * market_p * (1 - push)
    return {"odds": odds_am, "model": model_p, "market_p": market_p, "final": final, "push": push,
            "ev": expected_value(final, d, push), "gap": model_p - market_p * (1 - push)}


def verdict(p):
    if p is None:
        return "NO DATA"
    if abs(p["gap"]) > MAX_GAP:
        return "VETO: model far from market"
    return "+EV (unverified line)" if p["ev"] >= 0.02 else "no edge"


def main(now=None, out=None):
    now = now or datetime.now(timezone.utc)
    games, upcoming = load(now)
    cfg = SPORTS["NFL"]
    bt = walk_forward(cfg, games)
    alpha = bt.get("alpha") if bt.get("status") == "OK" else None
    model = TeamRatingModel(cfg, alpha).fit(games, now)
    next_week = upcoming[0]["week"] if upcoming else None
    report = {"generated_at": now.isoformat(), "source": "nflverse via nflreadpy (real data)",
              "games_used": len(games), "trained_through": model.trained_through.isoformat(),
              "backtest": {k: bt.get(k) for k in ("status", "n_predictions", "brier", "baseline_brier",
                                                  "brier_skill_score", "log_loss", "ece", "test_period")},
              "week": next_week, "games": []}
    for r in [u for u in upcoming if u["week"] == next_week]:
        h, a, neutral = r["home_team"], r["away_team"], r.get("location") == "Neutral"
        entry = {"game": f"{NFL_NAME.get(a, a)} @ {NFL_NAME.get(h, h)}", "kickoff_et": kickoff(r).astimezone(ET)
                 .strftime("%a %b %d %I:%M %p"), "neutral": neutral, "lines": {}, "bets": []}
        if not (model.is_rated(h) and model.is_rated(a)):
            entry["note"] = "team not rated"
            report["games"].append(entry)
            continue
        mu_h, mu_a = model.expected_scores(h, a, neutral)
        entry["model_score"] = {"home": round(mu_h, 1), "away": round(mu_a, 1)}
        hm, am = r.get("home_moneyline"), r.get("away_moneyline")
        if hm and am:
            fair_h, fair_a = remove_vig([american_to_decimal(hm), american_to_decimal(am)])
            ph = model.probability("h2h", "home", None, h, a, neutral)[0]
            for side, team, odds, fair, mp in (("home", h, hm, fair_h, ph), ("away", a, am, fair_a, 1 - ph)):
                p = price(mp, 0.0, fair, odds)
                entry["bets"].append({"market": "moneyline", "pick": NFL_NAME.get(team, team), **p, "verdict": verdict(p)})
        sl, hso, aso = r.get("spread_line"), r.get("home_spread_odds"), r.get("away_spread_odds")
        if sl is not None and hso and aso:
            fair_h, fair_a = remove_vig([american_to_decimal(hso), american_to_decimal(aso)])
            for side, team, pt, odds, fair in (("home", h, -sl, hso, fair_h), ("away", a, sl, aso, fair_a)):
                w, push = model.probability("spreads", side, pt, h, a, neutral)
                p = price(w, push, fair, odds)
                entry["bets"].append({"market": "spread", "pick": f"{NFL_NAME.get(team, team)} {pt:+g}", **p,
                                      "verdict": verdict(p)})
        tl, oo, uo = r.get("total_line"), r.get("over_odds"), r.get("under_odds")
        if tl is not None and oo and uo:
            fair_o, fair_u = remove_vig([american_to_decimal(oo), american_to_decimal(uo)])
            for side, odds, fair in (("over", oo, fair_o), ("under", uo, fair_u)):
                w, push = model.probability("totals", side, tl, h, a, neutral)
                p = price(w, push, fair, odds)
                entry["bets"].append({"market": "total", "pick": f"{side.title()} {tl:g}", **p, "verdict": verdict(p)})
        report["games"].append(entry)

    b = report["backtest"]
    print(f"Real nflverse data: {len(games)} completed games, trained through {model.trained_through:%Y-%m-%d}")
    if b["status"] == "OK":
        print(f"Backtest (hypothetical, out-of-sample {b['n_predictions']} games): Brier {b['brier']:.4f} vs "
              f"baseline {b['baseline_brier']:.4f}, calibration error {b['ece']:.3f}")
    print(f"\nWEEK {next_week} - lines from the nflverse schedule file (book and time not stated)\n")
    for g in report["games"]:
        print(f"{g['game']}  {g['kickoff_et']} ET" + ("  [neutral]" if g["neutral"] else ""))
        if "model_score" in g:
            print(f"   model score: {g['model_score']['away']} - {g['model_score']['home']}")
        for x in g["bets"]:
            print(f"   {x['market']:<9} {x['pick']:<28} {x['odds']:>+5}  model {x['model']:.1%}  market {x['market_p']:.1%}"
                  f"  blend {x['final']:.1%}  EV {x['ev']:+.3f}  {x['verdict']}")
    print("\nNothing here is a QUALIFIED pick: the line source and time cannot be verified, and the model does not "
          "see injuries or news.")
    if out:
        Path(out).write_text(json.dumps(report, indent=1, default=str))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    main(out=ap.parse_args().json)
