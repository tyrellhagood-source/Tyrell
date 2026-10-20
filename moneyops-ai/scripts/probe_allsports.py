"""Confirm AllSportsAPI's real response fields with YOUR key (run once).

Calls Leagues, Fixtures, Odds and Probabilities for football (today +/- 2
days, Premier League if resolvable), saves the raw responses to
logs/allsports_samples/ (API key redacted) and reports which fields the
MoneyOps parsers expect are present or missing. Uses ~4 requests.
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from data.provider_base import ProviderError  # noqa: E402
from services.analysis import build_providers  # noqa: E402
from services.runtime import get_repo  # noqa: E402
from utils.security import redact  # noqa: E402
from utils.timeutil import utcnow  # noqa: E402

EXPECTED = {
    "Fixtures": ["event_key", "event_date", "event_time", "event_home_team", "event_away_team", "event_status",
                 "event_final_result", "event_ft_result", "league_name", "league_key"],
    "Odds": ["odd_bookmakers", "odd_epoch", "odd_1", "odd_x", "odd_2", "o+2.5", "u+2.5"],
    "Probabilities": ["event_key", "event_HW", "event_D", "event_AW"],
    "Leagues": ["league_key", "league_name", "country_name"],
}

if __name__ == "__main__":
    s = get_settings()
    a = build_providers(s, get_repo()).allsports
    if not a.is_configured():
        sys.exit("Set ALLSPORTS_API_KEY in .env first.")
    out_dir = s.log_dir / "allsports_samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    now = utcnow()
    league = None
    try:
        league = a.resolve_league("football", "Premier League", "England")
    except ProviderError as exc:
        print("Leagues failed:", exc)
    params = {"from": f"{now - timedelta(days=2):%Y-%m-%d}", "to": f"{now + timedelta(days=2):%Y-%m-%d}"}
    if league:
        params["leagueId"] = league
    for met in ("Leagues", "Fixtures", "Odds", "Probabilities"):
        try:
            result, _ = a._call("football", met, {} if met == "Leagues" else params, 0)
        except ProviderError as exc:
            print(f"{met}: FAILED {exc}")
            continue
        (out_dir / f"{met.lower()}.json").write_text(redact(json.dumps(result, indent=1)[:200000]))
        if isinstance(result, dict):     # Odds: {match_key: [rows]}
            rows = [r for v in result.values() if isinstance(v, list) for r in v]
        else:
            rows = result if isinstance(result, list) else []
        keys = set().union(*(r.keys() for r in rows if isinstance(r, dict))) if rows else set()
        missing = [k for k in EXPECTED[met] if k not in keys]
        print(f"{met}: {len(rows)} rows; expected fields missing: {missing or 'none'}")
    print(f"Raw samples saved to {out_dir}. If fields are missing, the parsers in data/allsports_provider.py "
          "need adjusting - until then MoneyOps simply gets no data from that call (fail-closed).")
