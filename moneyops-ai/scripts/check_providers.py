"""Make ONE real request to each provider and report the outcome.
Uses the free /v4/sports endpoint (no Odds API credits) and one ESPN call.
Nothing is printed from the key itself."""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from data.provider_base import ProviderError  # noqa: E402
from services.analysis import build_providers  # noqa: E402
from services.runtime import get_repo  # noqa: E402
from utils.timeutil import utcnow  # noqa: E402

if __name__ == "__main__":
    s, repo = get_settings(), get_repo()
    prov = build_providers(s, repo)
    odds, espn = prov.odds, prov.espn
    ok = True
    try:
        sports = odds.list_sports()
        active = [x["key"] for x in sports if x.get("active")]
        print(f"[OK] The Odds API: {len(sports)} sports listed, {len(active)} active. Quota remaining: "
              f"{odds.status.quota_remaining}")
    except ProviderError as exc:
        ok = False
        print(f"[FAIL] The Odds API: {exc}")
    try:
        now = utcnow()
        games = espn.scoreboard("NBA", "basketball/nba", now - timedelta(days=1), now + timedelta(days=1), ttl=0)
        print(f"[OK] ESPN: {len(games)} NBA games in the +/-1 day window")
    except ProviderError as exc:
        ok = False
        print(f"[FAIL] ESPN: {exc}")
    if prov.allsports.is_configured():
        try:
            key = prov.allsports.resolve_league("football", "Premier League", "England")
            print(f"[OK] AllSportsAPI: met=Leagues answered; Premier League key = {key}")
        except ProviderError as exc:
            ok = False
            print(f"[FAIL] AllSportsAPI: {exc}")
    else:
        print("[--] AllSportsAPI: ALLSPORTS_API_KEY not set (optional)")
    if prov.sportsipy.is_configured():
        try:
            games = prov.sportsipy.team_games("nba", "Boston Celtics", utcnow())
            print(f"[OK] sportsipy: Boston Celtics schedule parsed ({len(games)} completed games this season)")
        except ProviderError as exc:
            ok = False
            print(f"[FAIL] sportsipy: {exc}")
    else:
        print("[--] sportsipy: disabled (set SPORTSIPY_ENABLED=true; optional)")
    for p in prov.all():
        repo.save_provider_status(p.status)
    sys.exit(0 if ok else 1)
