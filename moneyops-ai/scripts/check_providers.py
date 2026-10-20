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
    odds, espn = build_providers(s, repo)
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
    for p in (odds, espn):
        repo.save_provider_status(p.status)
    sys.exit(0 if ok else 1)
