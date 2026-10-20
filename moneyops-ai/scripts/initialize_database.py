"""Create the SQLite schema and the $25 virtual paper account."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from risk.bankroll import BankrollService  # noqa: E402
from services.runtime import get_repo  # noqa: E402

if __name__ == "__main__":
    s = get_settings()
    repo = get_repo()
    summary = BankrollService(repo, s).summary()
    print(f"Database ready at {s.database_url}")
    print(f"Virtual paper account #{summary['account_id']}: ${summary['balance']:.2f} ({summary['label']})")
    print(f"ODDS_API_KEY: {'configured' if s.odds_api_key else 'MISSING - set it in .env for live odds'}")
