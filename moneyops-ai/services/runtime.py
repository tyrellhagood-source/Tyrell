"""Process-wide singletons (settings, repository, logging) for the API,
dashboard and scripts."""
from __future__ import annotations

from functools import lru_cache

from config import get_settings
from database.repository import Repository
from utils.logging_setup import setup_logging


@lru_cache(maxsize=1)
def get_repo() -> Repository:
    s = get_settings()
    setup_logging(s.log_dir, [s.odds_api_key])
    repo = Repository(s.database_url)
    repo.init_db()
    return repo
