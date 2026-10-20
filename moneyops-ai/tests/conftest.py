"""Shared fixtures.

NOTE: files under tests/fixtures are HAND-WRITTEN sample payloads that follow
the documented provider formats. They exist only to test parsing and logic
offline and are never used by the application at runtime.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config import Settings
from data.schemas import HistoricalGame
from database.repository import Repository

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path):
    return Settings(odds_api_key="test-key-123456", database_url=f"sqlite:///{tmp_path / 'test.db'}",
                    log_dir=tmp_path / "logs", bootstrap_samples=8)


@pytest.fixture
def repo(settings):
    r = Repository(settings.database_url)
    r.init_db()
    return r


@pytest.fixture
def load_fixture():
    def _load(name):
        return json.loads((FIXTURES / name).read_text())
    return _load


def synthetic_league(n_teams=10, rounds=8, seed=1, start=None, scale=100.0, sd=11.0, poisson=False):
    """Synthetic games from KNOWN team strengths - used only to test that the
    model recovers structure (unit tests), never as app data."""
    rng = random.Random(seed)
    start = start or datetime(2025, 11, 1, tzinfo=timezone.utc)
    teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
    strength = {t: (i - n_teams / 2) * (0.08 if poisson else 1.5) for i, t in enumerate(teams)}
    games, day, gid = [], 0, 0
    for _ in range(rounds):
        for h in teams:
            for a in teams:
                if h == a or rng.random() > 0.25:
                    continue
                day += 1
                gid += 1
                date = start + timedelta(hours=8 * day)
                if poisson:
                    import numpy as np
                    nr = np.random.default_rng(gid)
                    hs = int(nr.poisson(max(0.2, 1.5 * (2.718 ** (strength[h] - strength[a] + 0.1)))))
                    as_ = int(nr.poisson(max(0.2, 1.3 * (2.718 ** (strength[a] - strength[h])))))
                else:
                    hs = int(round(rng.gauss(scale + strength[h] - strength[a] + 2.5, sd)))
                    as_ = int(round(rng.gauss(scale + strength[a] - strength[h], sd)))
                games.append(HistoricalGame(str(gid), date, h, a, hs, as_, False, "synthetic-test"))
    return teams, games
