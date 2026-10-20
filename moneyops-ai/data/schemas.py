"""Plain data structures exchanged between providers, agents and models.

Every record carries its `source` and `retrieved_at` so that agents can
attribute and age-check each data point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class OutcomePrice:
    market: str              # h2h | spreads | totals | player_points ...
    bookmaker: str           # provider bookmaker key
    bookmaker_title: str
    name: str                # team name, "Draw", "Over", "Under"
    price_decimal: float
    point: float | None
    last_update: datetime
    description: str | None = None  # player name for props
    provider: str = "the-odds-api"  # which data provider supplied this quote


@dataclass
class Event:
    event_id: str
    sport_code: str          # internal code e.g. NBA
    sport_key: str           # provider sport key e.g. basketball_nba
    league: str
    commence_time: datetime
    home_team: str
    away_team: str
    source: str
    retrieved_at: datetime
    prices: list[OutcomePrice] = field(default_factory=list)

    @property
    def name(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    def is_live(self, now: datetime) -> bool:
        return self.commence_time <= now


@dataclass(frozen=True)
class ScheduledGame:
    game_id: str
    sport_code: str
    start_time: datetime
    home_team: str
    away_team: str
    state: str               # pre | in | post
    status_detail: str
    completed: bool
    home_score: int | None
    away_score: int | None
    neutral_site: bool
    source: str
    retrieved_at: datetime


@dataclass(frozen=True)
class HistoricalGame:
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    neutral_site: bool
    source: str


@dataclass(frozen=True)
class InjuryReport:
    team: str
    player: str
    status: str
    detail: str
    source: str
    retrieved_at: datetime


@dataclass(frozen=True)
class FinalScore:
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    completed: bool
    source: str
    retrieved_at: datetime
