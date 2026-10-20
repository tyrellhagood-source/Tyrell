"""Shared agent data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

from data.schemas import Event, HistoricalGame, InjuryReport, OutcomePrice, ScheduledGame
from data.sports_registry import SportConfig
from utils.timeutil import utcnow

RESEARCH = "agent1_research"
VERIFY = "agent2_verification"
STATS = "agent3_statistics"
PROB = "agent4_probability"
AUDIT = "agent5_auditor"
AGENT_NAMES = {
    RESEARCH: "Agent 1 - Live Data Researcher",
    VERIFY: "Agent 2 - Independent Fact-Checker",
    STATS: "Agent 3 - Historical Statistician",
    PROB: "Agent 4 - Probability & Risk Engineer",
    AUDIT: "Agent 5 - Chief Auditor",
}
# Verdict that counts as an endorsement when tracking each agent's record.
ENDORSE = {RESEARCH: "COMPLETE", VERIFY: "VERIFIED", STATS: "SUPPORTS", PROB: "PASS", AUDIT: "QUALIFIED"}


@dataclass
class AgentReport:
    agent: str
    verdict: str
    score: float
    summary: str
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d


@dataclass
class SportContext:
    cfg: SportConfig
    history: list[HistoricalGame] = field(default_factory=list)
    model: object | None = None
    bootstrap_models: list = field(default_factory=list)
    backtest: dict | None = None
    injuries: list[InjuryReport] | None = None
    injuries_error: str | None = None
    schedules: dict[str, list[ScheduledGame]] = field(default_factory=dict)   # source -> games
    schedule_errors: dict[str, str] = field(default_factory=dict)              # source -> error
    league_keys: dict[str, str] = field(default_factory=dict)                  # provider -> league id
    external_probs: dict[str, dict] = field(default_factory=dict)              # event_id -> {home, draw, away}
    prop_module: str | None = None                                             # sportsipy module
    prop_logs: dict[str, list[dict]] = field(default_factory=dict)             # team -> player game lines
    prop_errors: dict[str, str] = field(default_factory=dict)                  # team -> error
    team_map: dict[str, str | None] = field(default_factory=dict)   # odds name -> history name
    notes: list[str] = field(default_factory=list)

    def history_name(self, team: str) -> str | None:
        return self.team_map.get(team)


@dataclass
class Candidate:
    key: str
    event: Event
    cfg: SportConfig
    market: str
    selection: str
    side: str
    point: float | None
    description: str | None
    books: list[OutcomePrice]
    best: OutcomePrice | None = None
    fair_prob: float | None = None
    fair_by_book: dict = field(default_factory=dict)
    is_live: bool = False
    neutral_site: bool = False
    reports: dict[str, AgentReport] = field(default_factory=dict)
    status: str = "PENDING"
    consensus_score: float = 0.0
    stake: float = 0.0

    @property
    def is_prop(self) -> bool:
        return self.market.startswith("player_")

    @property
    def label(self) -> str:
        pt = "" if self.point is None else (f" {self.point:+g}" if self.market == "spreads" else f" {self.point:g}")
        who = f"{self.description} " if self.description else ""
        return f"{who}{self.selection}{pt}"

    def verdict(self, agent: str) -> str | None:
        r = self.reports.get(agent)
        return r.verdict if r else None
