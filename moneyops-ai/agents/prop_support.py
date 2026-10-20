"""Shared lookup of a prop candidate's player history (used by Agents 3 and 4)."""
from __future__ import annotations

from agents.base import Candidate, SportContext
from data.sportsipy_provider import PROP_STATS
from models.props import player_values
from utils.teams import normalize_team


def prop_rows(c: Candidate, ctx: SportContext):
    """Return (rows, attrs, team, opponent) or (None, reason, None, None)."""
    module = ctx.prop_module
    if not module:
        return None, f"no player game-log source for {ctx.cfg.code}", None, None
    attrs = PROP_STATS.get(module, {}).get(c.market)
    if not attrs:
        return None, f"prop market {c.market} not mapped to a box-score stat", None, None
    player = normalize_team(c.description or "")
    for team, opp in ((c.event.home_team, c.event.away_team), (c.event.away_team, c.event.home_team)):
        logs = ctx.prop_logs.get(team) or []
        if any(normalize_team(l["player"]) == player for l in logs):
            # only games strictly before kick-off (logs are already filtered by date)
            return player_values(logs, c.description, attrs, module), attrs, team, opp
    return None, f"no game logs for {c.description}", None, None
