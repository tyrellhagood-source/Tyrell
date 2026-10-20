"""The Odds API (v4) provider: sports list, odds, event (prop) odds, scores.

Docs: https://the-odds-api.com/liveapi/guides/v4/
Usage cost (per the docs): /sports and /events are free; /odds costs
markets x regions credits; /scores costs 1 (2 with daysFrom). Remaining quota
is reported in the x-requests-remaining / x-requests-used / x-requests-last
response headers, which this provider tracks and enforces.
"""
from __future__ import annotations

import logging
from datetime import datetime

from data.provider_base import (FetchMeta, ProviderBase, ProviderBudgetExceeded, ProviderError,
                                ProviderNotConfigured)
from data.schemas import Event, FinalScore, OutcomePrice
from data.sports_registry import sport_for_odds_key
from utils.timeutil import parse_iso

log = logging.getLogger(__name__)


def parse_odds_response(body: list, retrieved_at: datetime, source: str = "the-odds-api") -> list[Event]:
    """Convert a /odds or /events/{id}/odds payload into Events.

    Malformed entries are skipped (never patched with guessed values)."""
    if isinstance(body, dict):
        body = [body]
    events: list[Event] = []
    for item in body or []:
        try:
            sport_key = item["sport_key"]
            commence = parse_iso(item["commence_time"])
            home, away = item["home_team"], item["away_team"]
        except (KeyError, TypeError):
            log.warning("Skipping malformed odds event: %s", str(item)[:200])
            continue
        if commence is None or not home or not away:
            continue
        cfg = sport_for_odds_key(sport_key, title=item.get("sport_title", ""))
        ev = Event(event_id=str(item.get("id")), sport_code=cfg.code, sport_key=sport_key, league=cfg.league,
                   commence_time=commence, home_team=home, away_team=away, source=source,
                   retrieved_at=retrieved_at)
        for book in item.get("bookmakers") or []:
            book_update = parse_iso(book.get("last_update"))
            for market in book.get("markets") or []:
                m_update = parse_iso(market.get("last_update")) or book_update
                if m_update is None:
                    continue  # no timestamp -> cannot age-check -> unusable
                for o in market.get("outcomes") or []:
                    price = o.get("price")
                    if not isinstance(price, (int, float)) or price <= 1.0:
                        continue
                    point = o.get("point")
                    ev.prices.append(OutcomePrice(
                        market=market.get("key", ""), bookmaker=book.get("key", ""),
                        bookmaker_title=book.get("title", book.get("key", "")), name=o.get("name", ""),
                        price_decimal=float(price), point=float(point) if point is not None else None,
                        last_update=m_update, description=o.get("description")))
        events.append(ev)
    return events


def parse_scores_response(body: list, retrieved_at: datetime) -> dict[str, FinalScore]:
    out: dict[str, FinalScore] = {}
    for item in body or []:
        scores = item.get("scores")
        if not item.get("completed") or not scores:
            continue
        by_name = {s.get("name"): s.get("score") for s in scores}
        try:
            hs, as_ = int(by_name[item["home_team"]]), int(by_name[item["away_team"]])
        except (KeyError, TypeError, ValueError):
            continue
        out[str(item["id"])] = FinalScore(item["home_team"], item["away_team"], hs, as_, True,
                                          "the-odds-api/scores", retrieved_at)
    return out


class OddsApiProvider(ProviderBase):
    name = "the-odds-api"

    def __init__(self, settings, cache=None, session=None):
        super().__init__(settings, cache, session)
        self.credits_spent_this_run = 0

    def is_configured(self) -> bool:
        return bool(self.settings.odds_api_key)

    def _require_key(self) -> None:
        if not self.is_configured():
            raise ProviderNotConfigured("ODDS_API_KEY is not set; live odds are unavailable.")

    def _observe_headers(self, headers: dict) -> None:
        for attr, header in (("quota_remaining", "x-requests-remaining"), ("quota_used", "x-requests-used"),
                             ("last_request_cost", "x-requests-last")):
            if header in headers:
                try:
                    setattr(self.status, attr, int(float(headers[header])))
                except ValueError:
                    pass

    def _cacheable_headers(self, headers: dict) -> dict:
        return {k: v for k, v in headers.items() if k.startswith("x-requests")}

    def _url(self, path: str) -> str:
        return f"{self.settings.odds_api_base_url.rstrip('/')}{path}"

    def _check_budget(self, cost: int) -> None:
        s = self.settings
        if self.credits_spent_this_run + cost > s.max_credits_per_run:
            raise ProviderBudgetExceeded(
                f"Skipped: request would cost {cost} credits; run budget {s.max_credits_per_run} "
                f"(spent {self.credits_spent_this_run}). Raise MAX_CREDITS_PER_RUN to scan more.")
        remaining = self.status.quota_remaining
        if remaining is not None and remaining - cost < s.min_credits_reserve:
            raise ProviderBudgetExceeded(
                f"Skipped: only {remaining} credits left, reserve is {s.min_credits_reserve}.")

    def _paid_get(self, path: str, params: dict, cost: int, ttl: int) -> tuple[object, FetchMeta]:
        self._require_key()
        self._check_budget(cost)
        body, meta = self.get_json(self._url(path), {"apiKey": self.settings.odds_api_key, **params}, ttl)
        if not meta.from_cache:
            self.credits_spent_this_run += cost
        return body, meta

    @staticmethod
    def odds_cost(markets, regions) -> int:
        return max(1, len(markets)) * max(1, len(regions))

    def list_sports(self) -> list[dict]:
        """Free endpoint (does not use quota)."""
        self._require_key()
        body, _ = self.get_json(self._url("/v4/sports"), {"apiKey": self.settings.odds_api_key}, 3600)
        return list(body or [])

    def get_odds(self, sport_key: str, markets=None, regions=None) -> list[Event]:
        markets = tuple(markets or self.settings.odds_markets)
        regions = tuple(regions or self.settings.odds_regions)
        params = {"regions": ",".join(regions), "markets": ",".join(markets),
                  "oddsFormat": "decimal", "dateFormat": "iso"}
        if self.settings.odds_bookmakers:
            params.pop("regions")
            params["bookmakers"] = ",".join(self.settings.odds_bookmakers)
        body, meta = self._paid_get(f"/v4/sports/{sport_key}/odds", params, self.odds_cost(markets, regions),
                                    self.settings.odds_cache_ttl_seconds)
        if not isinstance(body, list):
            raise ProviderError(f"Unexpected odds payload type {type(body).__name__}")
        return parse_odds_response(body, meta.fetched_at)

    def get_event_odds(self, sport_key: str, event_id: str, markets) -> list[Event]:
        markets = tuple(markets)
        regions = tuple(self.settings.odds_regions)
        params = {"regions": ",".join(regions), "markets": ",".join(markets),
                  "oddsFormat": "decimal", "dateFormat": "iso"}
        body, meta = self._paid_get(f"/v4/sports/{sport_key}/events/{event_id}/odds", params,
                                    self.odds_cost(markets, regions), self.settings.odds_cache_ttl_seconds)
        return parse_odds_response(body, meta.fetched_at)

    def get_scores(self, sport_key: str, days_from: int = 3) -> dict[str, FinalScore]:
        cost = 2 if days_from else 1
        params = {"daysFrom": days_from, "dateFormat": "iso"} if days_from else {"dateFormat": "iso"}
        body, meta = self._paid_get(f"/v4/sports/{sport_key}/scores", params, cost, 600)
        return parse_scores_response(body, meta.fetched_at)
