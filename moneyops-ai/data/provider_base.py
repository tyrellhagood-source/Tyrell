"""Provider abstraction: HTTP with retries, rate limits, caching, status."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

import requests

from utils.security import redact
from utils.timeutil import utcnow

log = logging.getLogger(__name__)


class ProviderError(Exception):
    """Any failure fetching data. The message is always secret-redacted."""

    def __init__(self, message: str):
        super().__init__(redact(message))


class ProviderNotConfigured(ProviderError):
    pass


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    pass


class ProviderBudgetExceeded(ProviderError):
    pass


@dataclass
class ProviderStatus:
    name: str
    configured: bool = False
    last_success: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    requests_made: int = 0
    cache_hits: int = 0
    quota_remaining: int | None = None
    quota_used: int | None = None
    last_request_cost: int | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class FetchMeta:
    url: str
    fetched_at: datetime
    from_cache: bool
    headers: dict


class ProviderBase:
    name = "base"
    secret_params: tuple[str, ...] = ("apiKey",)

    def __init__(self, settings, cache=None, session: requests.Session | None = None):
        self.settings = settings
        self.cache = cache
        self.session = session or requests.Session()
        self.status = ProviderStatus(self.name, configured=self.is_configured())

    def is_configured(self) -> bool:
        return True

    def _record_error(self, message: str) -> None:
        self.status.last_error = redact(message)
        self.status.last_error_at = utcnow()
        log.warning("%s error: %s", self.name, redact(message))

    def get_json(self, url: str, params: dict | None = None, ttl_seconds: int = 0,
                 headers: dict | None = None) -> tuple[object, FetchMeta]:
        params = dict(params or {})
        public_params = {k: v for k, v in params.items() if k not in self.secret_params}
        cache_key = None
        if self.cache is not None and ttl_seconds > 0:
            cache_key = self.cache.make_key(self.name, url, public_params)
            hit = self.cache.get(cache_key)
            if hit is not None:
                body, headers, fetched_at = hit
                self.status.cache_hits += 1
                return body, FetchMeta(url, fetched_at, True, headers)

        attempts = max(1, self.settings.http_max_retries + 1)
        last_exc: Exception | None = None
        for attempt in range(attempts):
            try:
                self._before_request()
                kwargs = {"headers": headers} if headers else {}
                resp = self.session.get(url, params=params, timeout=self.settings.http_timeout_seconds, **kwargs)
            except requests.RequestException as exc:  # network / TLS / proxy failure
                last_exc = ProviderError(f"{self.name}: network error for {url}: {exc}")
                time.sleep(min(8, 0.5 * 2 ** attempt))
                continue
            self.status.requests_made += 1
            headers = {k.lower(): v for k, v in resp.headers.items()}
            self._observe_headers(headers)
            if resp.status_code in (401, 403):
                err = ProviderAuthError(f"{self.name}: HTTP {resp.status_code} (check API key / plan) for {url}")
                self._record_error(str(err))
                raise err
            if resp.status_code == 429:
                retry_after = float(headers.get("retry-after", 2 ** attempt))
                last_exc = ProviderRateLimited(f"{self.name}: rate limited (HTTP 429)")
                if attempt + 1 < attempts:
                    time.sleep(min(30, retry_after))
                continue
            if resp.status_code >= 500:
                last_exc = ProviderError(f"{self.name}: HTTP {resp.status_code} for {url}")
                time.sleep(min(8, 0.5 * 2 ** attempt))
                continue
            if resp.status_code >= 400:
                err = ProviderError(f"{self.name}: HTTP {resp.status_code} for {url}: {resp.text[:300]}")
                self._record_error(str(err))
                raise err
            try:
                body = resp.json()
            except ValueError as exc:
                err = ProviderError(f"{self.name}: invalid JSON from {url}: {exc}")
                self._record_error(str(err))
                raise err
            now = utcnow()
            self.status.last_success = now
            if cache_key is not None:
                self.cache.set(cache_key, self.name, redact(url), body, self._cacheable_headers(headers), ttl_seconds)
            return body, FetchMeta(url, now, False, headers)
        err = last_exc or ProviderError(f"{self.name}: request failed for {url}")
        self._record_error(str(err))
        raise err

    def _before_request(self) -> None:
        """Hook run before every real (non-cached) HTTP attempt: throttling/budgets."""

    def _observe_headers(self, headers: dict) -> None:
        pass

    def _cacheable_headers(self, headers: dict) -> dict:
        return {}
