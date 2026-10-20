"""HTTP response cache persisted in SQLite (respecting provider TTLs)."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta

from database.models import ApiCache
from utils.timeutil import ensure_utc, utcnow


class ResponseCache:
    def __init__(self, session_factory):
        self._sf = session_factory

    @staticmethod
    def make_key(provider: str, url: str, params: dict) -> str:
        raw = provider + url + json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str):
        with self._sf() as s:
            row = s.get(ApiCache, key)
            if row is None or ensure_utc(row.expires_at) <= utcnow():
                return None
            return json.loads(row.body), dict(json.loads(row.headers or "{}")), ensure_utc(row.fetched_at)

    def set(self, key: str, provider: str, url_redacted: str, body, headers: dict, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        now = utcnow()
        with self._sf() as s:
            row = s.get(ApiCache, key) or ApiCache(key=key)
            row.provider = provider
            row.url = url_redacted
            row.body = json.dumps(body)
            row.headers = json.dumps(headers)
            row.fetched_at = now
            row.expires_at = now + timedelta(seconds=ttl_seconds)
            s.merge(row)
            s.commit()
