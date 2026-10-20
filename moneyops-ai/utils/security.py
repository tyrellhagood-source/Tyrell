"""Secret redaction for logs, cached URLs and error messages."""
from __future__ import annotations

import logging
import re

_PATTERNS = [
    re.compile(r"(?i)(apiKey|api_key|apikey|key|token|secret)=([^&\s\"']+)"),
    re.compile(r"(?i)(x-api-key|authorization|ocp-apim-subscription-key)(\"?\s*[:=]\s*\"?)([^\s\"',}]+)"),
]
_known_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    if value and len(value) >= 6:
        _known_secrets.add(value)


def redact(text: object) -> str:
    s = str(text)
    for secret in _known_secrets:
        s = s.replace(secret, "***REDACTED***")
    s = _PATTERNS[0].sub(lambda m: f"{m.group(1)}=***REDACTED***", s)
    s = _PATTERNS[1].sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", s)
    return s


class SecretRedactingFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed log call
            return True
        record.msg = redact(message)
        record.args = ()
        return True
