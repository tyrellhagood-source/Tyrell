"""Logging configuration with secret redaction on every handler."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.security import SecretRedactingFilter, register_secret

_configured = False


def setup_logging(log_dir: Path, api_keys: list[str] | None = None, level: int = logging.INFO) -> None:
    global _configured
    for key in api_keys or []:
        register_secret(key)
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    redactor = SecretRedactingFilter()
    root = logging.getLogger()
    root.setLevel(level)
    file_handler = RotatingFileHandler(log_dir / "moneyops.log", maxBytes=2_000_000, backupCount=5)
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(fmt)
        handler.addFilter(redactor)
        root.addHandler(handler)
    # urllib3 logs full URLs (which contain the apiKey query param) at DEBUG.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    _configured = True
