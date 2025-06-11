"""Shared utilities for the crypto-models service."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
SIGNALS_DIR = ROOT_DIR / "signals"
DATA_DIR = ROOT_DIR / "data"
FEATURES_DIR = ROOT_DIR / "features"
BUCKET_MAPPING_FILE = ROOT_DIR / "bucket_mapping.csv"
DATA_API_URL = os.getenv("DATA_API_URL", "http://127.0.0.1:8001")
PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "http://127.0.0.1:9091")

_DUR_RE = re.compile(r"^(\d+)([smhd])$")

# Ensure runtime directories exist
for _p in (MODELS_DIR, SIGNALS_DIR, DATA_DIR, FEATURES_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def env(key: str, default: str | None = None) -> str:
    """Read environment variable with an optional default."""
    val = os.getenv(key, default)
    if val is None:
        raise RuntimeError(f"Environment variable '{key}' must be set")
    return val


def dump_json(obj: Any, path: Path) -> None:
    """Pretty-print JSON to *path* with unix-style newlines."""
    path.write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False))


def timestamp() -> int:
    """Return current epoch seconds (int)."""
    return int(time.time())


def parse_duration(expr: str) -> int:  # noqa: D401
    """Return duration in seconds, supports e.g. '15m', '4h', '1d'."""
    m = _DUR_RE.match(expr)
    if not m:
        raise ValueError(f"bad duration: {expr}")
    qty, unit = int(m.group(1)), m.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return qty * multipliers[unit]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_session(retries: int = 3, backoff: float = 0.3) -> requests.Session:  # noqa: D401
    """Return Requests session with retry/backoff."""
    sess = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    return sess
