"""Global configuration and constants."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default settings
DEFAULT_TIMEFRAME = "1d"
DEFAULT_QUOTE_CURRENCY = "USDT"


def _load_env() -> None:
    """Load variables from .env file into os.environ."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_env()

# API keys
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")
