"""Global configuration and constants."""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default settings
DEFAULT_TIMEFRAME = "1d"
DEFAULT_QUOTE_CURRENCY = "USDT"
