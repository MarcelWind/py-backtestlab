"""Read/write parquet files for OHLCV data."""

from pathlib import Path

import pandas as pd

from ..config import DATA_DIR, DEFAULT_QUOTE_CURRENCY


def get_data_path(symbol: str, timeframe: str = "1d", quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> Path:
    """Get the parquet file path for a symbol/timeframe."""
    pair = f"{symbol}{quote_currency}"
    return DATA_DIR / pair / f"{timeframe}.parquet"


def save_ohlcv(df: pd.DataFrame, symbol: str, timeframe: str = "1d", quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> Path:
    """
    Save OHLCV DataFrame to parquet.

    Args:
        df: DataFrame with OHLCV data (timestamp index)
        symbol: Base asset symbol (e.g., "BTC")
        timeframe: Candle interval (e.g., "1d")
        quote_currency: Quote currency (default: "USDT")

    Returns:
        Path to saved file
    """
    path = get_data_path(symbol, timeframe, quote_currency)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return path


def load_ohlcv(symbol: str, timeframe: str = "1d", quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> pd.DataFrame:
    """
    Load OHLCV DataFrame from parquet.

    Args:
        symbol: Base asset symbol (e.g., "BTC")
        timeframe: Candle interval (e.g., "1d")
        quote_currency: Quote currency (default: "USDT")

    Returns:
        DataFrame with OHLCV data, or empty DataFrame if not found
    """
    path = get_data_path(symbol, timeframe, quote_currency)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def data_exists(symbol: str, timeframe: str = "1d", quote_currency: str = DEFAULT_QUOTE_CURRENCY) -> bool:
    """Check if data file exists for a symbol/timeframe."""
    return get_data_path(symbol, timeframe, quote_currency).exists()
