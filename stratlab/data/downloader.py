"""Fetch OHLCV data from Binance API."""

import time
from datetime import datetime, timezone

import pandas as pd
import requests

from ..config import DEFAULT_QUOTE_CURRENCY

BINANCE_API_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max per request


def download_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    quote_currency: str = DEFAULT_QUOTE_CURRENCY,
) -> pd.DataFrame:
    """
    Download OHLCV data from Binance.

    Args:
        symbol: Base asset symbol (e.g., "BTC")
        timeframe: Candle interval (e.g., "1d", "1h")
        start_date: Start date as "YYYY-MM-DD" (default: 5 years ago)
        end_date: End date as "YYYY-MM-DD" (default: now)
        quote_currency: Quote currency (default: "USDT")

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    pair = f"{symbol}{quote_currency}"

    # Default to 5 years of data
    if start_date is None:
        start_ts = int((datetime.now(timezone.utc).timestamp() - 5 * 365 * 24 * 3600) * 1000)
    else:
        start_ts = _date_to_ms(start_date)

    if end_date is None:
        end_ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    else:
        end_ts = _date_to_ms(end_date)

    all_candles = []
    current_start = start_ts

    while current_start < end_ts:
        params = {
            "symbol": pair,
            "interval": timeframe,
            "startTime": current_start,
            "endTime": end_ts,
            "limit": MAX_LIMIT,
        }

        response = requests.get(BINANCE_API_URL, params=params, timeout=30)
        response.raise_for_status()
        candles = response.json()

        if not candles:
            break

        all_candles.extend(candles)

        # Move start to after last candle
        current_start = candles[-1][0] + 1

        # Rate limiting
        time.sleep(0.1)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        all_candles,
        columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore",
        ],
    )

    # Keep only OHLCV columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Convert types
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df.set_index("timestamp").sort_index()

    return df


def _date_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD string to milliseconds timestamp."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
