"""Data layer - download, store, and clean OHLCV data."""

from .downloader import download_ohlcv
from .storage import data_exists, load_ohlcv, save_ohlcv
from .fred import (
    FRED_SERIES,
    download_fred_series,
    download_fred_catalog,
    compute_net_liquidity,
    compute_global_m2,
    save_fred,
    load_fred,
)
from .dbnomics import (
    DBNOMICS_SERIES,
    download_dbnomics_series,
    download_dbnomics_catalog,
    save_dbnomics,
    load_dbnomics,
)


def download_universe(
    symbols: list[str],
    timeframe: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """
    Download OHLCV data for multiple symbols.

    Args:
        symbols: List of base asset symbols (e.g., ["BTC", "ETH"])
        timeframe: Candle interval (e.g., "1d")
        start_date: Start date as "YYYY-MM-DD" (default: 5 years ago)
        end_date: End date as "YYYY-MM-DD" (default: now)
        skip_existing: Skip symbols that already have data

    Returns:
        Dict mapping symbol to number of candles downloaded
    """
    results = {}

    for symbol in symbols:
        if skip_existing and data_exists(symbol, timeframe):
            print(f"[SKIP] {symbol} - data exists")
            continue

        print(f"[DOWNLOAD] {symbol}...", end=" ", flush=True)
        try:
            df = download_ohlcv(symbol, timeframe, start_date, end_date)
            if df.empty:
                print("no data")
                results[symbol] = 0
            else:
                save_ohlcv(df, symbol, timeframe)
                print(f"{len(df)} candles")
                results[symbol] = len(df)
        except Exception as e:
            print(f"error: {e}")
            results[symbol] = 0

    return results
