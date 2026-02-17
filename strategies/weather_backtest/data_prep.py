from typing import Callable

import pandas as pd

from fetch_data import load_zip


def _load_event_rows(event_slug: str) -> pd.DataFrame:
    df = load_zip()
    df = df[df["event_slug"] == event_slug].copy()
    if df.empty:
        raise ValueError(f"No rows found for event_slug={event_slug!r}")

    if "datetime" in df.columns:
        ts = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"]).sort_values("_ts")

    close_raw = df["close"] if "close" in df.columns else pd.Series(index=df.index, dtype=float)
    volume_raw = df["volume"] if "volume" in df.columns else pd.Series(0.0, index=df.index, dtype=float)
    vwap_raw = df["vwap"] if "vwap" in df.columns else pd.Series(index=df.index, dtype=float)

    df["close"] = pd.to_numeric(close_raw, errors="coerce")
    df["volume"] = pd.to_numeric(volume_raw, errors="coerce").fillna(0.0)
    df["vwap"] = pd.to_numeric(vwap_raw, errors="coerce")
    return df


def _build_matrices(
    df: pd.DataFrame,
    resample_rule: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    last_value: Callable[[pd.Series], float] = lambda x: float(x.iloc[-1])
    closes = df.pivot_table(index="_ts", columns="market", values="close", aggfunc=last_value)

    # Raw volume matrix (sum over duplicate timestamps if present).
    volumes = df.pivot_table(index="_ts", columns="market", values="volume", aggfunc="sum")

    # Source-row VWAP weighted by volume.
    df = df.copy()
    df["src_vwap_value"] = df["vwap"] * df["volume"]
    src_vwap_values = df.pivot_table(index="_ts", columns="market", values="src_vwap_value", aggfunc="sum")

    closes = closes.sort_index().ffill().dropna(axis=1, how="all")
    volumes = volumes.reindex(closes.index).fillna(0.0).reindex(columns=closes.columns)
    src_vwap_values = src_vwap_values.reindex(closes.index).reindex(columns=closes.columns)

    # Instantaneous bar VWAP from source column, volume-weighted over
    # duplicate timestamps for each market.
    vwaps = src_vwap_values.divide(volumes.where(volumes > 0.0))

    min_rows = 10
    valid_cols = closes.notna().sum() >= min_rows
    closes = closes.loc[:, valid_cols]
    volumes = volumes.loc[:, valid_cols]
    vwaps = vwaps.loc[:, valid_cols]

    closes = closes.dropna(axis=0, how="any")
    volumes = volumes.reindex(closes.index).fillna(0.0)
    vwaps = vwaps.reindex(closes.index)

    if not resample_rule:
        return closes, vwaps, volumes

    closes = closes.resample(resample_rule).last().ffill()
    volumes = volumes.resample(resample_rule).sum().fillna(0.0).reindex(columns=closes.columns)

    # Resampled VWAP = sum(vwap * volume) / sum(volume), still no forward-fill.
    weighted = (vwaps * volumes.reindex(vwaps.index).fillna(0.0)).resample(resample_rule).sum()
    vwaps = weighted.divide(volumes.where(volumes > 0.0))
    vwaps = vwaps.reindex(columns=closes.columns)

    closes = closes.dropna(axis=0, how="any")
    volumes = volumes.reindex(closes.index).fillna(0.0)
    vwaps = vwaps.reindex(closes.index)
    return closes, vwaps, volumes


def load_event_ohlcv(event_slug: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load one Polymarket event and build close/vwap/volume matrices."""
    df = _load_event_rows(event_slug)
    return _build_matrices(df, resample_rule=None)


def load_event_ohlcv_resampled(
    event_slug: str,
    resample_rule: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load event OHLCV and optionally downsample bars.

    Args:
        event_slug: Event identifier.
        resample_rule: Pandas rule (e.g. "5min", "10min"). If None, no resample.
    """
    df = _load_event_rows(event_slug)
    return _build_matrices(df, resample_rule=resample_rule)
