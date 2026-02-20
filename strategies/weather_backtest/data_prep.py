from typing import Callable

import io
import zipfile

import pandas as pd

_SUFFIX_RE = r"__(?:yes|no)$"


def load_zip(zip_path: str = "data.zip") -> pd.DataFrame:
    """Load all parquet files from a zip into a single DataFrame.

    Adds 'event_slug' and 'market' columns derived from the file paths.
    """
    frames = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            parts = name.replace("\\", "/").split("/")
            if len(parts) != 3 or not parts[2].endswith(".parquet") or parts[1] == "unknown":
                continue
            event_slug = parts[1]
            market = parts[2].removesuffix(".parquet")
            df = pd.read_parquet(io.BytesIO(zf.read(name)))
            df["event_slug"] = event_slug
            df["market"] = market
            frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No parquet data found in {zip_path}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["event_slug", "market", "timestamp"]).reset_index(drop=True)
    return combined


def normalize_market_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "market" not in out.columns:
        raise ValueError("Expected 'market' column in dataframe")

    market_raw = out["market"].astype(str)
    # Preserve the original market string (including any __yes/__no suffix)
    out["market_raw"] = market_raw
    has_suffix = market_raw.str.contains(_SUFFIX_RE, regex=True)
    suffix_outcome = market_raw.str.extract(r"__(yes|no)$", expand=False)

    out["market"] = market_raw.str.replace(_SUFFIX_RE, "", regex=True)

    if "outcome" not in out.columns:
        out["outcome"] = ""

    out["outcome"] = out["outcome"].fillna("").astype(str).str.strip().str.lower()

    fill_mask = out["outcome"].eq("") & suffix_outcome.notna()
    out.loc[fill_mask, "outcome"] = suffix_outcome[fill_mask]

    conflict_mask = suffix_outcome.notna() & out["outcome"].ne("") & out["outcome"].ne(suffix_outcome)
    out = out[~conflict_mask].copy()

    out["_has_suffix"] = has_suffix.loc[out.index].astype(int)

    if "timestamp" in out.columns:
        sort_cols = ["_has_suffix", "timestamp"]
        out = out.sort_values(sort_cols, ascending=[False, True])

        subset_cols: list[str] = []
        for col in ["event_slug", "asset_id", "market", "outcome", "timestamp"]:
            if col in out.columns:
                subset_cols.append(col)

        if subset_cols:
            out = out.drop_duplicates(subset=subset_cols, keep="first")

        out = out.drop(columns=["_has_suffix"], errors="ignore").reset_index(drop=True)
    
    return out


    def load_and_prepare(
        zip_path: str = "data.zip",
        event_slug: str | None = None,
    ) -> pd.DataFrame:
        df = load_zip(zip_path)
        if event_slug:
            df = df[df["event_slug"] == event_slug].copy()
        return normalize_market_outcomes(df)


def pick_plot_frame(df: pd.DataFrame, prefer_outcome: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if prefer_outcome and "outcome" in out.columns:
        target = prefer_outcome.strip().lower()
        out = out[out["outcome"].astype(str).str.lower() == target].copy()
    return out


def _load_event_rows(event_slug: str, prefer_outcome: str | None = None) -> pd.DataFrame:
    df = load_zip()
    df = df[df["event_slug"] == event_slug].copy()
    if df.empty:
        raise ValueError(f"No rows found for event_slug={event_slug!r}")

    # Normalize market outcome suffixes and optionally filter to preferred outcome
    try:
        df = normalize_market_outcomes(df)
    except Exception:
        # If normalization fails, continue with raw df
        pass

    if prefer_outcome:
        try:
            df = pick_plot_frame(df, prefer_outcome=prefer_outcome)
        except Exception:
            pass

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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:

    last_value: Callable[[pd.Series], float] = lambda x: float(x.iloc[-1])
    closes = df.pivot_table(index="_ts", columns="market", values="close", aggfunc=last_value)

    # Raw volume matrix (sum over duplicate timestamps if present).
    volumes = df.pivot_table(index="_ts", columns="market", values="volume", aggfunc="sum")

    # Optional buy/sell per-update volumes, if present in raw rows.
    buy_volumes = None
    sell_volumes = None
    # Build per-update buy/sell matrices keyed by "market__outcome" so
    # yes and no tokens get separate columns (e.g. "35-f-or-below__yes"
    # and "35-f-or-below__no"). Fall back to plain market name when the
    # outcome column is absent or empty.
    if "buy_volume" in df.columns and "sell_volume" in df.columns:
        df_bv = df.copy()
        if "outcome" in df_bv.columns:
            mask = df_bv["outcome"].astype(str).str.strip().str.lower().isin(["yes", "no"])
            df_bv.loc[mask, "_market_key"] = (
                df_bv.loc[mask, "market"].astype(str)
                + "__"
                + df_bv.loc[mask, "outcome"].astype(str).str.strip().str.lower()
            )
            df_bv.loc[~mask, "_market_key"] = df_bv.loc[~mask, "market"].astype(str)
        else:
            df_bv["_market_key"] = df_bv["market"].astype(str)
        buy_volumes = df_bv.pivot_table(index="_ts", columns="_market_key", values="buy_volume", aggfunc="sum")
        sell_volumes = df_bv.pivot_table(index="_ts", columns="_market_key", values="sell_volume", aggfunc="sum")

    # Source-row VWAP weighted by volume.
    df = df.copy()
    df["src_vwap_value"] = df["vwap"] * df["volume"]
    src_vwap_values = df.pivot_table(index="_ts", columns="market", values="src_vwap_value", aggfunc="sum")

    closes = closes.sort_index().ffill().dropna(axis=1, how="all")
    volumes = volumes.reindex(closes.index).fillna(0.0).reindex(columns=closes.columns)
    src_vwap_values = src_vwap_values.reindex(closes.index).reindex(columns=closes.columns)
    if buy_volumes is not None:
        buy_volumes = buy_volumes.reindex(closes.index).fillna(0.0)
    if sell_volumes is not None:
        sell_volumes = sell_volumes.reindex(closes.index).fillna(0.0)

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
        return closes, vwaps, volumes, buy_volumes, sell_volumes

    closes = closes.resample(resample_rule).last().ffill()
    volumes = volumes.resample(resample_rule).sum().fillna(0.0).reindex(columns=closes.columns)
    if buy_volumes is not None:
        buy_volumes = buy_volumes.resample(resample_rule).sum().fillna(0.0)
    if sell_volumes is not None:
        sell_volumes = sell_volumes.resample(resample_rule).sum().fillna(0.0)

    # Resampled VWAP = sum(vwap * volume) / sum(volume), still no forward-fill.
    weighted = (vwaps * volumes.reindex(vwaps.index).fillna(0.0)).resample(resample_rule).sum()
    vwaps = weighted.divide(volumes.where(volumes > 0.0))
    vwaps = vwaps.reindex(columns=closes.columns)

    closes = closes.dropna(axis=0, how="any")
    volumes = volumes.reindex(closes.index).fillna(0.0)
    if buy_volumes is not None:
        buy_volumes = buy_volumes.reindex(closes.index).fillna(0.0)
    if sell_volumes is not None:
        sell_volumes = sell_volumes.reindex(closes.index).fillna(0.0)
    vwaps = vwaps.reindex(closes.index)
    return closes, vwaps, volumes, buy_volumes, sell_volumes


def load_event_ohlcv(event_slug: str, prefer_outcome: str | None = "yes") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load one Polymarket event and build close/vwap/volume matrices.

    By default this will prefer the `yes` outcome token when markets are
    suffixed with `__yes`/`__no`.
    """
    df = _load_event_rows(event_slug, prefer_outcome=prefer_outcome)
    return _build_matrices(df, resample_rule=None)


def load_event_ohlcv_resampled(
    event_slug: str,
    resample_rule: str | None = None,
    prefer_outcome: str | None = "yes",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Load event OHLCV and optionally downsample bars.

    Args:
        event_slug: Event identifier.
        resample_rule: Pandas rule (e.g. "5min", "10min"). If None, no resample.
    """
    df = _load_event_rows(event_slug, prefer_outcome=prefer_outcome)
    return _build_matrices(df, resample_rule=resample_rule)
