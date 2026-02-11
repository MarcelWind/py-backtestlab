"""VWAP-adjusted prediction market forecast.

Corrects market-implied expected values using volume/price microstructure
features. Trains a Ridge regression to predict the market's error
(delta between realised outcome and market-implied mean).

Input: OHLCV DataFrames from prediction-market buckets, loaded via
``fetch_data.load_zip``. Each row is one candle for one bucket with columns:
    event_slug, market, timestamp, datetime,
    open, high, low, close, volume, trade_count, vwap

``prepare_snapshots`` resamples per-bucket candles to a common frequency and
adds ``bucket_low`` / ``bucket_high`` columns parsed from the market name,
producing aligned cross-bucket snapshots ready for feature extraction.
"""

import re

import numpy as np
import pandas as pd

from stratlab.prediction import (
    market_mean,
    volume_entropy,
    volume_skew,
)


# ---------------------------------------------------------------------------
# Bucket-name parsing
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Fahrenheit range: "32-33-f"
    (re.compile(r"^(\d+)-(\d+)-f$"), "f_range"),
    # Fahrenheit edge-low: "31-f-or-below"
    (re.compile(r"^(\d+)-f-or-below$"), "f_low"),
    # Fahrenheit edge-high: "42-f-or-higher"
    (re.compile(r"^(\d+)-f-or-higher$"), "f_high"),
    # Celsius edge-low: "4-c-or-below" / "-7-c-or-below"
    (re.compile(r"^(-?\d+)-c-or-below$"), "c_low"),
    # Celsius edge-high: "10-c-or-higher" / "-1-c-or-higher"
    (re.compile(r"^(-?\d+)-c-or-higher$"), "c_high"),
    # Celsius single: "5-c" / "-6-c"
    (re.compile(r"^(-?\d+)-c$"), "c_single"),
]


def parse_bucket_name(name: str) -> dict:
    """Parse a market/bucket slug into boundary values.

    Returns dict with keys ``low``, ``high`` (floats or None for edge
    buckets), ``unit`` ('c' or 'f'), and ``edge`` ('lower'/'upper'/None).
    """
    for pat, kind in _PATTERNS:
        m = pat.match(name)
        if not m:
            continue
        if kind == "f_range":
            return {"low": int(m[1]), "high": int(m[2]), "unit": "f", "edge": None}
        if kind == "f_low":
            return {"low": None, "high": int(m[1]), "unit": "f", "edge": "lower"}
        if kind == "f_high":
            return {"low": int(m[1]), "high": None, "unit": "f", "edge": "upper"}
        if kind == "c_low":
            return {"low": None, "high": int(m[1]), "unit": "c", "edge": "lower"}
        if kind == "c_high":
            return {"low": int(m[1]), "high": None, "unit": "c", "edge": "upper"}
        if kind == "c_single":
            v = int(m[1])
            return {"low": v, "high": v, "unit": "c", "edge": None}
    raise ValueError(f"Cannot parse bucket name: {name!r}")


def compute_bucket_bounds(market_names: list[str]) -> dict[str, tuple[float, float]]:
    """Compute ``(bucket_low, bucket_high)`` for every bucket in an event.

    Edge buckets (``or-below`` / ``or-higher``) are extended by one step
    inferred from the neighbouring interior buckets.  Single-value Celsius
    buckets are widened to ±0.5 so that midpoints equal the stated value.
    """
    parsed = {n: parse_bucket_name(n) for n in market_names}

    # Determine step from interior buckets
    interior = [p for p in parsed.values() if p["edge"] is None]
    interior.sort(key=lambda p: p["low"])
    if len(interior) >= 2:
        step = interior[1]["low"] - interior[0]["low"]
    elif interior and interior[0]["low"] == interior[0]["high"]:
        step = 1  # single-degree Celsius
    elif interior:
        step = interior[0]["high"] - interior[0]["low"] + 1
    else:
        step = 1

    bounds: dict[str, tuple[float, float]] = {}
    for name, p in parsed.items():
        if p["edge"] == "lower":
            bounds[name] = (float(p["high"] - step), float(p["high"]))
        elif p["edge"] == "upper":
            bounds[name] = (float(p["low"]), float(p["low"] + step))
        elif p["low"] == p["high"]:
            # single-value Celsius — widen ±0.5 so midpoint == value
            bounds[name] = (p["low"] - 0.5, p["high"] + 0.5)
        else:
            bounds[name] = (float(p["low"]), float(p["high"]))
    return bounds


# ---------------------------------------------------------------------------
# Snapshot preparation (OHLCV → cross-bucket snapshots)
# ---------------------------------------------------------------------------

def prepare_snapshots(
    event_df: pd.DataFrame,
    freq: str = "1h",
    settlement_time: pd.Timestamp | None = None,
    realized: float | None = None,
) -> pd.DataFrame:
    """Resample per-bucket OHLCV candles into aligned cross-bucket snapshots.

    Args:
        event_df: Rows for a single event (all buckets).  Must contain
            ``market``, ``timestamp``, ``close``, ``vwap``, ``volume``.
        freq: Resample frequency (e.g. ``'1h'``, ``'30min'``).
        settlement_time: When the event resolves.  Used to compute
            ``hours_to_settlement`` for each snapshot.  If *None* the column
            is filled with NaN.
        realized: Actual outcome value.  Broadcast to every snapshot row
            so that ``build_dataset`` can compute the target.

    Returns:
        DataFrame with columns: ``event_slug, market, timestamp,
        bucket_low, bucket_high, close, vwap, volume``
        (plus ``hours_to_settlement`` and ``realized`` when provided).
    """
    event_slug = event_df["event_slug"].iloc[0]
    markets = event_df["market"].unique().tolist()
    bounds = compute_bucket_bounds(markets)

    resampled: list[pd.DataFrame] = []
    for mkt, grp in event_df.groupby("market"):
        mkt_str = str(mkt)
        ts = pd.to_datetime(grp["timestamp"], unit="s", utc=True)
        tmp = grp.set_index(ts).sort_index()
        r = tmp.resample(freq).agg({"close": "last", "vwap": "last", "volume": "sum"})
        r = r.dropna(subset=["close"])
        r["market"] = mkt_str
        lo, hi = bounds[mkt_str]
        r["bucket_low"] = lo
        r["bucket_high"] = hi
        resampled.append(r)

    combined = pd.concat(resampled)
    combined.index.name = "timestamp"
    combined = combined.reset_index()

    # Keep only timestamps present in ALL buckets
    ts_counts = combined.groupby("timestamp")["market"].nunique()
    valid_ts = ts_counts[ts_counts == len(markets)].index
    combined = combined[combined["timestamp"].isin(valid_ts)].copy()

    combined["event_slug"] = event_slug

    if settlement_time is not None:
        td = (settlement_time - combined["timestamp"]).values
        secs = np.array(td, dtype="timedelta64[s]").astype(np.float64)
        combined["hours_to_settlement"] = secs / 3600.0
    else:
        combined["hours_to_settlement"] = np.nan

    if realized is not None:
        combined["realized"] = realized

    combined = combined.sort_values(["timestamp", "bucket_low"]).reset_index(drop=True)
    return combined


# ---------------------------------------------------------------------------
# Event metadata auto-inference
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}


def parse_event_slug(slug: str) -> dict:
    """Extract city and date from an event slug.

    >>> parse_event_slug("highest-temperature-in-chicago-on-february-9-2026")
    {'city': 'chicago', 'date': Timestamp('2026-02-09 00:00:00+0000', tz='UTC')}
    """
    m = re.match(r"highest-temperature-in-(.+)-on-(.+)", slug)
    if not m:
        raise ValueError(f"Cannot parse event slug: {slug!r}")
    city = m.group(1)
    tokens = m.group(2).split("-")
    month = _MONTH_MAP[tokens[0]]
    day = int(tokens[1])
    year = int(tokens[2]) if len(tokens) >= 3 else 2026
    return {
        "city": city,
        "date": pd.Timestamp(year=year, month=month, day=day, tz="UTC"),
    }


def infer_settlement_time(slug: str) -> pd.Timestamp:
    """Return midnight UTC of the day *after* the event date."""
    info = parse_event_slug(slug)
    return info["date"] + pd.Timedelta(days=1)


def detect_settlement(
    event_df: pd.DataFrame,
    threshold: float = 0.99,
) -> dict | None:
    """Detect whether an event has settled and compute the realized value.

    Examines the last candle for each bucket.  If any bucket's final
    ``close >= threshold``, the event is considered settled and the
    realized temperature is the midpoint of that bucket's bounds.

    Returns dict with ``winner_market``, ``realized``, ``unit``
    if settled, or *None* otherwise.
    """
    markets = event_df["market"].unique().tolist()
    bounds = compute_bucket_bounds(markets)

    last_candles = event_df.loc[event_df.groupby("market")["timestamp"].idxmax()]
    winner_row = last_candles.loc[last_candles["close"] >= threshold]
    if winner_row.empty:
        return None

    winner_market = str(winner_row.iloc[0]["market"])
    lo, hi = bounds[winner_market]
    parsed = parse_bucket_name(winner_market)
    return {
        "winner_market": winner_market,
        "realized": (lo + hi) / 2,
        "unit": parsed["unit"],
    }


def build_event_registry(
    data: pd.DataFrame,
    threshold: float = 0.99,
) -> pd.DataFrame:
    """Build a metadata table for every event in *data*.

    Returns one row per ``event_slug`` with columns:
        event_slug, city, event_date, settlement_time, n_buckets,
        is_settled, winner_market, realized, unit
    """
    records = []
    for slug, grp in data.groupby("event_slug"):
        slug_str = str(slug)
        info = parse_event_slug(slug_str)
        settlement = infer_settlement_time(slug_str)
        n_buckets = grp["market"].nunique()

        settle_info = detect_settlement(grp, threshold=threshold)
        records.append({
            "event_slug": slug_str,
            "city": info["city"],
            "event_date": info["date"],
            "settlement_time": settlement,
            "n_buckets": n_buckets,
            "is_settled": settle_info is not None,
            "winner_market": settle_info["winner_market"] if settle_info else None,
            "realized": settle_info["realized"] if settle_info else None,
            "unit": settle_info["unit"] if settle_info else parse_bucket_name(
                grp["market"].iloc[0]
            )["unit"],
        })

    return pd.DataFrame(records).sort_values("event_slug").reset_index(drop=True)


FEATURE_COLS = ["mean_shift", "volume_skew", "volume_entropy", "hours_to_settlement"]

def extract_features(snapshot: pd.DataFrame) -> dict[str, float]:
    """
    Extract microstructure features from a single market snapshot.

    Args:
        snapshot: DataFrame with one row per bucket, must have columns
                  bucket_low, bucket_high, close, vwap, volume

    Returns:
        Dict with mu_m, mean_shift, volume_skew, volume_entropy
    """
    mids = ((snapshot["bucket_low"] + snapshot["bucket_high"]) / 2).values
    prices = snapshot["close"].values
    vwaps = snapshot["vwap"].values
    volumes = snapshot["volume"].values

    mu_m = market_mean(mids, prices)
    mu_vwap = market_mean(mids, vwaps)

    return {
        "mu_m": mu_m,
        "mean_shift": mu_vwap - mu_m,
        "volume_skew": volume_skew(mids, prices, volumes),
        "volume_entropy": volume_entropy(prices, volumes),
    }


def build_dataset(
    data: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build feature matrix from grouped market snapshots.

    Groups data by (event_slug, timestamp) — or custom group_cols —
    and extracts features per snapshot. Scalar columns (hours_to_settlement,
    realized) are carried forward from the first row of each group.

    Args:
        data: Full dataset with one row per bucket per snapshot
        group_cols: Columns to group by (default: ["event_slug", "timestamp"])

    Returns:
        DataFrame with one row per snapshot, feature columns + mu_m + target
    """
    if group_cols is None:
        group_cols = ["event_slug", "timestamp"]

    records = []
    for _key, snapshot in data.groupby(group_cols):
        feats = extract_features(snapshot)

        if "hours_to_settlement" in snapshot.columns:
            feats["hours_to_settlement"] = snapshot["hours_to_settlement"].iloc[0]

        if "realized" in snapshot.columns:
            realized = snapshot["realized"].iloc[0]
            feats["realized"] = realized
            feats["target"] = realized - feats["mu_m"]

        records.append(feats)

    return pd.DataFrame(records)


class AdjustedForecast:
    """Ridge regression that learns to predict market error from features."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self._x_mean: np.ndarray | None = None
        self._y_mean: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AdjustedForecast":
        """
        Fit Ridge regression: minimise ||y - Xw||^2 + alpha * ||w||^2.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (market error: realized - mu_m)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._x_mean = X.mean(axis=0)
        self._y_mean = float(y.mean())

        Xc = X - self._x_mean
        yc = y - self._y_mean

        A = Xc.T @ Xc
        A[np.diag_indices_from(A)] += self.alpha
        self.coef_ = np.linalg.solve(A, Xc.T @ yc)
        self.intercept_ = float(self._y_mean - self._x_mean @ self.coef_)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict market error delta_T_hat."""
        X = np.asarray(X, dtype=float)
        return X @ self.coef_ + self.intercept_


def forecast(model: AdjustedForecast, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Produce adjusted forecasts.

    Adds columns: delta_hat (predicted error), adjusted (mu_m + delta_hat).

    Args:
        model: Fitted AdjustedForecast
        dataset: Output of build_dataset with FEATURE_COLS present
    """
    X = dataset[FEATURE_COLS].values
    delta_hat = model.predict(X)

    out = dataset.copy()
    out["delta_hat"] = delta_hat
    out["adjusted"] = out["mu_m"] + delta_hat
    return out


def evaluate(dataset: pd.DataFrame) -> dict[str, float]:
    """
    Compare baseline (market-implied mean) vs adjusted forecast.

    Expects columns: realized, mu_m, adjusted.

    Returns:
        Dict with baseline_mae, adjusted_mae, improvement
    """
    realized = dataset["realized"].values
    mu_m = dataset["mu_m"].values
    adjusted = dataset["adjusted"].values

    baseline_mae = float(np.mean(np.abs(realized - mu_m)))
    adjusted_mae = float(np.mean(np.abs(realized - adjusted)))

    return {
        "baseline_mae": baseline_mae,
        "adjusted_mae": adjusted_mae,
        "improvement": baseline_mae - adjusted_mae,
    }


def shuffle_test(
    data: pd.DataFrame,
    alpha: float = 1.0,
    n_trials: int = 100,
    group_cols: list[str] | None = None,
    seed: int | None = None,
) -> pd.Series:
    """
    Shuffle VWAP/volume within each snapshot and refit.

    If performance doesn't collapse under shuffled features,
    the signal is spurious (leaking or overfitting).

    Args:
        data: Raw data (same format as build_dataset input)
        alpha: Ridge alpha
        n_trials: Number of shuffle iterations
        group_cols: Columns to group by
        seed: Random seed

    Returns:
        Series of improvement values under shuffled data
    """
    rng = np.random.default_rng(seed)
    improvements = []

    for _ in range(n_trials):
        shuffled = data.copy()

        groups = group_cols or ["event_slug", "timestamp"]
        for _key, idx in shuffled.groupby(groups).groups.items():
            perm = rng.permutation(len(idx))
            shuffled.loc[idx, "vwap"] = shuffled.loc[idx, "vwap"].values[perm]
            shuffled.loc[idx, "volume"] = shuffled.loc[idx, "volume"].values[perm]

        ds = build_dataset(shuffled, group_cols)
        n = len(ds)
        split = n // 2

        model = AdjustedForecast(alpha=alpha)
        model.fit(ds.iloc[:split][FEATURE_COLS].values, ds.iloc[:split]["target"].values)
        result = forecast(model, ds.iloc[split:])
        metrics = evaluate(result)
        improvements.append(metrics["improvement"])

    return pd.Series(improvements, name="shuffled_improvement")
