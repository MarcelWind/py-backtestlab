"""Fetch macroeconomic data from FRED (Federal Reserve Economic Data)."""

from datetime import datetime, timezone

import pandas as pd
from fredapi import Fred

from ..config import FRED_API_KEY, DATA_DIR

FRED_DIR = DATA_DIR / "fred"

# ---------------------------------------------------------------------------
# Series catalog — human-readable name -> FRED series ID
# ---------------------------------------------------------------------------
FRED_SERIES: dict[str, str] = {
    # Liquidity components
    "fed_balance_sheet": "WALCL",      # Fed total assets (weekly)
    "reverse_repo": "RRPONTSYD",       # Overnight reverse repo (daily)
    "treasury_general_account": "WTREGEN",  # TGA balance (weekly)
    "m2": "M2SL",                      # M2 money supply (monthly)

    # Interest rates
    "fed_funds_rate": "DFF",           # Effective federal funds rate (daily)
    "yield_2y": "DGS2",               # 2-Year Treasury yield (daily)
    "yield_10y": "DGS10",             # 10-Year Treasury yield (daily)
    "yield_curve_10y2y": "T10Y2Y",    # 10Y minus 2Y spread (daily)

    # Credit & dollar
    "hy_spread": "BAMLH0A0HYM2",     # ICE BofA High Yield spread (daily)
    "usd_index": "DTWEXBGS",          # Trade-weighted USD index (daily)
}


def _get_fred() -> Fred:
    if not FRED_API_KEY:
        raise ValueError("FRED_API_KEY not set. Add it to .env")
    return Fred(api_key=FRED_API_KEY)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_fred_series(
    series_id: str,
    start_date: str | None = None,
) -> pd.Series:
    """
    Download a single FRED series.

    Args:
        series_id: FRED series ID (e.g. "WALCL") or a catalog name (e.g. "fed_balance_sheet")
        start_date: Start date as "YYYY-MM-DD" (default: 5 years ago)

    Returns:
        pd.Series with datetime index and float values
    """
    # Allow passing catalog names
    if series_id in FRED_SERIES:
        series_id = FRED_SERIES[series_id]

    if start_date is None:
        start_date = (
            datetime.now(timezone.utc).replace(year=datetime.now(timezone.utc).year - 5)
            .strftime("%Y-%m-%d")
        )

    fred = _get_fred()
    data = fred.get_series(series_id, observation_start=start_date)
    data.name = series_id
    return data.dropna()


def download_fred_catalog(
    series_ids: list[str] | None = None,
    start_date: str | None = None,
) -> pd.DataFrame:
    """
    Download multiple FRED series into an aligned DataFrame.

    Args:
        series_ids: List of FRED series IDs or catalog names.
                    Defaults to all series in FRED_SERIES.
        start_date: Start date as "YYYY-MM-DD" (default: 5 years ago)

    Returns:
        DataFrame with datetime index, one column per series (named by catalog key)
    """
    if series_ids is None:
        series_ids = list(FRED_SERIES.keys())

    columns = {}
    for name in series_ids:
        sid = FRED_SERIES.get(name, name)
        label = name if name in FRED_SERIES else sid
        print(f"  [FRED] {label} ({sid})...", end=" ", flush=True)
        try:
            s = download_fred_series(sid, start_date=start_date)
            columns[label] = s
            print(f"{len(s)} observations")
        except Exception as e:
            print(f"error: {e}")

    if not columns:
        return pd.DataFrame()

    df = pd.DataFrame(columns)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Net liquidity
# ---------------------------------------------------------------------------

def compute_net_liquidity(start_date: str | None = None) -> pd.Series:
    """
    Compute US net liquidity proxy: Fed Balance Sheet - RRP - TGA.

    All three series are forward-filled to daily frequency before subtraction.

    Returns:
        pd.Series (daily, forward-filled) in billions USD
    """
    components = ["fed_balance_sheet", "reverse_repo", "treasury_general_account"]
    df = download_fred_catalog(components, start_date=start_date)

    # Unit normalization: WALCL & WTREGEN are in millions, RRPONTSYD is in billions
    df["reverse_repo"] = df["reverse_repo"] * 1_000  # billions -> millions

    # Forward-fill to daily so weekly/daily series align
    daily = df.asfreq("D").ffill()
    net_liq = daily["fed_balance_sheet"] - daily["reverse_repo"] - daily["treasury_general_account"]
    net_liq.name = "net_liquidity"
    return net_liq.dropna()


# ---------------------------------------------------------------------------
# Global M2
# ---------------------------------------------------------------------------

# Series used for the G4 global M2 indicator
_GLOBAL_M2_SERIES = {
    "us_m2": "M2SL",                    # USD billions, monthly
    "eu_broad_level": "MABMM301EZM189S",  # EUR, monthly (ends ~2023-11)
    "eu_broad_growth": "MABMM301EZM657S", # % MoM growth (current)
    "jp_broad_level": "MABMM301JPM189S",  # JPY, monthly (ends ~2023-11)
    "jp_broad_growth": "MABMM301JPM657S", # % MoM growth (current)
    "cn_m2": "MYAGM2CNM189N",            # CNY, monthly (ends ~2019)
    "fx_eurusd": "DEXUSEU",              # USD per EUR (daily)
    "fx_usdjpy": "DEXJPUS",             # JPY per USD (daily)
    "fx_usdcny": "DEXCHUS",             # CNY per USD (daily)
}


def _chain_level_with_growth(level: pd.Series, growth: pd.Series) -> pd.Series:
    """Extend a level series forward using month-over-month growth rates (%)."""
    # Find where level ends and growth continues
    last_level_date = level.dropna().index[-1]
    future_growth = growth[growth.index > last_level_date].dropna()

    if future_growth.empty:
        return level.dropna()

    chained = level.dropna().copy()
    current = chained.iloc[-1]
    for date, rate in future_growth.items():
        current = current * (1 + rate / 100)
        chained.loc[date] = current

    return chained


def compute_global_m2(start_date: str | None = None) -> pd.DataFrame:
    """
    Compute FX-adjusted G4 global M2: US + Eurozone + Japan + China.

    Formula: (US_M2*1e9 + EU_M3*EURUSD + JP_M3/USDJPY + CN_M2/USDCNY) / 1e12

    EU and JP levels are extended to present by chaining monthly growth rates.
    China M2 stops at 2019 on FRED — it contributes historically but goes flat
    after that (forward-filled). Check the 'cn_m2_stale' column to see the cutoff.

    Returns:
        DataFrame with columns: global_m2, us, eu, jp, cn (all in trillions USD)
    """
    fred = _get_fred()
    print("  [Global M2] Fetching components...")

    def _get(sid: str) -> pd.Series:
        s = fred.get_series(sid, observation_start=start_date or "2010-01-01")
        return s.dropna()

    # M2 / broad money levels in local currency
    us = _get(_GLOBAL_M2_SERIES["us_m2"])
    eu_level = _get(_GLOBAL_M2_SERIES["eu_broad_level"])
    eu_growth = _get(_GLOBAL_M2_SERIES["eu_broad_growth"])
    jp_level = _get(_GLOBAL_M2_SERIES["jp_broad_level"])
    jp_growth = _get(_GLOBAL_M2_SERIES["jp_broad_growth"])
    cn = _get(_GLOBAL_M2_SERIES["cn_m2"])

    # Chain EU/JP levels forward with growth rates
    eu = _chain_level_with_growth(eu_level, eu_growth)
    jp = _chain_level_with_growth(jp_level, jp_growth)

    # FX rates — resample daily to monthly (end of month)
    fx_eur = _get(_GLOBAL_M2_SERIES["fx_eurusd"]).resample("MS").last()
    fx_jpy = _get(_GLOBAL_M2_SERIES["fx_usdjpy"]).resample("MS").last()
    fx_cny = _get(_GLOBAL_M2_SERIES["fx_usdcny"]).resample("MS").last()

    # Align all to monthly, forward-fill
    df = pd.DataFrame({
        "us": us,
        "eu": eu,
        "jp": jp,
        "cn": cn,
        "fx_eur": fx_eur,
        "fx_jpy": fx_jpy,
        "fx_cny": fx_cny,
    }).ffill()

    # Convert to USD trillions
    # US M2 is in billions → *1e9 then /1e12 = /1e3
    result = pd.DataFrame(index=df.index)
    result["us"] = df["us"] / 1e3
    result["eu"] = (df["eu"] * df["fx_eur"]) / 1e12
    result["jp"] = (df["jp"] / df["fx_jpy"]) / 1e12
    result["cn"] = (df["cn"] / df["fx_cny"]) / 1e12
    result["global_m2"] = result[["us", "eu", "jp", "cn"]].sum(axis=1)

    result = result.dropna(subset=["global_m2"])
    result.index.name = "date"

    cn_end = cn.index[-1].strftime("%Y-%m")
    print(f"  [Global M2] US to {us.index[-1].strftime('%Y-%m')}, "
          f"EU to {eu.index[-1].strftime('%Y-%m')}, "
          f"JP to {jp.index[-1].strftime('%Y-%m')}, "
          f"CN to {cn_end} (stale after this)")

    return result


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------

def save_fred(df: pd.DataFrame, name: str = "catalog") -> None:
    """Save FRED DataFrame to data/fred/{name}.parquet."""
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(FRED_DIR / f"{name}.parquet")


def load_fred(name: str = "catalog") -> pd.DataFrame:
    """Load FRED DataFrame from data/fred/{name}.parquet."""
    path = FRED_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
