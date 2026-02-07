"""Fetch macroeconomic data from DBnomics (db.nomics.world)."""

import pandas as pd
import requests

from ..config import DATA_DIR

DBNOMICS_DIR = DATA_DIR / "dbnomics"
DBNOMICS_API = "https://api.db.nomics.world/v22/series"

# ---------------------------------------------------------------------------
# Series catalog — human-readable name -> (provider, dataset, series_code)
# ---------------------------------------------------------------------------
DBNOMICS_SERIES: dict[str, tuple[str, str, str]] = {
    # ISM Manufacturing
    "ism_pmi": ("ISM", "pmi", "pm"),
    "ism_production": ("ISM", "production", "in"),
    "ism_new_orders": ("ISM", "neword", "in"),
    "ism_employment": ("ISM", "employment", "in"),
    "ism_prices": ("ISM", "prices", "in"),
    "ism_inventories": ("ISM", "inventories", "in"),
    "ism_new_export_orders": ("ISM", "newexpord", "in"),
    "ism_imports": ("ISM", "imports", "in"),
    "ism_backlog": ("ISM", "bacord", "in"),
    "ism_supplier_deliveries": ("ISM", "supdel", "in"),

    # ISM Non-Manufacturing (Services)
    "ism_services_pmi": ("ISM", "nm-pmi", "pm"),
    "ism_services_business": ("ISM", "nm-busact", "in"),
    "ism_services_new_orders": ("ISM", "nm-neword", "in"),
    "ism_services_employment": ("ISM", "nm-employment", "in"),
    "ism_services_prices": ("ISM", "nm-prices", "in"),
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_dbnomics_series(
    provider: str,
    dataset: str,
    series_code: str = "in",
) -> pd.Series:
    """
    Download a single series from DBnomics.

    Args:
        provider: Provider code (e.g. "ISM")
        dataset: Dataset code (e.g. "production")
        series_code: Series code within the dataset (e.g. "in" for Index)

    Returns:
        pd.Series with datetime index
    """
    url = f"{DBNOMICS_API}/{provider}/{dataset}"
    params = {"facets": 1, "format": "json", "limit": 1000, "observations": 1}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    target = None
    for s in data.get("series", {}).get("docs", []):
        if s["series_code"] == series_code:
            target = s
            break

    if target is None:
        raise ValueError(f"Series '{series_code}' not found in {provider}/{dataset}")

    dates = pd.to_datetime(target["period_start_day"])
    values = pd.to_numeric(target["value"], errors="coerce")
    result = pd.Series(list(values), index=dates, name=f"{provider}/{dataset}/{series_code}")
    return result.dropna().sort_index()


def download_dbnomics_catalog(
    series_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Download multiple DBnomics series into an aligned DataFrame.

    Args:
        series_names: List of catalog names (e.g. ["ism_pmi", "ism_production"]).
                      Defaults to all series in DBNOMICS_SERIES.

    Returns:
        DataFrame with datetime index, one column per series
    """
    if series_names is None:
        series_names = list(DBNOMICS_SERIES.keys())

    columns = {}
    for name in series_names:
        if name not in DBNOMICS_SERIES:
            print(f"  [DBnomics] {name} — not in catalog, skipping")
            continue
        provider, dataset, code = DBNOMICS_SERIES[name]
        print(f"  [DBnomics] {name} ({provider}/{dataset}/{code})...", end=" ", flush=True)
        try:
            s = download_dbnomics_series(provider, dataset, code)
            columns[name] = s
            print(f"{len(s)} observations")
        except Exception as e:
            print(f"error: {e}")

    if not columns:
        return pd.DataFrame()

    df = pd.DataFrame(columns)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# Parquet persistence
# ---------------------------------------------------------------------------

def save_dbnomics(df: pd.DataFrame, name: str = "catalog") -> None:
    """Save DBnomics DataFrame to data/dbnomics/{name}.parquet."""
    DBNOMICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DBNOMICS_DIR / f"{name}.parquet")


def load_dbnomics(name: str = "catalog") -> pd.DataFrame:
    """Load DBnomics DataFrame from data/dbnomics/{name}.parquet."""
    path = DBNOMICS_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
