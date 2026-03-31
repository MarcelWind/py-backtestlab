"""Batch bar-permutation for vectorized Monte Carlo permutation tests.

Pre-generates N permutation indices and materialises the permuted data as
3D numpy arrays of shape ``(n_perms, n_bars, n_assets)`` so that downstream
vectorized indicators can process all permutations in a single pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(slots=True)
class BatchPermutedEvent:
    """Pre-permuted event data for one event across all permutations.

    Each field is a 3D numpy array of shape ``(n_perms, n_bars, n_cols)``
    unless noted otherwise.

    Attributes
    ----------
    n_perms : int
    n_bars : int
    assets : list[str]
        Submarket column names (order matches axis-2 of close/high/low/open_/vwap/volume).
    index : pd.Index
        Original DatetimeIndex.
    perm_indices : ndarray (n_perms, n_bars)
        Integer permutation indices used to derive the arrays.
    close : ndarray
    high : ndarray or None
    low : ndarray or None
    open_ : ndarray or None
    vwap : ndarray or None
    volume : ndarray or None
    buy_volume_yes : ndarray or None
        Shape ``(n_perms, n_bars, n_assets)`` — matched to *assets* order.
    buy_volume_no : ndarray or None
    sell_volume_yes : ndarray or None
    sell_volume_no : ndarray or None
    returns : ndarray
        ``(n_perms, n_bars, n_assets)`` — per-bar pct-change returns of
        permuted close prices.
    """

    n_perms: int
    n_bars: int
    assets: list[str]
    index: pd.Index
    perm_indices: np.ndarray
    close: np.ndarray
    high: np.ndarray | None
    low: np.ndarray | None
    open_: np.ndarray | None
    vwap: np.ndarray | None
    volume: np.ndarray | None
    buy_volume_yes: np.ndarray | None
    buy_volume_no: np.ndarray | None
    sell_volume_yes: np.ndarray | None
    sell_volume_no: np.ndarray | None
    returns: np.ndarray


def _df_to_array(df: pd.DataFrame | None, assets: list[str]) -> np.ndarray | None:
    """Convert DataFrame to 2D numpy array aligned to *assets* column order."""
    if df is None or df.empty:
        return None
    aligned = df.reindex(columns=assets, fill_value=np.nan)
    return aligned.to_numpy(dtype=np.float64)


def _permute_3d(
    arr_2d: np.ndarray,
    perm_indices: np.ndarray,
) -> np.ndarray:
    """Apply N permutations to a 2D array, producing a 3D result.

    Parameters
    ----------
    arr_2d : (n_bars, n_cols)
    perm_indices : (n_perms, n_bars)

    Returns
    -------
    (n_perms, n_bars, n_cols)
    """
    return arr_2d[perm_indices]  # advanced indexing broadcasts correctly


def generate_permutation_indices(
    n_bars: int,
    n_perms: int,
    seed_offset: int = 1,
) -> np.ndarray:
    """Generate *n_perms* independent row permutations.

    Returns shape ``(n_perms, n_bars)`` of integer indices.
    Permutation *i* uses seed ``seed_offset + i`` for reproducibility.
    """
    indices = np.empty((n_perms, n_bars), dtype=np.intp)
    for i in range(n_perms):
        rng = np.random.default_rng(seed_offset + i)
        indices[i] = rng.permutation(n_bars)
    return indices


def batch_permute_event(
    prices: pd.DataFrame,
    n_perms: int,
    *,
    high: pd.DataFrame | None = None,
    low: pd.DataFrame | None = None,
    open_: pd.DataFrame | None = None,
    vwap: pd.DataFrame | None = None,
    volume: pd.DataFrame | None = None,
    buy_volume: pd.DataFrame | None = None,
    sell_volume: pd.DataFrame | None = None,
    seed_offset: int = 1,
) -> BatchPermutedEvent:
    """Batch-permute all fields of one event for N permutations.

    Parameters
    ----------
    prices : DataFrame (n_bars × n_assets)
        Close prices — required.
    n_perms : int
        Number of permutations to generate.
    high, low, open_, vwap, volume : DataFrame, optional
        Additional field matrices aligned to *prices*.
    buy_volume, sell_volume : DataFrame, optional
        Yes/no volume matrices.  Columns are matched to the *prices* asset
        order by extracting ``{asset}__yes`` and ``{asset}__no`` columns.
    seed_offset : int
        Starting seed for reproducibility (permutation *i* uses seed
        ``seed_offset + i``).

    Returns
    -------
    BatchPermutedEvent
    """
    assets = prices.columns.tolist()
    n_bars = len(prices)
    idx = prices.index

    perm_indices = generate_permutation_indices(n_bars, n_perms, seed_offset)

    close_2d = prices.to_numpy(dtype=np.float64)
    close_3d = _permute_3d(close_2d, perm_indices)

    high_3d = _permute_3d(_df_to_array(high, assets), perm_indices) if high is not None else None
    low_3d = _permute_3d(_df_to_array(low, assets), perm_indices) if low is not None else None
    open_3d = _permute_3d(_df_to_array(open_, assets), perm_indices) if open_ is not None else None
    vwap_3d = _permute_3d(_df_to_array(vwap, assets), perm_indices) if vwap is not None else None
    vol_3d = _permute_3d(_df_to_array(volume, assets), perm_indices) if volume is not None else None

    # Buy/sell volume: extract yes/no for each asset
    def _resolve_yesno(vol_df: pd.DataFrame | None, suffix: str) -> np.ndarray | None:
        if vol_df is None or vol_df.empty:
            return None
        cols: list[str] = []
        for a in assets:
            c = f"{a}__{suffix}"
            if c in vol_df.columns:
                cols.append(c)
            else:
                cols.append(None)  # type: ignore[arg-type]
        if all(c is None for c in cols):
            return None
        arr = np.full((n_bars, len(assets)), np.nan, dtype=np.float64)
        for j, c in enumerate(cols):
            if c is not None and c in vol_df.columns:
                arr[:, j] = vol_df[c].to_numpy(dtype=np.float64)
        return _permute_3d(arr, perm_indices)

    bv_yes = _resolve_yesno(buy_volume, "yes")
    bv_no = _resolve_yesno(buy_volume, "no")
    sv_yes = _resolve_yesno(sell_volume, "yes")
    sv_no = _resolve_yesno(sell_volume, "no")

    # Compute returns from permuted close prices: pct_change along bars axis
    # First bar of returns is NaN; matches DataFrame.pct_change() behaviour.
    returns_3d = np.empty_like(close_3d)
    returns_3d[:, 0, :] = np.nan
    returns_3d[:, 1:, :] = close_3d[:, 1:, :] / np.where(
        close_3d[:, :-1, :] != 0, close_3d[:, :-1, :], np.nan
    ) - 1.0

    return BatchPermutedEvent(
        n_perms=n_perms,
        n_bars=n_bars,
        assets=assets,
        index=idx,
        perm_indices=perm_indices,
        close=close_3d,
        high=high_3d,
        low=low_3d,
        open_=open_3d,
        vwap=vwap_3d,
        volume=vol_3d,
        buy_volume_yes=bv_yes,
        buy_volume_no=bv_no,
        sell_volume_yes=sv_yes,
        sell_volume_no=sv_no,
        returns=returns_3d,
    )
