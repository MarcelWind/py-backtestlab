"""OHLC bar permutation utility for Monte Carlo permutation tests.

Main function:
- `get_permutation`: builds one or many synthetic OHLC paths by independently
    shuffling gap returns (open vs previous close) and intrabar moves
    (high/low/close vs open), while preserving index alignment.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd


def get_permutation(
    ohlc: Union[pd.DataFrame, list[pd.DataFrame]],
    start_index: int = 0,
    seed: int | None = None,
) -> Union[pd.DataFrame, list[pd.DataFrame]]:
    """Build synthetic OHLC path(s) via bar-level permutation.

    Independently shuffles gap returns (open vs previous close) and intrabar
    moves (high/low/close vs open) while preserving the original index.

    Parameters
    ----------
    ohlc:
        Single DataFrame or list of DataFrames with ``open/high/low/close``
        columns and a shared index.
    start_index:
        First bar whose successors are eligible for shuffling.  Bars before
        ``start_index`` are copied verbatim.
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    Permuted DataFrame (single input) or list of permuted DataFrames.
    """
    assert start_index >= 0

    np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        n_markets = len(ohlc)
    else:
        n_markets = 1
        time_index = ohlc.index
        ohlc = [ohlc]

    n_bars = len(ohlc[0])

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))

    for mkt_i, reg_bars in enumerate(ohlc):
        log_bars = np.log(reg_bars[["open", "high", "low", "close"]])

        start_bar[mkt_i] = log_bars.iloc[start_index].to_numpy()

        r_o = (log_bars["open"] - log_bars["close"].shift()).to_numpy()
        r_h = (log_bars["high"] - log_bars["open"]).to_numpy()
        r_l = (log_bars["low"] - log_bars["open"]).to_numpy()
        r_c = (log_bars["close"] - log_bars["open"]).to_numpy()

        relative_open[mkt_i] = r_o[perm_index:]
        relative_high[mkt_i] = r_h[perm_index:]
        relative_low[mkt_i] = r_l[perm_index:]
        relative_close[mkt_i] = r_c[perm_index:]

    idx = np.arange(perm_n)

    # Shuffle intrabar relative values (high/low/close).
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]

    # Shuffle last-close-to-open (gaps) separately.
    perm2 = np.random.permutation(idx)
    relative_open = relative_open[:, perm2]

    perm_ohlc: list[pd.DataFrame] = []
    for mkt_i, reg_bars in enumerate(ohlc):
        perm_bars = np.zeros((n_bars, 4))

        log_bars = np.log(reg_bars[["open", "high", "low", "close"]]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        perm_bars[start_index] = start_bar[mkt_i]

        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[mkt_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[mkt_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[mkt_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[mkt_i][k]

        perm_bars = np.exp(perm_bars)
        perm_ohlc.append(
            pd.DataFrame(perm_bars, index=time_index, columns=["open", "high", "low", "close"])
        )

    if n_markets > 1:
        return perm_ohlc
    else:
        return perm_ohlc[0]
