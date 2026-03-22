"""Small data dtype conversion helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_float32_frame(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Downcast float64 columns to float32 to reduce resident memory."""
    if df is None or df.empty:
        return df
    float_cols = list(df.select_dtypes(include=["float64"]).columns)
    if not float_cols:
        return df
    out = df.copy()
    out[float_cols] = out[float_cols].astype(np.float32)
    return out
