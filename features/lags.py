#features/lags.py
from __future__ import annotations

from typing import Iterable, List, Dict

import pandas as pd


def lag_features(
    df: pd.DataFrame,
    signals: Iterable[str],
    lags_s: List[int],
    group_col: str = "trip_id",
) -> pd.DataFrame:
    """
    For 1 Hz data: lag of N seconds == shift(N) within trip.

    Optimized:
    - Avoid df.copy() on large frames
    - Build lag columns dict and concat once
    """
    if group_col not in df.columns:
        raise ValueError(f"Expected grouping column '{group_col}' in df.")

    sigs = [s for s in signals if s in df.columns]
    if not sigs:
        return df

    lag_cols: Dict[str, pd.Series] = {}

    # Fast path: already per-trip
    if df[group_col].nunique(dropna=False) <= 1:
        for lag in lags_s:
            for s in sigs:
                lag_cols[f"{s}_lag_{lag}s"] = df[s].shift(lag)
        return pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)

    g = df.groupby(group_col, sort=False)
    for lag in lags_s:
        for s in sigs:
            lag_cols[f"{s}_lag_{lag}s"] = g[s].shift(lag)

    return pd.concat([df, pd.DataFrame(lag_cols, index=df.index)], axis=1)