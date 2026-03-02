# features/rolling.py
from __future__ import annotations

from typing import Iterable, List, Dict

import pandas as pd


def rolling_features(
    df: pd.DataFrame,
    signals: Iterable[str],
    windows_s: List[int],
    aggs: List[str],
    group_col: str = "trip_id",
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Rolling features at 1 Hz.

    Optimized for streaming:
    - If df contains only one trip_id, avoid groupby().rolling() overhead.
    - Build all feature Series into a dict, then concat once.
    """
    base = df  # no copy

    sigs = [s for s in signals if s in base.columns]
    if not sigs:
        return base

    # Fast path: already per-trip
    if group_col in base.columns and base[group_col].nunique(dropna=False) <= 1:
        feat_cols: Dict[str, pd.Series] = {}
        for w in windows_s:
            mp = min_periods if min_periods is not None else w
            for s in sigs:
                roll = base[s].rolling(window=w, min_periods=mp)
                if "mean" in aggs:
                    feat_cols[f"{s}_roll_mean_{w}s"] = roll.mean()
                if "std" in aggs:
                    feat_cols[f"{s}_roll_std_{w}s"] = roll.std(ddof=0)
                if "min" in aggs:
                    feat_cols[f"{s}_roll_min_{w}s"] = roll.min()
                if "max" in aggs:
                    feat_cols[f"{s}_roll_max_{w}s"] = roll.max()
        feats = pd.DataFrame(feat_cols, index=base.index)
        return pd.concat([base, feats], axis=1)

    # General case (kept for compatibility)
    if group_col not in base.columns:
        raise ValueError(f"Expected grouping column '{group_col}' in df.")

    feat_cols = {}
    g = base.groupby(group_col, sort=False)

    for w in windows_s:
        mp = min_periods if min_periods is not None else w
        for s in sigs:
            roll = g[s].rolling(window=w, min_periods=mp)
            if "mean" in aggs:
                feat_cols[f"{s}_roll_mean_{w}s"] = roll.mean().reset_index(level=0, drop=True)
            if "std" in aggs:
                feat_cols[f"{s}_roll_std_{w}s"] = roll.std(ddof=0).reset_index(level=0, drop=True)
            if "min" in aggs:
                feat_cols[f"{s}_roll_min_{w}s"] = roll.min().reset_index(level=0, drop=True)
            if "max" in aggs:
                feat_cols[f"{s}_roll_max_{w}s"] = roll.max().reset_index(level=0, drop=True)

    feats = pd.DataFrame(feat_cols, index=base.index)
    out = pd.concat([base, feats], axis=1)

    # IMPORTANT: don't do out.copy() (it doubles memory)
    return out