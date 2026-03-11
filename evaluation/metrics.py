# evaluation/metrics.py
#
# Shared metric functions imported by every task's evaluate.py.
# Regression metrics now; classification metrics added here when fault
# detection task is built.

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Regression ─────────────────────────────────────────────────────────────────

def regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
    verbose: bool = True,
) -> Dict[str, float]:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(_safe_mape(y_true, y_pred))

    if verbose:
        prefix = f"  [{label}] " if label else "  "
        print(f"{prefix}MAE={mae:.3f}  RMSE={rmse:.3f}  MAPE={mape:.2f}%")

    return dict(mae=mae, rmse=rmse, mape=mape)


def _safe_mape(y_true, y_pred, eps: float = 1.0) -> float:
    """MAPE clipped to avoid div-by-zero on near-zero SOC values."""
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_pred - y_true) / denom) * 100)


# ── Per-group breakdown ────────────────────────────────────────────────────────

def per_group_regression(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    group_col: str,
) -> pd.DataFrame:
    """MAE / RMSE per group (trip, vehicle, SOC bucket, etc.)."""
    rows = []
    for group_val, grp in df.groupby(group_col):
        m = regression_metrics(grp[y_true_col].values,
                               grp[y_pred_col].values,
                               verbose=False)
        rows.append({group_col: group_val, "rows": len(grp), **m})
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


# ── SOC bucket breakdown ───────────────────────────────────────────────────────

SOC_BUCKETS = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]

def soc_bucket_metrics(
    df: pd.DataFrame,
    y_true_col: str,
    y_pred_col: str,
    soc_col: str = "soc_pct",
) -> List[Dict]:
    rows = []
    for lo, hi in SOC_BUCKETS:
        mask = (df[soc_col] >= lo) & (df[soc_col] < hi)
        sub  = df[mask]
        if len(sub) == 0:
            continue
        m = regression_metrics(sub[y_true_col].values,
                               sub[y_pred_col].values,
                               verbose=False)
        rows.append({"soc_bucket": f"{lo}–{hi}%", "rows": len(sub), **m})
    return rows


# ── Error distribution ─────────────────────────────────────────────────────────

def error_distribution(errors: pd.Series) -> Dict[str, float]:
    return {
        "mean":  float(errors.mean()),
        "std":   float(errors.std()),
        "p5":    float(errors.quantile(0.05)),
        "p25":   float(errors.quantile(0.25)),
        "p50":   float(errors.quantile(0.50)),
        "p75":   float(errors.quantile(0.75)),
        "p95":   float(errors.quantile(0.95)),
        "max_abs": float(errors.abs().max()),
    }


# ── Baselines ──────────────────────────────────────────────────────────────────

def persistence_baseline(y_true: np.ndarray, soc_now: np.ndarray) -> float:
    """Naive baseline: predict current SOC = future SOC."""
    return float(mean_absolute_error(y_true, soc_now))


def rolling_baseline(y_true: np.ndarray, soc_roll: np.ndarray) -> float:
    """Baseline: predict rolling-mean SOC = future SOC."""
    return float(mean_absolute_error(y_true, soc_roll))
