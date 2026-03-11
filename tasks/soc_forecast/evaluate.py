from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from evaluation.metrics import (
    error_distribution,
    per_group_regression,
    persistence_baseline,
    regression_metrics,
    rolling_baseline,
    soc_bucket_metrics,
)


def _require_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {where}: {missing}")


def evaluate(
    model: xgb.XGBRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    features: List[str],
    target: str,
    run_id: str,
    thresholds: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Full evaluation for the SOC forecast task.
    Returns a dict written to eval_report.json in the run folder.
    """
    th = thresholds or {}

    _require_columns(train_df, features + [target], "train_df")
    _require_columns(
        val_df,
        features + [target, "trip_id", "soc_pct", "soc_pct_roll_mean_60s"],
        "val_df",
    )

    # ── Predictions ────────────────────────────────────────────────────────────
    pred_train = np.clip(model.predict(train_df[features].astype("float32")), 0, 100)
    pred_val = np.clip(model.predict(val_df[features].astype("float32")), 0, 100)

    val_df = val_df.copy()
    val_df["pred"] = pred_val
    val_df["error"] = pred_val - val_df[target]

    # ── Overall metrics ────────────────────────────────────────────────────────
    print("\n  Overall metrics")
    train_metrics = regression_metrics(train_df[target].values, pred_train, "Train")
    val_metrics = regression_metrics(val_df[target].values, pred_val, "Val")
    gap = round(val_metrics["mae"] - train_metrics["mae"], 3)
    print(f"  Train-val gap: {gap:.3f}")

    # ── Baselines ──────────────────────────────────────────────────────────────
    baseline_persistence = persistence_baseline(
        val_df[target].values, val_df["soc_pct"].values
    )
    baseline_rolling = rolling_baseline(
        val_df[target].values, val_df["soc_pct_roll_mean_60s"].values
    )
    print(f"\n  Baselines (val):")
    print(f"    Persistence (current SOC) MAE: {baseline_persistence:.3f}")
    print(f"    Rolling mean 60s MAE:          {baseline_rolling:.3f}")

    # ── Per-trip breakdown ─────────────────────────────────────────────────────
    per_trip = per_group_regression(val_df, target, "pred", "trip_id")
    print(f"\n  Per-trip MAE (val, {len(per_trip)} trips):")
    print(per_trip.to_string(index=False))

    # ── SOC bucket breakdown ───────────────────────────────────────────────────
    buckets = soc_bucket_metrics(val_df, target, "pred")
    print("\n  MAE by SOC bucket (val):")
    for b in buckets:
        print(f"    {b['soc_bucket']:8s}  rows={b['rows']:5d}  MAE={b['mae']:.3f}")

    # ── Error distribution ─────────────────────────────────────────────────────
    err_dist = error_distribution(val_df["error"])
    print(
        f"\n  Error distribution (val):  "
        f"mean={err_dist['mean']:.3f}  std={err_dist['std']:.3f}  "
        f"p5={err_dist['p5']:.3f}  p95={err_dist['p95']:.3f}"
    )

    # ── Worst predictions ─────────────────────────────────────────────────────
    worst_cols = [
        "timestamp",
        "trip_id",
        "soc_pct",
        target,
        "pred",
        "error",
    ]
    optional_cols = ["battery_current_a", "output_power_kw", "fault_any"]
    worst_cols += [c for c in optional_cols if c in val_df.columns]

    worst_idx = val_df["error"].abs().sort_values(ascending=False).head(10).index
    worst = val_df.loc[worst_idx, worst_cols].to_dict(orient="records")

    # ── Feature importance ─────────────────────────────────────────────────────
    imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n  Top 10 features:")
    for feat, score in imp.head(10).items():
        print(f"    {feat:45s}  {score:.4f}")

    # ── Status label ──────────────────────────────────────────────────────────
    mae_excellent = float(th.get("mae_excellent", 0.5))
    mae_acceptable = float(th.get("mae_acceptable", 1.5))
    mae_investigate = float(th.get("mae_investigate", 2.0))
    gap_ok = float(th.get("train_val_gap_ok", 0.5))

    if val_metrics["mae"] < mae_excellent and gap < gap_ok:
        status = "EXCELLENT"
    elif val_metrics["mae"] < mae_acceptable:
        status = "ACCEPTABLE"
    elif val_metrics["mae"] < mae_investigate:
        status = "INVESTIGATE"
    else:
        status = "FAILING"

    print(f"\n  Model status: {status}")

    best_iter = getattr(model, "best_iteration", None)

    report = {
        "run_id": run_id,
        "task": "soc_forecast",
        "target": target,
        "status": status,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_val_gap": gap,
        "baselines": {
            "persistence_mae": round(baseline_persistence, 3),
            "rolling_mae": round(baseline_rolling, 3),
        },
        "per_trip_val": per_trip.to_dict(orient="records"),
        "soc_buckets_val": buckets,
        "error_distribution": err_dist,
        "worst_predictions": _serialise_worst(worst),
        "feature_importance": imp.head(20).round(4).to_dict(),
        "best_iteration": int(best_iter) if best_iter is not None else None,
    }

    return report


def _serialise_worst(rows: List[Dict]) -> List[Dict]:
    """Convert Timestamps and numpy types for JSON serialisation."""
    out = []
    for r in rows:
        clean = {}
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                clean[k] = v.isoformat()
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = round(float(v), 4)
            else:
                clean[k] = v
        out.append(clean)
    return out