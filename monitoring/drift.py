# monitoring/drift.py
#
# Compares a new day's Gold feature distributions against the training baseline
# saved in a model run folder.
#
# Metrics per feature:
#   PSI   — Population Stability Index  (distribution shift)
#   KS    — Kolmogorov-Smirnov test     (shape change, p-value)
#   Mean shift %                         (gradual sensor drift)
#   Null rate change                     (sensor dropout)
#
# PSI thresholds (industry standard):
#   < 0.1   STABLE
#   0.1–0.2 MONITOR
#   > 0.2   DRIFT

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


# ── PSI ────────────────────────────────────────────────────────────────────────

def _psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two arrays."""
    # Use quantile-based bins from the expected (reference) distribution
    eps = 1e-6

    bins = np.nanquantile(expected, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)  # remove duplicate edges (low-variance features)

    if len(bins) < 2:
        return 0.0

    exp_counts, _ = np.histogram(expected, bins=bins)
    act_counts, _ = np.histogram(actual,   bins=bins)

    exp_pct = exp_counts / (len(expected) + eps)
    act_pct = act_counts / (len(actual)   + eps)

    exp_pct = np.where(exp_pct == 0, eps, exp_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(psi, 4)


# ── Per-feature drift ──────────────────────────────────────────────────────────

def _feature_drift(
    baseline_row: pd.Series,
    new_series: pd.Series,
    psi_alert: float,
    psi_monitor: float,
    ks_pvalue: float,
    mean_shift_pct: float,
) -> Dict:
    new_vals   = new_series.dropna().values
    ref_mean   = float(baseline_row["mean"])
    ref_std    = float(baseline_row["std"])
    ref_null   = float(baseline_row["null_rate"])
    new_null   = float(new_series.isna().mean())

    if len(new_vals) < 10:
        return {
            "feature": baseline_row["feature"],
            "status": "INSUFFICIENT_DATA",
            "psi": None, "ks_pvalue": None,
            "mean_shift_pct": None, "null_rate_change": None,
        }

    # PSI: need reference samples — reconstruct from p10..p90 stored in baseline
    ref_quantiles = np.array([
        baseline_row["p10"], baseline_row["p25"], baseline_row["p50"],
        baseline_row["p75"], baseline_row["p90"],
    ])
    # Generate synthetic reference from stored distribution shape
    ref_approx = np.interp(
        np.linspace(0, 1, 1000),
        [0.10, 0.25, 0.50, 0.75, 0.90],
        ref_quantiles,
    )

    psi_val        = _psi(ref_approx, new_vals)
    ks_stat, ks_p  = stats.ks_2samp(ref_approx, new_vals)
    new_mean       = float(np.mean(new_vals))
    shift_pct      = abs(new_mean - ref_mean) / (abs(ref_mean) + 1e-6) * 100
    null_change    = new_null - ref_null

    # Status
    if psi_val >= psi_alert or ks_p < ks_pvalue or shift_pct > mean_shift_pct:
        status = "DRIFT"
    elif psi_val >= psi_monitor:
        status = "MONITOR"
    else:
        status = "STABLE"

    return {
        "feature":          baseline_row["feature"],
        "status":           status,
        "psi":              round(psi_val, 4),
        "ks_pvalue":        round(float(ks_p), 4),
        "mean_ref":         round(ref_mean, 4),
        "mean_new":         round(new_mean, 4),
        "mean_shift_pct":   round(shift_pct, 2),
        "null_rate_ref":    round(ref_null, 4),
        "null_rate_new":    round(new_null, 4),
        "null_rate_change": round(null_change, 4),
    }


# ── Main drift report ──────────────────────────────────────────────────────────

def compute_drift_report(
    dt: str,
    new_df: pd.DataFrame,
    run_dir: Path,
    thresholds: Optional[Dict] = None,
) -> Dict:
    """
    Compare new_df feature distributions against drift_baseline.parquet
    from the given model run folder.

    Returns a drift report dict (also written to data/drift_reports/).
    """
    baseline_path = run_dir / "drift_baseline.parquet"
    if not baseline_path.exists():
        raise FileNotFoundError(f"No drift baseline found in {run_dir}")

    baseline = pd.read_parquet(baseline_path)

    th = thresholds or {}
    psi_alert      = float(th.get("psi_alert",      0.20))
    psi_monitor    = float(th.get("psi_monitor",    0.10))
    ks_pvalue      = float(th.get("ks_pvalue",      0.05))
    mean_shift_pct = float(th.get("mean_shift_pct", 10.0))

    feature_results = []
    for _, row in baseline.iterrows():
        feat = row["feature"]
        if feat not in new_df.columns:
            continue
        result = _feature_drift(
            row, new_df[feat], psi_alert, psi_monitor, ks_pvalue, mean_shift_pct
        )
        feature_results.append(result)

    n_drift   = sum(1 for r in feature_results if r["status"] == "DRIFT")
    n_monitor = sum(1 for r in feature_results if r["status"] == "MONITOR")
    n_stable  = sum(1 for r in feature_results if r["status"] == "STABLE")

    if n_drift > 0:
        overall = "DRIFT"
    elif n_monitor > 0:
        overall = "MONITOR"
    else:
        overall = "STABLE"

    report = {
        "dt":                dt,
        "model_run_id":      run_dir.name,
        "computed_at":       datetime.now(timezone.utc).isoformat(),
        "features_checked":  len(feature_results),
        "n_drift":           n_drift,
        "n_monitor":         n_monitor,
        "n_stable":          n_stable,
        "overall_status":    overall,
        "thresholds":        {"psi_alert": psi_alert, "psi_monitor": psi_monitor,
                              "ks_pvalue": ks_pvalue, "mean_shift_pct": mean_shift_pct},
        "features":          feature_results,
    }

    _print_drift_summary(report)
    _write_drift_report(dt, report)

    return report


def _print_drift_summary(report: Dict):
    print(f"\n  Drift report [{report['dt']}] — {report['overall_status']}")
    print(f"    Stable: {report['n_stable']}  "
          f"Monitor: {report['n_monitor']}  "
          f"Drift: {report['n_drift']}")
    drifted = [r for r in report["features"] if r["status"] == "DRIFT"]
    if drifted:
        print("    Drifted features:")
        for r in drifted:
            print(f"      {r['feature']:45s}  PSI={r['psi']:.3f}  "
                  f"shift={r['mean_shift_pct']:.1f}%  "
                  f"ks_p={r['ks_pvalue']:.3f}")


def _write_drift_report(dt: str, report: Dict):
    try:
        from configs.settings import DATA_DIR
        drift_dir = Path(DATA_DIR) / "drift_reports"
    except Exception:
        drift_dir = Path("data/drift_reports")

    drift_dir.mkdir(parents=True, exist_ok=True)
    out = drift_dir / f"dt={dt}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"  Drift report → {out}")
