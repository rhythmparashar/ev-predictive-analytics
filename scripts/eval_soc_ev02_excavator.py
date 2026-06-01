"""
SOC Model v3 — Cross-Machine Evaluation: EV01 Loader → EV02 Excavator
======================================================================
Tests the EV01 Loader-trained SOC model (v3) on EV02 Excavator data it has never seen.
Confirms that a Loader model does not transfer to an Excavator (different duty cycle).

Usage:
    python scripts/eval_soc_ev02_excavator.py

Prerequisites:
    Gold window features must exist for EV02. If not, run first:
        python -m scripts.backfill_gold --start 2026-01-15 --end 2026-05-14

Outputs:
    reports/soc_eval_ev02_excavator.md
    outputs/soc_eval_ev02_excavator.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.settings import DATA_DIR

MODEL_DIR   = PROJECT_ROOT / "models" / "soc_forecast" / "v3__2026-05-20__7b14ac62"
GOLD_DIR    = DATA_DIR / "gold" / "window_features"
REPORT_DIR  = PROJECT_ROOT / "reports"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"

VEHICLE_ID       = "EV02"
MACHINE_TYPE     = "Excavator"
TRAIN_VEHICLE    = "EV01"
TRAIN_MACHINE    = "Loader"
TARGET           = "y_soc_t_plus_300s"
LABEL_COL        = "label_available"
DROP_QUALITY_MASK = 52   # bits 4|16|32 — gap-inserted, range breach, time anomaly (same as EV01 training)

# BMS reset days — SOC is completely unreliable, exclude from eval
BMS_RESET_DATES = {"2026-05-04", "2026-05-06"}

# v3 reference metrics (trained + validated on EV01)
V3_VAL_MAE         = 0.511
V3_VAL_RMSE        = 0.7257
V3_PERSISTENCE_MAE = 1.383


class Accumulator:
    def __init__(self):
        self.n = 0
        self.abs_err = 0.0
        self.sq_err  = 0.0
        self.mape_num = 0.0
        self.mape_den = 0

    def update(self, y: np.ndarray, yhat: np.ndarray):
        err = y - yhat
        self.n        += len(y)
        self.abs_err  += float(np.sum(np.abs(err)))
        self.sq_err   += float(np.sum(err ** 2))
        mask = np.abs(y) > 0.5
        self.mape_num += float(np.sum(np.abs(err[mask] / y[mask]))) * 100
        self.mape_den += int(mask.sum())

    def metrics(self) -> dict:
        if self.n == 0:
            return {"mae": None, "rmse": None, "mape": None}
        return {
            "mae":  round(self.abs_err / self.n, 4),
            "rmse": round(float(np.sqrt(self.sq_err / self.n)), 4),
            "mape": round(self.mape_num / self.mape_den, 4) if self.mape_den > 0 else None,
        }


def load_day(dt: str, features: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    path = GOLD_DIR / f"dt={dt}" / f"vehicle_id={VEHICLE_ID}"
    if not path.exists():
        return None

    parts = sorted(path.glob("*.parquet"))
    if not parts:
        return None

    chunks = []
    for f in parts:
        df = pd.read_parquet(f)
        if LABEL_COL in df.columns:
            df = df[df[LABEL_COL] == 1]
        df = df[df["trip_id"].notna()]
        # Drop gap-inserted / bad-quality rows — same mask used in EV01 training
        if "quality_flag" in df.columns:
            df = df[(df["quality_flag"].astype(int) & DROP_QUALITY_MASK) == 0]
        # Drop rows where the primary input feature is null (can't fill with 0 safely)
        if "soc_pct" in df.columns:
            df = df[df["soc_pct"].notna()]
        if df.empty or TARGET not in df.columns:
            continue
        chunks.append(df)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    X = df[features].fillna(0).astype("float32")
    y = df[TARGET].astype(float).to_numpy()
    return y, X.to_numpy()


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading v3 model …")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_DIR / "model.json"))
    feature_set = json.loads((MODEL_DIR / "feature_set.json").read_text())
    features = feature_set["features"]
    print(f"  Features : {len(features)}")
    print(f"  Ref MAE  : {V3_VAL_MAE}%  (EV01 val, never seen EV02)\n")

    all_dates = sorted(
        d.name.replace("dt=", "")
        for d in GOLD_DIR.iterdir()
        if d.is_dir()
        and d.name.startswith("dt=")
        and (d / f"vehicle_id={VEHICLE_ID}").exists()
        and d.name.replace("dt=", "") not in BMS_RESET_DATES
    )

    if not all_dates:
        print("ERROR: No EV02 gold window features found.")
        print("Run this first:")
        print("  python -m scripts.backfill_gold --start 2026-01-15 --end 2026-05-14")
        sys.exit(1)

    print(f"EV02 dates found: {len(all_dates)}  ({all_dates[0]} → {all_dates[-1]})\n")

    total_acc = Accumulator()
    per_day   = []

    for dt in all_dates:
        result = load_day(dt, features)
        if result is None:
            print(f"  {dt}  skip (no usable rows)")
            continue

        y, X = result
        yhat = model.predict(X, validate_features=False)
        total_acc.update(y, yhat)

        day_acc = Accumulator()
        day_acc.update(y, yhat)
        m = day_acc.metrics()
        m["date"] = dt
        m["rows"] = len(y)
        per_day.append(m)

        flag = "  ← degraded" if m["mae"] and m["mae"] > V3_VAL_MAE * 1.5 else ""
        print(f"  {dt}  rows={len(y):>6,}  MAE={m['mae']:.4f}%{flag}")

    overall = total_acc.metrics()
    overall["n_rows"] = total_acc.n

    print("\n" + "=" * 60)
    print("  SOC MODEL v3 — CROSS-VEHICLE EVAL (EV02)")
    print("=" * 60)
    print(f"\n  EV01 val MAE  (training reference) : {V3_VAL_MAE:.4f}%")
    print(f"  EV02 overall MAE (never-seen data)  : {overall['mae']:.4f}%")
    delta = overall["mae"] - V3_VAL_MAE
    print(f"  Delta                               : {delta:+.4f}%")
    print(f"  RMSE                                : {overall['rmse']:.4f}%")
    print(f"  MAPE                                : {overall['mape']:.4f}%")
    print(f"  Total EV02 rows evaluated           : {overall['n_rows']:,}")
    print(f"  Persistence baseline MAE            : {V3_PERSISTENCE_MAE:.3f}%")
    print(f"  vs persistence                      : {V3_PERSISTENCE_MAE - overall['mae']:+.4f}% better")

    if overall["mae"] < V3_VAL_MAE * 1.1:
        verdict = "GENERALISES — model transfers well to EV02 (same battery architecture)"
        action  = "Safe to deploy v3 on EV02. Fine-tune in 3–4 months when more EV02 data accumulates."
    elif overall["mae"] < V3_VAL_MAE * 1.5:
        verdict = "MODERATE DRIFT — acceptable but vehicles may differ in BMS calibration"
        action  = "Use v3 on EV02 short-term. Train an EV02-specific model once 30+ days of data exist."
    else:
        verdict = "HIGH DRIFT — EV02 behaviour differs significantly from EV01"
        action  = "Train an EV02-specific model. Investigate BMS calibration differences between vehicles."

    print(f"\n  Verdict : {verdict}")
    print(f"  Action  : {action}")
    print("=" * 60)

    print("\n  Per-day breakdown:")
    print(f"  {'Date':<14} {'Rows':>7} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("  " + "-" * 52)
    for d in per_day:
        flag = "  ← " if d["mae"] and d["mae"] > V3_VAL_MAE * 1.5 else ""
        print(f"  {d['date']:<14} {d['rows']:>7,} {d['mae']:>8.4f} {d['rmse']:>8.4f} {str(d['mape']):>8}{flag}")

    result = {
        "model":           "v3__2026-05-20__7b14ac62",
        "eval_vehicle":    VEHICLE_ID,
        "train_vehicle":   "EV01",
        "ev01_val_mae":    V3_VAL_MAE,
        "ev01_val_rmse":   V3_VAL_RMSE,
        "persistence_mae": V3_PERSISTENCE_MAE,
        "overall":         overall,
        "mae_delta":       round(delta, 4),
        "verdict":         verdict,
        "action":          action,
        "per_day":         per_day,
    }

    json_path = OUTPUT_DIR / "soc_eval_ev02_excavator.json"
    json_path.write_text(json.dumps(result, indent=2))

    md_lines = [
        "# SOC Model v3 — Cross-Vehicle Evaluation on EV02",
        "",
        f"**Model:** `v3__2026-05-20__7b14ac62` (trained on EV01 only)",
        f"**Evaluated on:** EV02 ({all_dates[0]} → {all_dates[-1]}, {overall['n_rows']:,} rows)",
        f"**Date:** {pd.Timestamp.now().date()}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | EV01 val (reference) | EV02 (never seen) | Delta |",
        "|---|---|---|---|",
        f"| MAE (%) | {V3_VAL_MAE:.4f} | {overall['mae']:.4f} | {delta:+.4f} |",
        f"| RMSE (%) | {V3_VAL_RMSE:.4f} | {overall['rmse']:.4f} | {overall['rmse'] - V3_VAL_RMSE:+.4f} |",
        f"| Persistence MAE | {V3_PERSISTENCE_MAE:.3f} | {V3_PERSISTENCE_MAE:.3f} | — |",
        "",
        f"**Verdict:** {verdict}",
        f"**Action:** {action}",
        "",
        "---",
        "",
        "## Per-Day Breakdown",
        "",
        "| Date | Rows | MAE (%) | RMSE (%) | MAPE (%) |",
        "|---|---|---|---|---|",
    ]
    for d in per_day:
        md_lines.append(f"| {d['date']} | {d['rows']:,} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['mape']} |")

    md_path = REPORT_DIR / "soc_eval_ev02_excavator.md"
    md_path.write_text("\n".join(md_lines))
    print(f"\n  JSON   → {json_path}")
    print(f"  Report → {md_path}")


if __name__ == "__main__":
    main()
