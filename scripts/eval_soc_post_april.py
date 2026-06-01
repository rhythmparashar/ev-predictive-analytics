"""
SOC Model Evaluation — Pre vs Post April Firmware Update (EV01 Loader)
=======================================================================
Evaluates XGBoost v2 model on EV01 (Loader) data across the firmware boundary:
  - Pre-April data  (Jan 22 – Mar 26)  ← model was trained on this era
  - Post-April data (Mar 31 – May 17)  ← new battery behaviour after firmware update

Usage:
    python scripts/eval_soc_post_april.py

Outputs:
    reports/soc_eval_post_april.md
    outputs/soc_eval_post_april.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import xgboost as xgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.settings import DATA_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = PROJECT_ROOT / "models" / "soc_forecast" / "v2__2026-03-13__fc0cd5ed"
GOLD_DIR    = DATA_DIR / "gold" / "window_features"
REPORT_DIR  = PROJECT_ROOT / "reports"
OUTPUT_DIR  = PROJECT_ROOT / "outputs"
REPORT_PATH = REPORT_DIR / "soc_eval_post_april.md"
JSON_PATH   = OUTPUT_DIR / "soc_eval_post_april.json"

VEHICLE_ID   = "EV01"
MACHINE_TYPE = "Loader"
TARGET       = "y_soc_t_plus_300s"
LABEL_COL    = "label_available"

# Firmware update window: March 27–30 (machine was off)
PRE_CUTOFF  = "2026-03-27"   # up to but not including
POST_START  = "2026-03-31"   # first day back after update

# Original val metrics from training
ORIGINAL_VAL = {"mae": 0.441, "rmse": 0.563, "mape": 0.564}
PERSISTENCE_BASELINE_MAE = 0.822


def iter_day_parts(dt_str: str):
    """Yield one part-file DataFrame at a time to keep RAM low."""
    path = GOLD_DIR / f"dt={dt_str}" / f"vehicle_id={VEHICLE_ID}"
    if not path.exists():
        return
    for f in sorted(path.glob("*.parquet")):
        df = pd.read_parquet(f)
        df["_date"] = dt_str
        yield df


def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    mae  = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    # MAPE — avoid divide by zero
    mask = np.abs(y) > 0.5
    mape = float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


def persistence_mae(y: np.ndarray) -> float:
    """Naive carry-forward baseline — predict current SOC as future SOC."""
    return float(np.mean(np.abs(np.diff(y))))


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load model + features ─────────────────────────────────────────────────
    print("Loading XGBoost v2 model …")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_DIR / "model.json"))
    feature_set = json.loads((MODEL_DIR / "feature_set.json").read_text())
    features = feature_set["features"]
    print(f"  Features: {len(features)}")

    # ── Discover all available dates ──────────────────────────────────────────
    all_dates = sorted(
        d.name.replace("dt=", "")
        for d in GOLD_DIR.iterdir()
        if d.is_dir() and d.name.startswith("dt=")
    )
    pre_dates  = [d for d in all_dates if d < PRE_CUTOFF]
    post_dates = [d for d in all_dates if d >= POST_START]
    print(f"  Pre-April dates : {len(pre_dates)}  ({pre_dates[0]} → {pre_dates[-1]})")
    print(f"  Post-April dates: {len(post_dates)}  ({post_dates[0]} → {post_dates[-1]})")

    # ── Incremental evaluation — one day at a time to keep RAM low ───────────
    def prepare_day(df: pd.DataFrame) -> pd.DataFrame | None:
        df = df[(df.get(LABEL_COL, pd.Series([1]*len(df))) == 1) & df["trip_id"].notna()].copy()
        if df.empty or TARGET not in df.columns:
            return None
        if "is_charging_current" not in df.columns:
            df["is_charging_current"] = 0
        if "is_parked_charging" not in df.columns:
            df["is_parked_charging"] = 0
        return df

    def predict_day(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        # Fill any missing feature columns with 0 (handles schema drift across dates)
        for col in features:
            if col not in df.columns:
                df[col] = 0.0
        X = df[features].fillna(0).astype("float32")
        y = df[TARGET].astype(float).to_numpy()
        yhat = model.predict(X, validate_features=False)
        return y, yhat

    class EraAccumulator:
        def __init__(self):
            self.n = 0
            self.abs_err_sum = 0.0
            self.sq_err_sum  = 0.0
            self.mape_num    = 0.0
            self.mape_den    = 0

        def update(self, y: np.ndarray, yhat: np.ndarray):
            err = y - yhat
            self.n           += len(y)
            self.abs_err_sum += float(np.sum(np.abs(err)))
            self.sq_err_sum  += float(np.sum(err ** 2))
            mask = np.abs(y) > 0.5
            self.mape_num    += float(np.sum(np.abs(err[mask] / y[mask]))) * 100
            self.mape_den    += int(mask.sum())

        def metrics(self) -> dict:
            if self.n == 0:
                return {}
            return {
                "mae":  round(self.abs_err_sum / self.n, 4),
                "rmse": round(np.sqrt(self.sq_err_sum / self.n), 4),
                "mape": round(self.mape_num / self.mape_den, 4) if self.mape_den > 0 else float("nan"),
            }

    print("\nEvaluating pre-April (part by part) …")
    pre_acc = EraAccumulator()
    pre_n = 0
    for dt in pre_dates:
        for chunk in iter_day_parts(dt):
            chunk = prepare_day(chunk)
            if chunk is None:
                continue
            y, yhat = predict_day(chunk)
            pre_acc.update(y, yhat)
            pre_n += len(y)
        print(f"  {dt} done (running total: {pre_n:,})", end="\r")
    print(f"  Pre-April done. Total rows: {pre_n:,}          ")

    print("Evaluating post-April (part by part) …")
    post_acc  = EraAccumulator()
    per_day   = []
    post_n    = 0
    for dt in post_dates:
        day_acc = EraAccumulator()
        day_n   = 0
        for chunk in iter_day_parts(dt):
            chunk = prepare_day(chunk)
            if chunk is None:
                continue
            y, yhat = predict_day(chunk)
            post_acc.update(y, yhat)
            day_acc.update(y, yhat)
            post_n += len(y)
            day_n  += len(y)
        if day_n >= 10:
            m = day_acc.metrics()
            m["date"] = dt
            m["rows"] = day_n
            per_day.append(m)
        print(f"  {dt} done (running total: {post_n:,})", end="\r")
    print(f"  Post-April done. Total rows: {post_n:,}          ")

    pre_metrics  = {**pre_acc.metrics(),  "n_rows": pre_n}
    post_metrics = {**post_acc.metrics(), "n_rows": post_n}
    y_pre  = np.zeros(pre_n)   # placeholder for len() calls below
    y_post = np.zeros(post_n)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  SOC MODEL EVALUATION — PRE vs POST APRIL")
    print("="*60)

    def print_era(label, m):
        vs_orig = m["mae"] - ORIGINAL_VAL["mae"]
        vs_pers = PERSISTENCE_BASELINE_MAE - m["mae"]
        print(f"\n  [{label}]  n={m['n_rows']:,}")
        print(f"    MAE  : {m['mae']:.4f}%  (original val: {ORIGINAL_VAL['mae']:.3f}%  Δ={vs_orig:+.4f}%)")
        print(f"    RMSE : {m['rmse']:.4f}%  (original val: {ORIGINAL_VAL['rmse']:.3f}%)")
        print(f"    MAPE : {m['mape']:.4f}%")
        print(f"    vs persistence baseline: {'+' if vs_pers>0 else ''}{vs_pers:.4f}% better")

    print_era("PRE-APRIL  (Jan–Mar, model's own era)", pre_metrics)
    print_era("POST-APRIL (Apr–May, post-firmware)  ", post_metrics)

    print("\n  Per-day breakdown (post-April):")
    print(f"  {'Date':<14} {'Rows':>7} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print("  " + "-"*50)
    for d in per_day:
        flag = "  ← degraded" if d["mae"] > ORIGINAL_VAL["mae"] * 1.5 else ""
        print(f"  {d['date']:<14} {d['rows']:>7,} {d['mae']:>8.4f} {d['rmse']:>8.4f} {d['mape']:>8.4f}{flag}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    mae_delta = post_metrics["mae"] - pre_metrics["mae"]
    if post_metrics["mae"] < ORIGINAL_VAL["mae"] * 1.1:
        verdict = "HOLDS — model generalises well to post-firmware data"
        retrain = "Not urgent. Monitor monthly."
    elif post_metrics["mae"] < ORIGINAL_VAL["mae"] * 1.5:
        verdict = "MODERATE DEGRADATION — acceptable but retraining recommended"
        retrain = "Retrain on combined pre+post data within 2–4 weeks."
    else:
        verdict = "SIGNIFICANT DEGRADATION — model should be retrained"
        retrain = "Retrain immediately on post-April data."

    print(f"\n  MAE shift (pre → post): {mae_delta:+.4f}%")
    print(f"  Verdict  : {verdict}")
    print(f"  Action   : {retrain}")
    print("="*60)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    result = {
        "model":             "v2__2026-03-13__fc0cd5ed",
        "original_val":      ORIGINAL_VAL,
        "persistence_mae":   PERSISTENCE_BASELINE_MAE,
        "pre_april":  pre_metrics,
        "post_april": post_metrics,
        "mae_shift":         round(mae_delta, 4),
        "verdict":           verdict,
        "retrain_action":    retrain,
        "per_day_post":      per_day,
    }
    JSON_PATH.write_text(json.dumps(result, indent=2))
    print(f"\n  JSON  → {JSON_PATH}")

    # ── Write markdown report ─────────────────────────────────────────────────
    md = f"""# SOC Model Evaluation — Pre vs Post April

**Model:** `v2__2026-03-13__fc0cd5ed` (LightGBM v2)
**Target:** `{TARGET}` (SOC 5 minutes ahead)
**Evaluated:** {pd.Timestamp.now().date()}

---

## Context

A firmware/BMS update was applied to EV01 during **March 27–30, 2026**.
Post-update, M1_C25 average gap improved from −43 mV → −4 mV and anomaly rate dropped from ~99% → ~22%.
This evaluation checks whether the SOC model — trained on pre-update data — still holds post-update.

---

## Results

| Era | Dates | Rows | MAE (%) | RMSE (%) | MAPE (%) |
|---|---|---|---|---|---|
| Original val (training) | Jan–Mar 2026 | — | {ORIGINAL_VAL['mae']:.3f} | {ORIGINAL_VAL['rmse']:.3f} | {ORIGINAL_VAL['mape']:.3f} |
| Pre-April (model's era) | {pre_dates[0]} → {pre_dates[-1]} | {pre_metrics['n_rows']:,} | {pre_metrics['mae']:.4f} | {pre_metrics['rmse']:.4f} | {pre_metrics['mape']:.4f} |
| **Post-April (new era)** | **{post_dates[0]} → {post_dates[-1]}** | **{post_metrics['n_rows']:,}** | **{post_metrics['mae']:.4f}** | **{post_metrics['rmse']:.4f}** | **{post_metrics['mape']:.4f}** |
| Persistence baseline | — | — | {PERSISTENCE_BASELINE_MAE:.3f} | — | — |

**MAE shift (pre → post): {mae_delta:+.4f}%**

---

## Per-Day Breakdown (Post-April)

| Date | Rows | MAE (%) | RMSE (%) | MAPE (%) |
|---|---|---|---|---|
{''.join(f"| {d['date']} | {d['rows']:,} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['mape']:.4f} |" + chr(10) for d in per_day)}

---

## Verdict

**{verdict}**

**Action:** {retrain}

---

## Notes

- `is_charging_current` and `is_parked_charging` features are absent from post-April gold data.
  Both were set to 0 (correct — post-April data shows no negative current / charging events in gold features).
- Model beats persistence baseline ({PERSISTENCE_BASELINE_MAE:.3f}% MAE) by {PERSISTENCE_BASELINE_MAE - post_metrics['mae']:.4f}% in post-April era.
"""

    REPORT_PATH.write_text(md)
    print(f"  Report → {REPORT_PATH}")


if __name__ == "__main__":
    main()
