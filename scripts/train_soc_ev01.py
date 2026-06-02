"""
SOC Forecast — EV01 Loader (single retraining script)
======================================================
To retrain: add new dates to TRAIN_POST and/or VAL_DATES below, then run:

    python scripts/train_soc_ev01.py

The version number auto-increments — no new files needed ever.
Model is saved to models/soc_forecast/v<N>__<date>__<hash>/
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.settings import DATA_DIR

# ─────────────────────────────────────────────────────────────────────────────
# EDIT THIS SECTION WHEN RETRAINING
# ─────────────────────────────────────────────────────────────────────────────

VERSION = 4  # bump this each time you retrain

TRAIN_PRE = [
    "2026-02-10", "2026-02-13", "2026-02-14",
    "2026-03-04", "2026-03-08", "2026-03-13",
]

TRAIN_POST = [
    "2026-04-04", "2026-04-08", "2026-04-13",
    "2026-04-17", "2026-04-22", "2026-04-24",
    "2026-05-08", "2026-05-10",
    "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
    "2026-05-23", "2026-05-24", "2026-05-25", "2026-05-26", "2026-05-27",
    "2026-05-28",
]

# Last few days — must be AFTER all training dates
VAL_DATES = [
    "2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01",
]

# Previous version's val MAE (for comparison printout)
PREV_VAL_MAE  = 0.511
PREV_VERSION  = 3

# ─────────────────────────────────────────────────────────────────────────────

TRAIN_DATES  = TRAIN_PRE + TRAIN_POST
TARGET       = "y_soc_t_plus_300s"
VEHICLE_ID   = "EV01"
GOLD_DIR     = DATA_DIR / "gold" / "window_features"
MODEL_BASE   = PROJECT_ROOT / "models" / "soc_forecast"

FEATURES = [
    "soc_pct", "soc_pct_lag_60s", "soc_pct_lag_300s", "soc_pct_lag_600s",
    "soc_pct_roll_mean_60s", "soc_pct_roll_mean_300s",
    "soc_pct_roll_std_60s", "soc_pct_roll_std_300s",
    "battery_current_a", "battery_current_a_lag_60s", "battery_current_a_lag_300s",
    "battery_current_a_roll_mean_30s", "battery_current_a_roll_mean_300s",
    "battery_current_a_roll_std_60s", "battery_current_a_roll_max_60s",
    "battery_current_a_roll_min_60s",
    "stack_voltage_v", "stack_voltage_v_lag_60s", "stack_voltage_v_lag_300s",
    "stack_voltage_v_roll_mean_60s", "stack_voltage_v_roll_mean_300s",
    "stack_voltage_v_roll_std_60s",
    "max_cell_voltage_v", "min_cell_voltage_v",
    "cell_voltage_delta_v", "cell_voltage_delta_norm",
    "max_cell_voltage_v_roll_mean_60s", "min_cell_voltage_v_roll_mean_60s",
    "output_power_kw", "output_power_kw_lag_60s", "output_power_kw_lag_300s",
    "output_power_kw_roll_mean_60s", "output_power_kw_roll_mean_300s",
    "output_power_kw_roll_std_60s", "output_power_kw_roll_max_60s",
    "elec_power_kw_proxy",
    "motor_speed_rpm", "motor_speed_rpm_lag_60s",
    "motor_speed_rpm_roll_mean_60s", "motor_speed_rpm_roll_mean_300s",
    "motor_speed_rpm_roll_std_60s", "motor_speed_rpm_roll_max_60s",
    "avg_battery_temp_c", "avg_battery_temp_c_lag_60s", "avg_battery_temp_c_lag_300s",
    "avg_battery_temp_c_roll_mean_300s", "avg_battery_temp_c_roll_std_300s",
    "battery_temp_delta_c", "battery_temp_delta_norm",
    "motor_temperature_c", "motor_temperature_c_lag_300s",
    "motor_temperature_c_roll_mean_300s", "motor_temperature_c_roll_std_60s",
    "total_kwh_consumed", "total_kwh_consumed_lag_60s", "total_kwh_consumed_lag_300s",
    "fault_any",
    "power_proxy_error_kw", "power_proxy_ratio",
    "is_charging_current", "is_parked_charging",
]

MIN_TRIP_ROWS      = 60
MIN_TRIP_SOC_RANGE = 3.0


def load_dates(dates: list[str], label: str) -> pd.DataFrame:
    import duckdb
    needed = FEATURES + [TARGET, "label_available", "trip_id", "soc_pct"]
    frames = []
    for dt in dates:
        path = GOLD_DIR / f"dt={dt}" / f"vehicle_id={VEHICLE_ID}"
        if not path.exists():
            print(f"  WARNING: {dt} not found, skipping")
            continue
        parts = list(path.glob("*.parquet"))
        day_rows = 0
        for f in parts:
            available = set(
                duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{f}')").df()["column_name"].tolist()
            )
            read_cols = [c for c in needed if c in available]
            cols_sql  = ", ".join(f'"{c}"' for c in read_cols)
            chunk = duckdb.query(f"SELECT {cols_sql} FROM read_parquet('{f}')").df()
            for col in FEATURES:
                if col not in chunk.columns:
                    chunk[col] = 0.0
            chunk = chunk[chunk["label_available"] == 1].dropna(subset=[TARGET, "soc_pct"])
            if chunk.empty:
                continue
            frames.append(chunk)
            day_rows += len(chunk)
        print(f"  {dt}: {day_rows:,} rows")
    if not frames:
        raise RuntimeError(f"No data loaded for {label}")
    df = pd.concat(frames, ignore_index=True)
    print(f"  --- {label} total: {len(df):,} rows, {df['trip_id'].nunique()} trips")
    return df


def filter_micro_trips(df: pd.DataFrame, label: str) -> pd.DataFrame:
    stats = df.groupby("trip_id").agg(
        rows=("soc_pct", "count"),
        soc_range=("soc_pct", lambda x: x.max() - x.min()),
    )
    valid = stats[
        (stats["rows"] >= MIN_TRIP_ROWS) & (stats["soc_range"] >= MIN_TRIP_SOC_RANGE)
    ].index
    skipped = len(stats) - len(valid)
    if skipped:
        print(f"  {label}: skipped {skipped} micro-trips")
    filtered = df[df["trip_id"].isin(valid)]
    print(f"  {label} after filter: {len(filtered):,} rows, {filtered['trip_id'].nunique()} trips")
    return filtered


def compute_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    mae  = float(mean_absolute_error(y, yhat))
    rmse = float(np.sqrt(mean_squared_error(y, yhat)))
    mask = np.abs(y) > 0.5
    mape = float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100) if mask.sum() > 0 else float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"  SOC Forecast — v{VERSION} (EV01 Loader)")
print("=" * 60)
print(f"\nTrain dates ({len(TRAIN_DATES)}): {TRAIN_DATES[0]} → {TRAIN_DATES[-1]}")
print(f"Val dates   ({len(VAL_DATES)}): {VAL_DATES[0]} → {VAL_DATES[-1]}")

print("\nLoading train data …")
train_df = filter_micro_trips(load_dates(TRAIN_DATES, "TRAIN"), "TRAIN")

print("\nLoading val data …")
val_df = filter_micro_trips(load_dates(VAL_DATES, "VAL"), "VAL")

X_train = train_df[FEATURES].fillna(0).astype("float32")
y_train = train_df[TARGET].astype("float32").to_numpy()
X_val   = val_df[FEATURES].fillna(0).astype("float32")
y_val   = val_df[TARGET].astype("float32").to_numpy()

print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}")
print("\nTraining XGBoost …")

model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42,
    early_stopping_rounds=40,
    eval_metric="mae",
)

model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

train_metrics    = compute_metrics(y_train, model.predict(X_train))
val_metrics      = compute_metrics(y_val,   model.predict(X_val))
persistence_mae  = float(mean_absolute_error(y_val, val_df["soc_pct"].astype(float).to_numpy()))
improvement      = PREV_VAL_MAE - val_metrics["mae"]

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Train  MAE={train_metrics['mae']:.4f}%  RMSE={train_metrics['rmse']:.4f}%")
print(f"  Val    MAE={val_metrics['mae']:.4f}%  RMSE={val_metrics['rmse']:.4f}%")
print(f"  Persistence baseline MAE={persistence_mae:.4f}%")
print(f"  vs v{PREV_VERSION} ({PREV_VAL_MAE}%): {improvement:+.4f}%  {'IMPROVED' if improvement > 0 else 'REGRESSED'}")
print(f"  Best iteration: {model.best_iteration}")

imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Top 10 features:")
for feat, score in imp.head(10).items():
    print(f"    {feat:<45s} {score:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
run_date   = date.today().isoformat()
config_str = f"v{VERSION}_{run_date}_train={len(TRAIN_DATES)}days"
run_hash   = hashlib.md5(config_str.encode()).hexdigest()[:8]
run_id     = f"v{VERSION}__{run_date}__{run_hash}"
model_dir  = MODEL_BASE / run_id
model_dir.mkdir(parents=True, exist_ok=True)

model.save_model(str(model_dir / "model.json"))

(model_dir / "feature_set.json").write_text(json.dumps({
    "task": "soc_forecast", "target": TARGET,
    "features": FEATURES, "n_features": len(FEATURES),
}, indent=2))

status = "EXCELLENT" if val_metrics["mae"] < 0.5 else "GOOD" if val_metrics["mae"] < 0.8 else "ACCEPTABLE"
(model_dir / "eval_report.json").write_text(json.dumps({
    "run_id": run_id, "version": VERSION, "status": status,
    "train_dates": TRAIN_DATES, "val_dates": VAL_DATES,
    "train_metrics": train_metrics, "val_metrics": val_metrics,
    "train_val_gap": round(val_metrics["mae"] - train_metrics["mae"], 3),
    "baselines": {"persistence_mae": round(persistence_mae, 3)},
    "vs_prev": {
        f"v{PREV_VERSION}_val_mae": PREV_VAL_MAE,
        f"v{VERSION}_val_mae": val_metrics["mae"],
        "delta": round(val_metrics["mae"] - PREV_VAL_MAE, 4),
        "verdict": "IMPROVED" if improvement > 0 else "REGRESSED",
    },
    "best_iteration": int(model.best_iteration),
    "top_features": imp.head(20).to_dict(),
}, indent=2))

val_df[FEATURES].fillna(0).astype("float32").describe().T.reset_index().rename(
    columns={"index": "feature"}
).to_parquet(str(model_dir / "drift_baseline.parquet"), index=False)

(model_dir / "config.yaml").write_text(yaml.dump({
    "run_id": run_id, "version": VERSION,
    "train_dates": TRAIN_DATES, "val_dates": VAL_DATES,
    "target": TARGET, "vehicle_id": VEHICLE_ID, "n_features": len(FEATURES),
}, sort_keys=False))

print(f"\n{'='*60}")
print(f"  Model saved → {model_dir}")
print(f"  Run ID      : {run_id}")
print(f"  Status      : {status}")
print(f"{'='*60}")
print(f"\nNext: update tasks/soc_forecast/config.yaml")
print(f"  model_path: models/soc_forecast/{run_id}/model.json")
