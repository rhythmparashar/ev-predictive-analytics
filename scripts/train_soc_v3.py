"""
SOC Forecast Model — v3 Retraining (EV01 Loader)
=================================================
Trains XGBoost on a balanced pre + post April dataset for EV01 (Loader).
Validates on held-out May dates (time-ordered split).

Changes from v2:
  - Training data expanded from 4 days (Feb only) to 14 days (Feb + Mar + Apr + May)
  - Balanced pre/post firmware split: 6 pre-April + 8 post-April days
  - Validation: last 4 days of May (time-ordered, unseen era)
  - Part-file loading to avoid OOM

Usage:
    python scripts/train_soc_v3.py

Outputs:
    models/soc_forecast/v3__<date>__<hash>/
        model.json
        feature_set.json
        eval_report.json
        drift_baseline.parquet
        config.yaml
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from configs.settings import DATA_DIR

# ── Date splits ───────────────────────────────────────────────────────────────
# Pre-April: diverse days, skip Feb-23 (801K rows, outlier day)
TRAIN_PRE = [
    "2026-02-10", "2026-02-13", "2026-02-14",
    "2026-03-04", "2026-03-08", "2026-03-13",
]

# Post-April: diverse days across Apr + early May
TRAIN_POST = [
    "2026-04-04", "2026-04-08", "2026-04-13",
    "2026-04-17", "2026-04-22", "2026-04-24",
    "2026-05-08", "2026-05-10",
]

TRAIN_DATES = TRAIN_PRE + TRAIN_POST

# Val: last 4 days of May — fully unseen, post-firmware era
VAL_DATES = [
    "2026-05-14", "2026-05-15", "2026-05-16", "2026-05-17",
]

TARGET       = "y_soc_t_plus_300s"
VEHICLE_ID   = "EV01"
MACHINE_TYPE = "Loader"
GOLD_DIR     = DATA_DIR / "gold" / "window_features"
MODEL_BASE  = PROJECT_ROOT / "models" / "soc_forecast"

# ── Feature set (same 61 as v2) ───────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────

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
            # DuckDB handles duplicate column names gracefully (unlike pyarrow/pandas)
            available = set(
                duckdb.query(f"DESCRIBE SELECT * FROM read_parquet('{f}')").df()["column_name"].tolist()
            )
            read_cols = [c for c in needed if c in available]
            cols_sql = ", ".join(f'"{c}"' for c in read_cols)
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
    valid = stats[(stats["rows"] >= MIN_TRIP_ROWS) & (stats["soc_range"] >= MIN_TRIP_SOC_RANGE)].index
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


# ── Load data ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  SOC Forecast — v3 Retraining")
print("=" * 60)

print(f"\nTrain dates ({len(TRAIN_DATES)}): {TRAIN_DATES[0]} → {TRAIN_DATES[-1]}")
print(f"  Pre-April : {TRAIN_PRE}")
print(f"  Post-April: {TRAIN_POST}")
print(f"Val dates   : {VAL_DATES}")

print("\nLoading train data …")
train_raw = load_dates(TRAIN_DATES, "TRAIN")
train_df  = filter_micro_trips(train_raw, "TRAIN")

print("\nLoading val data …")
val_raw = load_dates(VAL_DATES, "VAL")
val_df  = filter_micro_trips(val_raw, "VAL")

X_train = train_df[FEATURES].fillna(0).astype("float32")
y_train = train_df[TARGET].astype("float32").to_numpy()

X_val = val_df[FEATURES].fillna(0).astype("float32")
y_val = val_df[TARGET].astype("float32").to_numpy()

print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}")

# ── Train ─────────────────────────────────────────────────────────────────────
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

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# ── Metrics ───────────────────────────────────────────────────────────────────
pred_train = model.predict(X_train)
pred_val   = model.predict(X_val)

train_metrics = compute_metrics(y_train, pred_train)
val_metrics   = compute_metrics(y_val,   pred_val)

# Persistence baseline on val
persistence_mae = float(mean_absolute_error(y_val, val_df["soc_pct"].astype(float).to_numpy()))

v2_val_mae = 0.441
improvement = v2_val_mae - val_metrics["mae"]

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")
print(f"  Train  MAE={train_metrics['mae']:.4f}%  RMSE={train_metrics['rmse']:.4f}%")
print(f"  Val    MAE={val_metrics['mae']:.4f}%  RMSE={val_metrics['rmse']:.4f}%")
print(f"  Persistence baseline MAE={persistence_mae:.4f}%")
print(f"  vs v2 val MAE ({v2_val_mae}%): {improvement:+.4f}%  {'IMPROVED' if improvement > 0 else 'REGRESSED'}")
print(f"  Best iteration: {model.best_iteration}")

# Feature importance
imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Top 10 features:")
for feat, score in imp.head(10).items():
    print(f"    {feat:<45s} {score:.4f}")

# ── Version and save ──────────────────────────────────────────────────────────
from datetime import date
run_date = date.today().isoformat()
config_str = f"v3_{run_date}_train={len(TRAIN_DATES)}days_val={len(VAL_DATES)}days"
run_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
run_id = f"v3__{run_date}__{run_hash}"

model_dir = MODEL_BASE / run_id
model_dir.mkdir(parents=True, exist_ok=True)

# Save model
model.save_model(str(model_dir / "model.json"))

# feature_set.json
(model_dir / "feature_set.json").write_text(json.dumps({
    "task": "soc_forecast",
    "target": TARGET,
    "features": FEATURES,
    "n_features": len(FEATURES),
}, indent=2))

# eval_report.json
status = "EXCELLENT" if val_metrics["mae"] < 0.5 else "GOOD" if val_metrics["mae"] < 0.8 else "ACCEPTABLE"
eval_report = {
    "run_id": run_id,
    "task": "soc_forecast",
    "target": TARGET,
    "status": status,
    "train_dates": TRAIN_DATES,
    "val_dates": VAL_DATES,
    "train_pre_april": TRAIN_PRE,
    "train_post_april": TRAIN_POST,
    "train_metrics": train_metrics,
    "val_metrics": val_metrics,
    "train_val_gap": round(val_metrics["mae"] - train_metrics["mae"], 3),
    "baselines": {"persistence_mae": round(persistence_mae, 3)},
    "vs_v2": {
        "v2_val_mae": v2_val_mae,
        "v3_val_mae": val_metrics["mae"],
        "delta": round(val_metrics["mae"] - v2_val_mae, 4),
        "verdict": "IMPROVED" if improvement > 0 else "REGRESSED",
    },
    "best_iteration": int(model.best_iteration),
    "top_features": imp.head(20).to_dict(),
    "model_params": {
        "n_estimators": 600, "max_depth": 6, "learning_rate": 0.04,
        "subsample": 0.8, "colsample_bytree": 0.8,
    },
}
(model_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2))

# drift_baseline.parquet  (val feature distributions for production monitoring)
drift_baseline = X_val.describe().T.reset_index().rename(columns={"index": "feature"})
drift_baseline.to_parquet(str(model_dir / "drift_baseline.parquet"), index=False)

# config.yaml
config = {
    "run_id": run_id,
    "train_dates": TRAIN_DATES,
    "val_dates": VAL_DATES,
    "target": TARGET,
    "vehicle_id": VEHICLE_ID,
    "n_features": len(FEATURES),
}
(model_dir / "config.yaml").write_text(yaml.dump(config, sort_keys=False))

print(f"\n{'='*60}")
print(f"  Model saved → {model_dir}")
print(f"  Run ID      : {run_id}")
print(f"  Status      : {status}")
print(f"{'='*60}")
