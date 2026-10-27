"""
Optuna Hyperparameter Tuning — SOC Forecast (EV01 Loader)
==========================================================
Runs a Bayesian search over XGBoost hyperparameters using Optuna's TPE sampler.
Each trial trains on the same train dates as v4 and evaluates on the same val dates,
so the search is directly comparable to the production model.

Best params are saved to outputs/optuna_best_params.json.
Every trial is logged to MLflow (if server is running); the study itself is also
persisted to outputs/optuna_study.db so you can resume or inspect it later.

Usage:
    python scripts/tune_soc.py                 # 50 trials (default)
    python scripts/tune_soc.py --trials 100    # more trials
    python scripts/tune_soc.py --trials 10     # quick smoke-test

How to read the output:
    Best MAE < 0.5419 → better than current v4 production model
    Best MAE > 0.5419 → v4 params were already near-optimal for this data
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import duckdb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import get_logger
from configs.settings import DATA_DIR, settings

logger = get_logger(__name__)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)   # suppress per-trial noise

# ── Config ────────────────────────────────────────────────────────────────────

GOLD_DIR   = DATA_DIR / "gold" / "window_features"
VEHICLE_ID = "EV01"
TARGET     = "y_soc_t_plus_300s"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Same dates as v4 — makes comparison fair
TRAIN_DATES = [
    "2026-02-10", "2026-02-13", "2026-02-14",
    "2026-03-04", "2026-03-08", "2026-03-13",
    "2026-04-04", "2026-04-08", "2026-04-13", "2026-04-17", "2026-04-22", "2026-04-24",
    "2026-05-08", "2026-05-10",
    "2026-05-18", "2026-05-19", "2026-05-20", "2026-05-21", "2026-05-22",
    "2026-05-23", "2026-05-24", "2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28",
]
VAL_DATES = ["2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01"]

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

# v4 baseline to compare against
V4_VAL_MAE = 0.5419


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dates(dates: list[str], label: str) -> pd.DataFrame:
    needed = FEATURES + [TARGET, "label_available", "soc_pct", "trip_id"]
    frames = []
    for dt in dates:
        path = GOLD_DIR / f"dt={dt}" / f"vehicle_id={VEHICLE_ID}"
        if not path.exists():
            logger.warning("Date not found, skipping: %s", dt)
            continue
        for f in path.glob("*.parquet"):
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
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        raise RuntimeError(f"No data found for {label}")
    df = pd.concat(frames, ignore_index=True)
    # Filter micro-trips
    stats = df.groupby("trip_id").agg(
        rows=("soc_pct", "count"),
        soc_range=("soc_pct", lambda x: x.max() - x.min()),
    )
    valid = stats[(stats["rows"] >= 60) & (stats["soc_range"] >= 3.0)].index
    df = df[df["trip_id"].isin(valid)]
    logger.info("%s: %d rows, %d trips", label, len(df), df["trip_id"].nunique())
    return df


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
            "max_depth":        trial.suggest_int("max_depth", 3, 9),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_jobs":           -1,
            "random_state":     42,
            "eval_metric":      "mae",
            "early_stopping_rounds": 30,
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        mae = float(mean_absolute_error(y_val, model.predict(X_val)))
        return mae

    return objective


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(study: optuna.Study, best_mae: float) -> None:
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_uri)
        mlflow.set_experiment("soc_forecast_tuning")
        with mlflow.start_run(run_name="optuna_tune_v4"):
            mlflow.log_params(study.best_params)
            mlflow.log_metric("best_val_mae", best_mae)
            mlflow.log_metric("n_trials", len(study.trials))
            mlflow.log_metric("v4_baseline_mae", V4_VAL_MAE)
            mlflow.log_metric("improvement_vs_v4", V4_VAL_MAE - best_mae)
        logger.info("Results logged to MLflow experiment 'soc_forecast_tuning'")
    except Exception as e:
        logger.warning("MLflow logging skipped (%s) — server may not be running", type(e).__name__)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    args = ap.parse_args()

    logger.info("=" * 60)
    logger.info("  Optuna Hyperparameter Tuning — SOC Forecast")
    logger.info("=" * 60)
    logger.info("Trials: %d  |  Baseline (v4 val MAE): %.4f%%", args.trials, V4_VAL_MAE)

    # Load data once — shared across all trials
    logger.info("Loading training data …")
    train_df = load_dates(TRAIN_DATES, "TRAIN")
    logger.info("Loading val data …")
    val_df   = load_dates(VAL_DATES,   "VAL")

    X_train = train_df[FEATURES].fillna(0).astype("float32")
    y_train = train_df[TARGET].astype("float32").to_numpy()
    X_val   = val_df[FEATURES].fillna(0).astype("float32")
    y_val   = val_df[TARGET].astype("float32").to_numpy()

    logger.info("X_train: %s  X_val: %s", X_train.shape, X_val.shape)

    # Optuna study — persisted to SQLite so you can resume
    OUTPUT_DIR.mkdir(exist_ok=True)
    db_path = OUTPUT_DIR / "optuna_study.db"
    storage = f"sqlite:///{db_path}"

    study = optuna.create_study(
        study_name="soc_forecast_v4_tune",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    # Show progress every 10 trials
    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

    def progress_callback(study, trial):
        n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if n % 10 == 0:
            logger.info(
                "Trial %d/%d  current=%.4f%%  best=%.4f%%",
                n, args.trials + completed_before, trial.value or float("nan"), study.best_value,
            )

    logger.info("Starting %d trials (TPE sampler) …", args.trials)
    study.optimize(
        make_objective(X_train, y_train, X_val, y_val),
        n_trials=args.trials,
        callbacks=[progress_callback],
        show_progress_bar=False,
    )

    best_mae    = study.best_value
    best_params = study.best_params
    improvement = V4_VAL_MAE - best_mae

    # ── Results ───────────────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("  TUNING RESULTS")
    logger.info("=" * 60)
    logger.info("Best val MAE:   %.4f%%", best_mae)
    logger.info("v4 baseline:    %.4f%%", V4_VAL_MAE)
    logger.info("Improvement:    %+.4f%%  (%s)", improvement,
                "BETTER than v4" if improvement > 0 else "v4 was already near-optimal")
    logger.info("Best params:")
    for k, v in best_params.items():
        logger.info("  %-25s = %s", k, v)

    # Top 5 trials
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    top5 = sorted(completed, key=lambda t: t.value)[:5]
    logger.info("Top 5 trials by val MAE:")
    for i, t in enumerate(top5, 1):
        logger.info("  #%d  trial=%d  MAE=%.4f%%", i, t.number, t.value)

    # ── Save ─────────────────────────────────────────────────────────────────

    output = {
        "vehicle_id":      VEHICLE_ID,
        "n_trials_run":    args.trials,
        "n_trials_total":  len(completed),
        "v4_val_mae":      V4_VAL_MAE,
        "best_val_mae":    round(best_mae, 4),
        "improvement":     round(improvement, 4),
        "verdict":         "IMPROVED" if improvement > 0 else "V4_NEAR_OPTIMAL",
        "best_params":     best_params,
        "top5_trials": [
            {"trial": t.number, "mae": round(t.value, 4), "params": t.params}
            for t in top5
        ],
    }
    out_path = OUTPUT_DIR / "optuna_best_params.json"
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("Saved: %s", out_path)
    logger.info("Study DB: %s  (resume with --trials N)", db_path)

    # Log to MLflow
    log_to_mlflow(study, best_mae)

    logger.info("=" * 60)

    if improvement > 0.01:
        logger.info("Recommendation: retrain with best_params (%.4f%% improvement)", improvement)
        logger.info("  Update n_estimators/max_depth/learning_rate in train_soc_ev01.py")
    else:
        logger.info("v4 params are near-optimal for this dataset — no retraining needed")


if __name__ == "__main__":
    main()
