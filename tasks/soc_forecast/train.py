from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import xgboost as xgb
import yaml

from tasks.soc_forecast.evaluate import evaluate
from tasks.soc_forecast.feature_set import (
    FEATURES,
    MIN_TRIP_ROWS,
    MIN_TRIP_SOC_RANGE,
    TARGET,
)
from training.artifacts import make_run_id, save_run
from training.dataset import filter_micro_trips, fingerprint_gold, load_gold_dates

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def run_training():
    # ── Load config ────────────────────────────────────────────────────────────
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    task = cfg["task"]
    target = cfg["target"]
    train_dates = cfg["data"]["train_dates"]
    val_dates = cfg["data"]["val_dates"]
    vehicles = cfg["data"]["vehicles"]
    drop_mask = int(cfg["data"].get("drop_quality_mask", 52))
    model_cfg = cfg["model"]
    thresholds = cfg.get("thresholds", {})

    if target != TARGET:
        raise ValueError(
            f"Config target '{target}' does not match feature_set TARGET '{TARGET}'"
        )

    print(f"\n{'='*60}")
    print(f"  TASK:    {task}")
    print(f"  TARGET:  {target}")
    print(f"  TRAIN:   {train_dates}")
    print(f"  VAL:     {val_dates}")
    print(f"  VEHICLES:{vehicles}")
    print(f"{'='*60}")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading train data...")
    train_df = load_gold_dates(
        dates=train_dates,
        vehicles=vehicles,
        features=FEATURES,
        target=target,
        drop_quality_mask=drop_mask,
    )

    print("\nLoading val data...")
    val_raw = load_gold_dates(
        dates=val_dates,
        vehicles=vehicles,
        features=FEATURES,
        target=target,
        drop_quality_mask=drop_mask,
    )

    print("\nFiltering micro-trips from val...")
    val_df = filter_micro_trips(
        val_raw,
        min_rows=MIN_TRIP_ROWS,
        min_soc_range=MIN_TRIP_SOC_RANGE,
    )

    # ── Build X / y ────────────────────────────────────────────────────────────
    X_train = train_df[FEATURES].astype("float32")
    y_train = train_df[target].astype("float32")
    X_val = val_df[FEATURES].astype("float32")
    y_val = val_df[target].astype("float32")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(
        f"\nTraining XGBoost  "
        f"(train={len(X_train):,} rows, val={len(X_val):,} rows)..."
    )

    model = xgb.XGBRegressor(
        n_estimators=int(model_cfg.get("n_estimators", 800)),
        max_depth=int(model_cfg.get("max_depth", 6)),
        learning_rate=float(model_cfg.get("learning_rate", 0.05)),
        subsample=float(model_cfg.get("subsample", 0.8)),
        colsample_bytree=float(model_cfg.get("colsample_bytree", 0.8)),
        min_child_weight=int(model_cfg.get("min_child_weight", 10)),
        reg_alpha=float(model_cfg.get("reg_alpha", 0.1)),
        reg_lambda=float(model_cfg.get("reg_lambda", 1.0)),
        early_stopping_rounds=int(model_cfg.get("early_stopping_rounds", 30)),
        eval_metric=str(model_cfg.get("eval_metric", "mae")),
        n_jobs=-1,
        random_state=int(model_cfg.get("random_state", 42)),
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    run_id = make_run_id(task, train_dates[-1])

    eval_report = evaluate(
        model=model,
        train_df=train_df,
        val_df=val_df,
        features=FEATURES,
        target=target,
        run_id=run_id,
        thresholds=thresholds,
    )

    # ── Data fingerprint ───────────────────────────────────────────────────────
    print("\nFingerprinting training data...")
    parquet_hashes = fingerprint_gold(train_dates + val_dates, vehicles)

    # ── Save versioned run ─────────────────────────────────────────────────────
    run_dir = save_run(
        task=task,
        run_id=run_id,
        model=model,
        features=FEATURES,
        target=target,
        train_dates=train_dates,
        val_dates=val_dates,
        train_rows=len(X_train),
        val_rows=len(X_val),
        train_trips=int(train_df["trip_id"].nunique()),
        val_trips=int(val_df["trip_id"].nunique()),
        parquet_hashes=parquet_hashes,
        eval_report=eval_report,
        train_df=train_df,
        config_path=CONFIG_PATH,
    )

    print(f"\n{'='*60}")
    print(f"  Run ID:  {run_id}")
    print(f"  Status:  {eval_report['status']}")
    print(f"  Val MAE: {eval_report['val_metrics']['mae']:.3f}%")
    print(f"  Saved →  {run_dir}")
    print(f"{'='*60}\n")

    return run_dir


if __name__ == "__main__":
    run_training()