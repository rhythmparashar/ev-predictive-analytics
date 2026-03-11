#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path


def _run_script(script: str, *args):
    """Run a project script as a subprocess, inheriting stdout/stderr."""
    import os

    env = os.environ.copy()
    project_root = str(Path(__file__).resolve().parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root

    cmd = [sys.executable, script, *args]
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


# ══════════════════════════════════════════════════════════════════════════════
#  INGEST
# ══════════════════════════════════════════════════════════════════════════════

def cmd_ingest(args):
    if args.backfill:
        if not args.start or not args.end:
            raise ValueError("--backfill requires --start and --end")
        _ingest_backfill(args.start, args.end)
    else:
        if not args.dt:
            raise ValueError("ingest requires --dt when not using --backfill")
        _ingest_day(args.dt)


def _ingest_day(dt: str):
    print(f"\n── Ingest {dt} ────────────────────────────────────────────")
    _run_script("scripts/run_day.py", "--dt", dt)


def _ingest_backfill(start: str, end: str):
    d = date.fromisoformat(start)
    end_d = date.fromisoformat(end)
    while d <= end_d:
        _ingest_day(d.isoformat())
        d += timedelta(days=1)


# ══════════════════════════════════════════════════════════════════════════════
#  GOLD
# ══════════════════════════════════════════════════════════════════════════════

def cmd_gold(args):
    print(f"\n── Gold features {args.dt} ─────────────────────────────────")
    _run_script("scripts/run_gold_day.py", "--dt", args.dt)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def cmd_train(args):
    task = args.task
    print(f"\n── Train  task={task} ──────────────────────────────────────")

    if task == "soc_forecast":
        from tasks.soc_forecast.train import run_training
        run_training()
    else:
        _unknown_task(task)


# ══════════════════════════════════════════════════════════════════════════════
#  EVAL
# ══════════════════════════════════════════════════════════════════════════════

def cmd_eval(args):
    task = args.task
    dt = args.dt
    run_id = getattr(args, "run_id", None)

    print(
        f"\n── Eval  task={task}  dt={dt}  run={'latest' if not run_id else run_id} ──"
    )

    if task == "soc_forecast":
        _eval_soc_forecast(dt, run_id)
    else:
        _unknown_task(task)


def _eval_soc_forecast(dt: str, run_id: str | None):
    import json
    import numpy as np
    import yaml

    from tasks.soc_forecast.feature_set import MIN_TRIP_ROWS, MIN_TRIP_SOC_RANGE
    from training.artifacts import load_latest_run, load_run_by_id
    from training.dataset import filter_micro_trips, load_gold_dates
    from monitoring.drift import compute_drift_report
    from evaluation.metrics import (
        regression_metrics,
        per_group_regression,
        soc_bucket_metrics,
        error_distribution,
        persistence_baseline,
        rolling_baseline,
    )

    CONFIG_PATH = Path("tasks/soc_forecast/config.yaml")
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    vehicles = cfg["data"]["vehicles"]
    drop_mask = int(cfg["data"].get("drop_quality_mask", 52))
    drift_th = cfg.get("drift", {})

    if run_id:
        model, feature_set, run_dir = load_run_by_id("soc_forecast", run_id)
    else:
        model, feature_set, run_dir = load_latest_run("soc_forecast")

    saved_features = feature_set["features"]
    saved_target = feature_set["target"]

    print(f"\n  Model run:  {run_dir.name}")
    print(f"  Features:   {len(saved_features)}")

    print(f"\nLoading Gold for {dt}...")
    raw_df = load_gold_dates(
        dates=[dt],
        vehicles=vehicles,
        features=saved_features,
        target=saved_target,
        drop_quality_mask=drop_mask,
    )

    eval_df = filter_micro_trips(
        raw_df,
        min_rows=MIN_TRIP_ROWS,
        min_soc_range=MIN_TRIP_SOC_RANGE,
    )

    if eval_df.empty:
        raise RuntimeError(f"No evaluation rows available for dt={dt} after filtering.")

    print("\nRunning drift check...")
    drift_report = compute_drift_report(
        dt=dt,
        new_df=eval_df,
        run_dir=run_dir,
        thresholds=drift_th,
    )

    print("\nRunning evaluation...")

    required_baseline_cols = ["soc_pct", "soc_pct_roll_mean_60s", "trip_id"]
    missing = [c for c in required_baseline_cols if c not in eval_df.columns]
    if missing:
        raise ValueError(f"Missing required eval columns: {missing}")

    eval_df = eval_df.copy()
    eval_df["pred"] = np.clip(
        model.predict(eval_df[saved_features].astype("float32")),
        0,
        100,
    )
    eval_df["error"] = eval_df["pred"] - eval_df[saved_target]

    val_metrics = regression_metrics(
        eval_df[saved_target].values,
        eval_df["pred"].values,
        f"Val ({dt})",
    )
    baseline_pers = persistence_baseline(
        eval_df[saved_target].values,
        eval_df["soc_pct"].values,
    )
    baseline_roll = rolling_baseline(
        eval_df[saved_target].values,
        eval_df["soc_pct_roll_mean_60s"].values,
    )

    print(f"\n  Baselines:  persistence={baseline_pers:.3f}  rolling={baseline_roll:.3f}")

    per_trip = per_group_regression(eval_df, saved_target, "pred", "trip_id")
    buckets = soc_bucket_metrics(eval_df, saved_target, "pred")
    err_dist = error_distribution(eval_df["error"])

    print(f"\n  Per-trip MAE ({len(per_trip)} trips):")
    print(per_trip.to_string(index=False))

    eval_report = {
        "run_id": run_dir.name,
        "eval_dt": dt,
        "task": "soc_forecast",
        "val_metrics": val_metrics,
        "baselines": {
            "persistence_mae": round(baseline_pers, 3),
            "rolling_mae": round(baseline_roll, 3),
        },
        "per_trip": per_trip.to_dict(orient="records"),
        "soc_buckets": buckets,
        "error_distribution": err_dist,
        "drift_status": drift_report["overall_status"],
    }

    _write_eval_report(dt, eval_report, drift_report)


def _write_eval_report(dt: str, eval_report: dict, drift_report: dict):
    import json
    try:
        from configs.settings import DATA_DIR
        out_dir = Path(DATA_DIR) / "drift_reports"
    except Exception:
        out_dir = Path("data/drift_reports")

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"dt={dt}_eval.json"
    combined = {
        "eval": eval_report,
        "drift": drift_report,
    }
    out.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\n  Eval + drift report → {out}")


# ══════════════════════════════════════════════════════════════════════════════
#  DRIFT
# ══════════════════════════════════════════════════════════════════════════════

def cmd_drift(args):
    task = args.task
    dt = args.dt
    run_id = getattr(args, "run_id", None)

    print(f"\n── Drift  task={task}  dt={dt} ────────────────────────────")

    import yaml
    from training.artifacts import load_latest_run, load_run_by_id
    from training.dataset import load_gold_dates
    from monitoring.drift import compute_drift_report

    CONFIG_PATH = Path(f"tasks/{task}/config.yaml")
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    vehicles = cfg["data"]["vehicles"]
    drop_mask = int(cfg["data"].get("drop_quality_mask", 52))
    drift_th = cfg.get("drift", {})

    if run_id:
        _, feature_set, run_dir = load_run_by_id(task, run_id)
    else:
        _, feature_set, run_dir = load_latest_run(task)

    saved_features = feature_set["features"]
    saved_target = feature_set["target"]

    new_df = load_gold_dates(
        dates=[dt],
        vehicles=vehicles,
        features=saved_features,
        target=saved_target,
        drop_quality_mask=drop_mask,
    )

    if new_df.empty:
        raise RuntimeError(f"No drift rows available for dt={dt}")

    compute_drift_report(dt=dt, new_df=new_df, run_dir=run_dir, thresholds=drift_th)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _unknown_task(task: str):
    print(f"\nERROR: Unknown task '{task}'.")
    print("To add a new task, create tasks/{task}/train.py and wire it here.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="EV Telemetry ML pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py ingest  --dt 2026-03-01
  python run.py ingest  --backfill --start 2026-01-01 --end 2026-03-01
  python run.py gold    --dt 2026-03-01
  python run.py train   --task soc_forecast
  python run.py eval    --task soc_forecast --dt 2026-03-05
  python run.py eval    --task soc_forecast --dt 2026-03-05 --run-id v2__2026-02-26__abc12345
  python run.py drift   --task soc_forecast --dt 2026-03-05
        """,
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Run ingestion pipeline")
    p_ingest.add_argument("--dt", help="Date YYYY-MM-DD")
    p_ingest.add_argument("--backfill", action="store_true")
    p_ingest.add_argument("--start", help="Backfill start YYYY-MM-DD")
    p_ingest.add_argument("--end", help="Backfill end YYYY-MM-DD")

    p_gold = sub.add_parser("gold", help="Build Gold feature datasets")
    p_gold.add_argument("--dt", required=True, help="Date YYYY-MM-DD")

    p_train = sub.add_parser("train", help="Train a model (full retrain + new run folder)")
    p_train.add_argument("--task", required=True, help="Task name, e.g. soc_forecast")

    p_eval = sub.add_parser("eval", help="Evaluate saved model on a new day (no retraining)")
    p_eval.add_argument("--task", required=True)
    p_eval.add_argument("--dt", required=True, help="Date to evaluate YYYY-MM-DD")
    p_eval.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Pin specific run ID (default: latest)",
    )

    p_drift = sub.add_parser("drift", help="Run drift check only")
    p_drift.add_argument("--task", required=True)
    p_drift.add_argument("--dt", required=True)
    p_drift.add_argument("--run-id", dest="run_id", default=None)

    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "gold": cmd_gold,
        "train": cmd_train,
        "eval": cmd_eval,
        "drift": cmd_drift,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()