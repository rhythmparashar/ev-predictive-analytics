from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import xgboost as xgb


# ── Run ID ─────────────────────────────────────────────────────────────────────

def _next_version(task_model_dir: Path) -> int:
    """Find the next version number by scanning existing run folders."""
    existing = [d for d in task_model_dir.glob("v*__*__*") if d.is_dir()]
    if not existing:
        return 1

    versions = []
    for d in existing:
        try:
            versions.append(int(d.name.split("__")[0][1:]))
        except (ValueError, IndexError):
            pass

    return max(versions, default=0) + 1


def make_run_id(task: str, train_end_date: str) -> str:
    """Generate a unique run ID: v{n}__{date}__{hash8}"""
    ts_hash = hashlib.sha256(
        f"{task}{train_end_date}{time.time()}".encode()
    ).hexdigest()[:8]

    models_dir = _models_root() / task
    models_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(models_dir)

    return f"v{version}__{train_end_date}__{ts_hash}"


def _models_root() -> Path:
    try:
        from configs.settings import MODELS_DIR
        return Path(MODELS_DIR)
    except Exception:
        return Path("models")


# ── Save run ───────────────────────────────────────────────────────────────────

def save_run(
    task: str,
    run_id: str,
    model: xgb.XGBRegressor,
    features: List[str],
    target: str,
    train_dates: List[str],
    val_dates: List[str],
    train_rows: int,
    val_rows: int,
    train_trips: int,
    val_trips: int,
    parquet_hashes: Dict[str, str],
    eval_report: Dict[str, Any],
    train_df: pd.DataFrame,
    config_path: Path,
) -> Path:
    """
    Write all run artifacts to models/<task>/<run_id>/ using tmp dir -> rename.
    This avoids leaving a partial run folder on failure.
    """
    task_dir = _models_root() / task
    task_dir.mkdir(parents=True, exist_ok=True)

    run_dir_final = task_dir / run_id
    if run_dir_final.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir_final}")

    with tempfile.TemporaryDirectory(dir=task_dir) as tmp:
        tmp_root = Path(tmp)
        tmp_dir = tmp_root / run_id
        tmp_dir.mkdir()

        # 1. Model
        model.save_model(tmp_dir / "model.json")

        # 2. Feature set snapshot
        (tmp_dir / "feature_set.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "target": target,
                    "features": features,
                    "n_features": len(features),
                },
                indent=2,
                default=str,
            )
        )

        # 3. Data fingerprint
        fingerprint = {
            "task": task,
            "run_id": run_id,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "train_dates": train_dates,
            "val_dates": val_dates,
            "train_rows": train_rows,
            "val_rows": val_rows,
            "train_trips": train_trips,
            "val_trips": val_trips,
            "parquet_hashes": parquet_hashes,
        }
        (tmp_dir / "data_fingerprint.json").write_text(
            json.dumps(fingerprint, indent=2, default=str)
        )

        # 4. Eval report
        (tmp_dir / "eval_report.json").write_text(
            json.dumps(eval_report, indent=2, default=str)
        )

        # 5. Drift baseline
        baseline = _compute_baseline(train_df, features)
        baseline.to_parquet(tmp_dir / "drift_baseline.parquet", index=False)

        # 6. Config snapshot
        if config_path.exists():
            shutil.copy(config_path, tmp_dir / "config.yaml")

        # Atomic-ish commit within same filesystem
        tmp_dir.replace(run_dir_final)

    print(f"\n  Run saved → {run_dir_final}")
    return run_dir_final


def _compute_baseline(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Per-feature distribution stats from training data."""
    rows = []

    for col in features:
        if col not in df.columns:
            continue

        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()

        if len(non_null) == 0:
            rows.append(
                {
                    "feature": col,
                    "mean": None,
                    "std": None,
                    "p10": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "p90": None,
                    "null_rate": float(s.isna().mean()),
                    "n": 0,
                }
            )
            continue

        rows.append(
            {
                "feature": col,
                "mean": float(non_null.mean()),
                "std": float(non_null.std()),
                "p10": float(non_null.quantile(0.10)),
                "p25": float(non_null.quantile(0.25)),
                "p50": float(non_null.quantile(0.50)),
                "p75": float(non_null.quantile(0.75)),
                "p90": float(non_null.quantile(0.90)),
                "null_rate": float(s.isna().mean()),
                "n": int(len(non_null)),
            }
        )

    return pd.DataFrame(rows)


# ── Load latest model ──────────────────────────────────────────────────────────

def load_latest_run(task: str) -> tuple[xgb.XGBRegressor, dict, Path]:
    """
    Load the most recent run for a task.
    Returns (model, feature_set_dict, run_dir).
    """
    task_dir = _models_root() / task

    runs = sorted(
        [d for d in task_dir.glob("v*__*__*") if d.is_dir()],
        key=lambda d: int(d.name.split("__")[0][1:]),
    )

    if not runs:
        raise FileNotFoundError(f"No model runs found for task '{task}' in {task_dir}")

    run_dir = runs[-1]
    print(f"  Loading run: {run_dir.name}")

    model = xgb.XGBRegressor()
    model.load_model(run_dir / "model.json")

    feature_set = json.loads((run_dir / "feature_set.json").read_text())

    return model, feature_set, run_dir


def load_run_by_id(task: str, run_id: str) -> tuple[xgb.XGBRegressor, dict, Path]:
    """Load a specific run by run_id."""
    run_dir = _models_root() / task / run_id

    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")

    model = xgb.XGBRegressor()
    model.load_model(run_dir / "model.json")
    feature_set = json.loads((run_dir / "feature_set.json").read_text())

    return model, feature_set, run_dir