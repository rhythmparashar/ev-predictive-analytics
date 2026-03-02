# scripts/eval_soc.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml


def load_cfg(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"mae": mae, "rmse": rmse}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True)
    ap.add_argument("--vehicle_id", required=True)
    ap.add_argument("--cfg", default="configs/train_soc.yaml")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.cfg))
    gold_dir = cfg["data"]["gold_dir"]
    target_col = cfg["data"]["target_col"]
    label_col = cfg["data"]["label_col"]

    model_dir = Path(cfg["outputs"]["model_dir"])
    model = joblib.load(model_dir / "model.pkl")
    features = json.loads((model_dir / "feature_list.json").read_text())

    p = Path(gold_dir) / "window_features" / f"dt={args.dt}" / f"vehicle_id={args.vehicle_id}.parquet"
    df = pd.read_parquet(p)

    df = df[(df[label_col] == 1) & df["trip_id"].notna()].copy()

    X = df[features]
    y = df[target_col].astype(float).to_numpy()

    yhat = model.predict(X)
    print("Eval metrics:", metrics(y, yhat))


if __name__ == "__main__":
    main()