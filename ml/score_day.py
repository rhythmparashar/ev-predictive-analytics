"""
Score one vehicle-day with the latest SOC forecast model.

CLI usage:
    python ml/score_day.py --dt 2026-06-01 --vehicle-id EV03 --machine-type Loader

Importable usage:
    from ml.score_day import score_day
    ok, msg = score_day("2026-06-01", "EV03", "Loader")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from configs.logging_config import get_logger

logger = get_logger(__name__)

MODEL_DIR = PROJECT_ROOT / "models" / "soc_forecast"
GOLD_DIR  = PROJECT_ROOT / "data" / "gold"
OUT_DIR   = PROJECT_ROOT / "outputs" / "soc_scores"

SOC_MODELS = {
    "Loader":    "v4__2026-06-02__fdd89133",
    "Excavator": "v4__2026-06-02__fdd89133",
}
DROP_QUALITY_MASK = 52


def score_day(dt: str, vehicle_id: str, machine_type: str) -> tuple[bool, str]:
    try:
        import numpy as np
        import pandas as pd
        import xgboost as xgb

        run_id = SOC_MODELS.get(machine_type) or SOC_MODELS.get("Loader")
        if not run_id:
            return False, f"no model configured for machine type '{machine_type}'"

        model_path = MODEL_DIR / run_id / "model.json"
        feat_path  = MODEL_DIR / run_id / "feature_set.json"
        if not model_path.exists():
            return False, f"model file not found: {model_path}"

        gold_path = GOLD_DIR / "window_features" / f"dt={dt}" / f"vehicle_id={vehicle_id}"
        parts = list(gold_path.glob("*.parquet")) if gold_path.exists() else []
        if not parts:
            return False, f"no gold features at {gold_path}"

        model    = xgb.XGBRegressor()
        model.load_model(str(model_path))
        features = json.loads(feat_path.read_text())["features"]
        target   = "y_soc_t_plus_300s"

        chunks = []
        for f in parts:
            df = pd.read_parquet(f)
            if "label_available" in df.columns:
                df = df[df["label_available"] == 1]
            if "quality_flag" in df.columns:
                df = df[(df["quality_flag"].astype(int) & DROP_QUALITY_MASK) == 0]
            if "soc_pct" in df.columns:
                df = df[df["soc_pct"].notna()]
            if not df.empty and target in df.columns:
                chunks.append(df)

        if not chunks:
            return False, "no clean rows after quality filter"

        df = pd.concat(chunks, ignore_index=True)
        for col in features:
            if col not in df.columns:
                df[col] = 0.0

        yhat = model.predict(df[features].fillna(0).astype("float32"), validate_features=False)
        mae  = float(np.mean(np.abs(df[target].astype(float).values - yhat)))

        out = OUT_DIR / f"dt={dt}"
        out.mkdir(parents=True, exist_ok=True)
        pred_df = df[["soc_pct", target]].copy() if "soc_pct" in df.columns else df[[target]].copy()
        pred_df["soc_pred_t300s"] = yhat
        pred_df["vehicle_id"]     = vehicle_id
        pred_df["machine_type"]   = machine_type
        pred_df["model_run"]      = run_id
        pred_df.to_parquet(out / f"vehicle_id={vehicle_id}.parquet", index=False)

        return True, f"MAE={mae:.3f}%  n={len(df):,}"

    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt",           required=True)
    ap.add_argument("--vehicle-id",   required=True)
    ap.add_argument("--machine-type", default="Loader")
    args = ap.parse_args()

    ok, msg = score_day(args.dt, args.vehicle_id, args.machine_type)
    level = "info" if ok else "error"
    getattr(logger, level)("[soc] dt=%s vehicle=%s %s: %s", args.dt, args.vehicle_id, "OK" if ok else "FAIL", msg)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
