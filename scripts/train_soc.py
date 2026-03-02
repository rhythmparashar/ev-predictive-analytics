# scripts/train_soc.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml


def load_cfg(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f)


def gold_path(gold_dir: str, dt: str, vehicle_id: str) -> Path:
    return Path(gold_dir) / "window_features" / f"dt={dt}" / f"vehicle_id={vehicle_id}.parquet"


def add_trip_position(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    df = df.copy()
    df["pos_in_trip"] = df.groupby(group_col, sort=False).cumcount()
    return df


def choose_features(df: pd.DataFrame, target: str, label: str) -> List[str]:
    # Drop obvious non-features / leakage
    drop_cols = {
        target,
        label,
        "timestamp",
        "vehicle_id",
        "trip_id",
        "__fragment_index",
        "__batch_index",
        "__last_in_fragment",
        "__filename",
        "pos_in_trip",
        "trip_len",
        "frac",
    }

    # numeric-only features (simple + robust)
    feature_cols: List[str] = []
    for c in df.columns:
        if c in drop_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    if len(feature_cols) == 0:
        raise RuntimeError("No numeric feature columns found after filtering.")
    return sorted(feature_cols)


def split_trips(
    trips: List[str], test_frac: float, seed: int, strategy: str
) -> Tuple[List[str], List[str]]:
    trips = list(trips)

    if strategy == "time_order_trips":
        # deterministic: last X% trips as test
        n_test = max(1, int(round(len(trips) * test_frac)))
        train_trips = trips[:-n_test]
        test_trips = trips[-n_test:]
        return train_trips, test_trips

    # default: shuffled trips
    rng = np.random.default_rng(seed)
    rng.shuffle(trips)
    n_test = max(1, int(round(len(trips) * test_frac)))
    test_trips = trips[:n_test]
    train_trips = trips[n_test:]
    return train_trips, test_trips


def metrics_regression(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    # guard against NaNs
    mask = np.isfinite(y) & np.isfinite(yhat)
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    y = y[mask]
    yhat = yhat[mask]

    mae = float(np.mean(np.abs(y - yhat)))
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return {"mae": mae, "rmse": rmse}


def slice_metrics(y: np.ndarray, yhat: np.ndarray, soc_now: np.ndarray) -> Dict[str, Dict[str, float]]:
    # (future - now)
    mask = np.isfinite(y) & np.isfinite(yhat) & np.isfinite(soc_now)
    y = y[mask]
    yhat = yhat[mask]
    soc_now = soc_now[mask]

    delta = y - soc_now
    out: Dict[str, Dict[str, float]] = {}

    for name, m in {
        "charging": delta > 0,
        "discharging": delta < 0,
        "flat": delta == 0,
    }.items():
        if m.sum() == 0:
            out[name] = {"mae": float("nan"), "rmse": float("nan"), "n": 0}
        else:
            mm = metrics_regression(y[m], yhat[m])
            mm["n"] = int(m.sum())
            out[name] = mm

    return out


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
    group_col = cfg["data"]["group_col"]

    min_pos = int(cfg["filtering"]["min_pos_in_trip"])

    test_frac = float(cfg["split"]["test_frac"])
    seed = int(cfg["split"]["seed"])
    strategy = str(cfg["split"]["split_strategy"])

    model_dir = Path(cfg["outputs"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    p = gold_path(gold_dir, args.dt, args.vehicle_id)
    if not p.exists():
        raise FileNotFoundError(f"Gold window_features not found: {p}")

    # --- Load ---
    df = pd.read_parquet(p)

    # --- Validate required columns ---
    for c in [label_col, target_col, group_col, "soc_pct"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # --- Filter rows ---
    df = df[df[group_col].notna()].copy()
    df = add_trip_position(df, group_col=group_col)

    # keep labeled only
    df = df[df[label_col] == 1].copy()

    # drop early trip rows (reduces NaNs for rolling + lag)
    df = df[df["pos_in_trip"] >= min_pos].copy()

    if len(df) == 0:
        raise RuntimeError("No training rows left after filtering.")

    # --- Features ---
    feature_cols = choose_features(df, target=target_col, label=label_col)

    # --- Split by trip_id ---
    trips = df[group_col].astype(str).unique().tolist()
    train_trips, test_trips = split_trips(trips, test_frac=test_frac, seed=seed, strategy=strategy)

    train = df[df[group_col].astype(str).isin(train_trips)].copy()
    test = df[df[group_col].astype(str).isin(test_trips)].copy()

    # --- Drop NaNs needed for baseline + metrics ---
    # (LightGBM can handle NaNs in FEATURES, but baseline/metrics can't.)
    train = train[train["soc_pct"].notna() & train[target_col].notna()].copy()
    test = test[test["soc_pct"].notna() & test[target_col].notna()].copy()

    if len(train) == 0:
        raise RuntimeError("No train rows left after dropping NaNs in soc/target.")
    if len(test) == 0:
        raise RuntimeError("No test rows left after dropping NaNs in soc/target.")

    X_train = train[feature_cols]
    y_train = train[target_col].astype(float).to_numpy()

    X_test = test[feature_cols]
    y_test = test[target_col].astype(float).to_numpy()

    # --- Baseline (persistence) ---
    base_hat = test["soc_pct"].astype(float).to_numpy()
    baseline = metrics_regression(y_test, base_hat)

    # --- Model ---
    try:
        import lightgbm as lgb

        params = dict(cfg["model"]["params"])
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        model_type = "lightgbm"
    except Exception as e:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(random_state=seed)
        model.fit(X_train, y_train)
        yhat = model.predict(X_test)

        model_type = f"sklearn_histgb (fallback because lightgbm failed: {type(e).__name__})"

    model_metrics = metrics_regression(y_test, yhat)

    # slice metrics (charging/discharging/flat)
    slice_m = slice_metrics(y_test, yhat, test["soc_pct"].astype(float).to_numpy())

    # --- Save artifacts ---
    joblib.dump(model, model_dir / "model.pkl")

    (model_dir / "feature_list.json").write_text(json.dumps(feature_cols, indent=2))
    (model_dir / "split.json").write_text(
        json.dumps(
            {
                "dt": args.dt,
                "vehicle_id": args.vehicle_id,
                "strategy": strategy,
                "seed": seed,
                "test_frac": test_frac,
                "n_trips_total": len(trips),
                "n_trips_train": len(train_trips),
                "n_trips_test": len(test_trips),
                "train_trips": train_trips,
                "test_trips": test_trips,
            },
            indent=2,
        )
    )

    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "rows": {"train": int(len(train)), "test": int(len(test))},
                "baseline_persistence": baseline,
                "model": model_metrics,
                "slice_metrics": slice_m,
                "filters": {"label_only": True, "min_pos_in_trip": min_pos, "drop_nan_soc_and_target": True},
            },
            indent=2,
        )
    )

    # --- Print summary ---
    print("=== Phase 3 Training Complete ===")
    print("Gold:", str(p))
    print("Rows train/test:", len(train), len(test))
    print("Trips train/test:", len(train_trips), len(test_trips))
    print("Baseline MAE/RMSE:", baseline["mae"], baseline["rmse"])
    print("Model   MAE/RMSE:", model_metrics["mae"], model_metrics["rmse"])
    print("Saved to:", str(model_dir))


if __name__ == "__main__":
    main()