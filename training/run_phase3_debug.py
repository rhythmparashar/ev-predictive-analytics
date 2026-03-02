import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────────────────────
# CONFIG (edit these 3 if needed)
# ─────────────────────────────────────────────────────────────
DATE = "2026-02-23"
VEHICLE = "EV01"
TARGET = "y_soc_t_plus_300s"
VAL_TRIPS = 20

OUT_DIR = Path("models/phase3_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# FEATURES (your pruned set) + debug-friendly additions
# ─────────────────────────────────────────────────────────────
BASE_FEATURES = [
    "soc_pct","soc_pct_lag_60s","soc_pct_lag_300s","soc_pct_lag_600s",
    "soc_pct_roll_mean_60s","soc_pct_roll_mean_300s",
    "soc_pct_roll_std_60s","soc_pct_roll_std_300s",

    "battery_current_a","battery_current_a_lag_60s","battery_current_a_lag_300s",
    "battery_current_a_roll_mean_30s","battery_current_a_roll_mean_300s",
    "battery_current_a_roll_std_60s","battery_current_a_roll_max_60s",
    "battery_current_a_roll_min_60s",

    "stack_voltage_v","stack_voltage_v_lag_60s","stack_voltage_v_lag_300s",
    "stack_voltage_v_roll_mean_60s","stack_voltage_v_roll_mean_300s",
    "stack_voltage_v_roll_std_60s",

    "max_cell_voltage_v","min_cell_voltage_v",
    "cell_voltage_delta_v","cell_voltage_delta_norm",
    "max_cell_voltage_v_roll_mean_60s","min_cell_voltage_v_roll_mean_60s",

    "output_power_kw","output_power_kw_lag_60s","output_power_kw_lag_300s",
    "output_power_kw_roll_mean_60s","output_power_kw_roll_mean_300s",
    "output_power_kw_roll_std_60s","output_power_kw_roll_max_60s",
    "elec_power_kw_proxy",

    "motor_speed_rpm","motor_speed_rpm_lag_60s",
    "motor_speed_rpm_roll_mean_60s","motor_speed_rpm_roll_mean_300s",
    "motor_speed_rpm_roll_std_60s","motor_speed_rpm_roll_max_60s",

    "avg_battery_temp_c","avg_battery_temp_c_lag_60s","avg_battery_temp_c_lag_300s",
    "avg_battery_temp_c_roll_mean_300s","avg_battery_temp_c_roll_std_300s",
    "battery_temp_delta_c","battery_temp_delta_norm",

    "motor_temperature_c","motor_temperature_c_lag_300s",
    "motor_temperature_c_roll_mean_300s","motor_temperature_c_roll_std_60s",

    "total_kwh_consumed","total_kwh_consumed_lag_60s","total_kwh_consumed_lag_300s",

    "fault_any","power_proxy_error_kw","power_proxy_ratio",
]

# model-time derived features (help debug charging / parked behavior)
DERIVED_FEATURES = [
    "is_charging_current",
    "is_parked_charging",
    "is_physics_quiet",
]

FEATURES = BASE_FEATURES + DERIVED_FEATURES


# Columns we must load from Gold to compute derived debug features
LOAD_COLS = list(dict.fromkeys(
    BASE_FEATURES + [
        TARGET, "label_available", "trip_id", "timestamp", "quality_flag",
        # required for derived features + debugging
        "battery_current_a", "output_power_kw", "motor_speed_rpm", "stack_voltage_v",
    ]
))


# ─────────────────────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────────────────────
def load_gold_one_date(date: str) -> pd.DataFrame:
    base = Path(f"data/gold/window_features/dt={date}/vehicle_id={VEHICLE}")
    parts = sorted(base.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No parquet parts found under: {base}")

    df = pd.concat([pd.read_parquet(p, columns=LOAD_COLS) for p in parts], ignore_index=True)

    print(f"\nRaw rows: {len(df):,}")
    df = df[df["label_available"] == 1].copy()
    print(f"Labeled rows: {len(df):,}")

    # Drop GAP_INSERTED
    df = df[(df["quality_flag"] & 4) == 0].copy()
    print(f"After GAP filter: {len(df):,}")

    # Derived / debug features
    df["is_charging_current"] = (df["battery_current_a"] > 0).astype("int8")
    df["is_parked_charging"] = ((df["battery_current_a"] > 5) & (df["motor_speed_rpm"].abs() < 50)).astype("int8")
    df["is_physics_quiet"] = (
        (df["motor_speed_rpm"].abs() < 50) &
        (df["output_power_kw"].abs() < 0.2) &
        (df["battery_current_a"].abs() < 1.0)
    ).astype("int8")
    df["delta_soc_5m"] = (df[TARGET] - df["soc_pct"]).astype("float32")

    # Drop NaNs for modeling
    df = df.dropna(subset=BASE_FEATURES + [TARGET]).copy()
    print(f"Final usable rows: {len(df):,}")
    print(f"Trips: {df.trip_id.nunique()}")

    # stable ordering
    df = df.sort_values(["trip_id", "timestamp"], kind="mergesort")
    return df


def split_train_val_by_trip(df: pd.DataFrame, val_trips: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    trip_start = df.groupby("trip_id")["timestamp"].min().sort_values()
    if len(trip_start) <= val_trips:
        raise ValueError(f"Not enough trips ({len(trip_start)}) for val_trips={val_trips}")

    val_ids = trip_start.tail(val_trips).index
    train_ids = trip_start.head(len(trip_start) - val_trips).index

    train_df = df[df["trip_id"].isin(train_ids)].copy()
    val_df = df[df["trip_id"].isin(val_ids)].copy()

    print(f"\nSplit: train_trips={len(train_ids)}, val_trips={len(val_ids)}")
    print(f"Train trip-start range: {trip_start.loc[train_ids].min()} → {trip_start.loc[train_ids].max()}")
    print(f"Val   trip-start range: {trip_start.loc[val_ids].min()} → {trip_start.loc[val_ids].max()}")

    return train_df, val_df


# ─────────────────────────────────────────────────────────────
# DATA DIAGNOSTICS: constant/near-constant columns
# ─────────────────────────────────────────────────────────────
def constant_feature_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        # ignore NaNs
        sn = s.dropna()
        if len(sn) == 0:
            rows.append((c, "all_nan", np.nan, np.nan, np.nan, np.nan, 1.0))
            continue
        nun = int(sn.nunique())
        vmin = float(sn.min())
        vmax = float(sn.max())
        mean = float(sn.mean()) if pd.api.types.is_numeric_dtype(sn) else np.nan
        std = float(sn.std()) if pd.api.types.is_numeric_dtype(sn) else np.nan
        zero_pct = float((sn == 0).mean()) if pd.api.types.is_numeric_dtype(sn) else np.nan
        rows.append((c, "ok", nun, vmin, vmax, mean, std, zero_pct))

    rep = pd.DataFrame(rows, columns=["col", "status", "n_unique", "min", "max", "mean", "std", "zero_pct"])
    rep["is_constant"] = rep["n_unique"] == 1
    rep["is_near_constant"] = (rep["std"].fillna(0) < 1e-6) & (rep["n_unique"].fillna(0) <= 2)
    rep = rep.sort_values(["is_constant", "is_near_constant", "zero_pct"], ascending=[False, False, False])
    return rep


# ─────────────────────────────────────────────────────────────
# TRAIN + EVAL
# ─────────────────────────────────────────────────────────────
def metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


def main():
    print("Loading gold...")
    df = load_gold_one_date(DATE)

    # diagnostics on BASE features only (derived features are expected to be low-var sometimes)
    rep = constant_feature_report(df, BASE_FEATURES)
    rep_path = OUT_DIR / "feature_constant_report.csv"
    rep.to_csv(rep_path, index=False)
    print(f"\nWrote: {rep_path}")

    print("\nTop constant/near-constant columns:")
    print(rep.head(20).to_string(index=False))

    # split
    train_df, val_df = split_train_val_by_trip(df, VAL_TRIPS)

    # Baselines
    y_val = val_df[TARGET].to_numpy(dtype="float32")
    persist_pred = np.clip(val_df["soc_pct"].to_numpy(dtype="float32"), 0.0, 100.0)
    roll_pred = np.clip(val_df["soc_pct_roll_mean_60s"].to_numpy(dtype="float32"), 0.0, 100.0)

    baseline = {
        "persistence": metrics(y_val, persist_pred),
        "rolling_60s": metrics(y_val, roll_pred),
    }

    # Train model
    X_train = train_df[FEATURES].astype("float32")
    y_train = train_df[TARGET].astype("float32")

    X_val = val_df[FEATURES].astype("float32")
    y_val_s = val_df[TARGET].astype("float32")

    print(f"\nTraining XGBoost: train_rows={len(X_train):,}, val_rows={len(X_val):,}, features={len(FEATURES)}")

    model = xgb.XGBRegressor(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=15,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        early_stopping_rounds=50,
        eval_metric="mae",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val_s)], verbose=50)

    # Predict
    pred_train = np.clip(model.predict(X_train), 0.0, 100.0)
    pred_val = np.clip(model.predict(X_val), 0.0, 100.0)

    train_m = metrics(y_train.to_numpy(), pred_train)
    val_m = metrics(y_val_s.to_numpy(), pred_val)

    # attach predictions for debugging + plotting
    val_dbg = val_df[["timestamp", "trip_id", "soc_pct", TARGET, "battery_current_a", "output_power_kw", "motor_speed_rpm", "quality_flag"]].copy()
    val_dbg["pred"] = pred_val
    val_dbg["error"] = val_dbg["pred"] - val_dbg[TARGET]
    val_dbg["abs_error"] = val_dbg["error"].abs()

    dbg_path = OUT_DIR / "val_predictions_debug.csv"
    val_dbg.to_csv(dbg_path, index=False)
    print(f"\nWrote: {dbg_path}")

    # per-trip MAE
    per_trip = (
        val_dbg.groupby("trip_id")
        .apply(lambda g: mean_absolute_error(g[TARGET], g["pred"]))
        .sort_values(ascending=False)
    )
    per_trip_path = OUT_DIR / "val_per_trip_mae.csv"
    per_trip.to_csv(per_trip_path, header=["mae"])
    print(f"Wrote: {per_trip_path}")

    worst_rows = val_dbg.sort_values("abs_error", ascending=False).head(15)

    # feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    imp_path = OUT_DIR / "feature_importance.csv"
    imp.to_csv(imp_path, header=["importance"])
    print(f"Wrote: {imp_path}")

    # report JSON (founder-friendly)
    report = {
        "date_partition": DATE,
        "vehicle": VEHICLE,
        "target": TARGET,
        "val_trips": int(per_trip.shape[0]),
        "rows": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
        },
        "baselines": baseline,
        "model_metrics": {
            "train": train_m,
            "val": val_m,
            "best_iteration": int(getattr(model, "best_iteration", -1)),
        },
        "top_features": imp.head(20).to_dict(),
        "worst_trips_mae": per_trip.head(10).to_dict(),
        "worst_rows": worst_rows.to_dict(orient="records"),
        "notes": [
            "feature_constant_report.csv lists constant/near-constant signals (often indicates sensor stuck, wrong mapping, or fill artifacts).",
            "val_predictions_debug.csv contains timestamp-level predictions + key signals to plot and explain failure modes.",
        ],
    }

    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote: {report_path}")

    # save model
    model_path = OUT_DIR / "xgb_model.ubj"
    model.save_model(model_path)
    print(f"Wrote: {model_path}")

    # console summary
    print("\n=== SUMMARY ===")
    print("Baselines (VAL):", baseline)
    print("Model (TRAIN):", train_m)
    print("Model (VAL):", val_m)
    print("\nTop 10 features:")
    print(imp.head(10).to_string())
    print("\nWorst 10 trips (VAL):")
    print(per_trip.head(10).to_string())
    print("\nWorst 10 rows (VAL):")
    print(worst_rows[["timestamp","trip_id","soc_pct",TARGET,"pred","error","battery_current_a","output_power_kw","motor_speed_rpm","quality_flag"]].to_string(index=False))


if __name__ == "__main__":
    main()