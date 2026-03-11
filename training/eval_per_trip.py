# eval_per_trip.py

import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

DATE = "2026-02-23"
VEHICLE = "EV01"
TARGET = "y_soc_t_plus_300s"

VAL_TRIPS = 20


# ─────────────────────────────────────────────
# Feature Set
# ─────────────────────────────────────────────

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


# Charging detection features

CHARGING_FEATURES = [

    "is_charging_current",
    "is_parked_charging"

]


FEATURES = BASE_FEATURES + CHARGING_FEATURES


# Columns to read from Gold

LOAD_COLS = list(dict.fromkeys(

    BASE_FEATURES +
    [TARGET,
     "label_available",
     "trip_id",
     "timestamp",
     "quality_flag",

     "battery_current_a",
     "output_power_kw",
     "stack_voltage_v",
     "motor_speed_rpm"]

))


# ─────────────────────────────────────────────
# Load Gold Data
# ─────────────────────────────────────────────

def load_gold(date):

    base = Path(f"data/gold/window_features/dt={date}/vehicle_id={VEHICLE}")

    parts = sorted(base.glob("*.parquet"))

    if not parts:
        raise FileNotFoundError(base)

    df = pd.concat(
        [pd.read_parquet(p, columns=LOAD_COLS) for p in parts],
        ignore_index=True
    )

    print(f"\nRaw rows: {len(df):,}")

    # Keep labeled rows
    df = df[df["label_available"] == 1]

    print(f"Labeled rows: {len(df):,}")

    # Drop GAP_INSERTED rows
    df = df[(df["quality_flag"] & 4) == 0]

    print(f"After GAP filter: {len(df):,}")

    # Charging features

    df["is_charging_current"] = (
        df["battery_current_a"] > 0
    ).astype("int8")

    df["is_parked_charging"] = (
        (df["battery_current_a"] > 5) &
        (df["motor_speed_rpm"].abs() < 50)
    ).astype("int8")

    # Drop NaNs

    df = df.dropna(subset=BASE_FEATURES + [TARGET])

    print(f"Final usable rows: {len(df):,}")
    print(f"Trips: {df.trip_id.nunique()}")

    return df


# ─────────────────────────────────────────────
# Split by Trip Chronologically
# ─────────────────────────────────────────────

def split_trips(df):

    trip_start = (
        df.groupby("trip_id")["timestamp"]
        .min()
        .sort_values()
    )

    val_ids = trip_start.tail(VAL_TRIPS).index
    train_ids = trip_start.head(len(trip_start)-VAL_TRIPS).index

    train_df = df[df.trip_id.isin(train_ids)].copy()
    val_df   = df[df.trip_id.isin(val_ids)].copy()

    print("\nSplit Summary")

    print(f"Train trips: {len(train_ids)}")
    print(f"Val trips: {len(val_ids)}")

    print("\nTrain range:")
    print(trip_start.loc[train_ids].min(),
          "→",
          trip_start.loc[train_ids].max())

    print("\nVal range:")
    print(trip_start.loc[val_ids].min(),
          "→",
          trip_start.loc[val_ids].max())

    return train_df, val_df


# ─────────────────────────────────────────────
# Load + Split
# ─────────────────────────────────────────────

print("\nLoading data...")

df = load_gold(DATE)

train_df, val_df = split_trips(df)


X_train = train_df[FEATURES].astype("float32")
y_train = train_df[TARGET].astype("float32")

X_val = val_df[FEATURES].astype("float32")
y_val = val_df[TARGET].astype("float32")


# ─────────────────────────────────────────────
# Train Model
# ─────────────────────────────────────────────

print("\nTraining model...\n")

model = xgb.XGBRegressor(

    n_estimators=800,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,

    early_stopping_rounds=30,
    eval_metric="mae",

    n_jobs=-1,
    random_state=42
)

model.fit(
    X_train,
    y_train,
    eval_set=[(X_val,y_val)],
    verbose=50
)


# ─────────────────────────────────────────────
# Predictions
# ─────────────────────────────────────────────

pred_val = np.clip(
    model.predict(X_val),
    0,
    100
)

val_df = val_df.copy()

val_df["pred"] = pred_val
val_df["error"] = pred_val - val_df[TARGET]


# ─────────────────────────────────────────────
# Baselines
# ─────────────────────────────────────────────

print("\nBaselines")

print("Persistence MAE:",
mean_absolute_error(
    y_val,
    val_df["soc_pct"]
))

print("Rolling MAE:",
mean_absolute_error(
    y_val,
    val_df["soc_pct_roll_mean_60s"]
))


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

print("\nMetrics")

print("Train MAE:",
mean_absolute_error(
    y_train,
    np.clip(model.predict(X_train),0,100)
))

print("Val MAE:",
mean_absolute_error(
    y_val,
    pred_val
))


# ─────────────────────────────────────────────
# Per Trip
# ─────────────────────────────────────────────

print("\nPer Trip MAE\n")

for trip,grp in val_df.groupby("trip_id"):

    mae = mean_absolute_error(grp[TARGET],grp["pred"])

    print(
        trip,
        "rows=",len(grp),
        "MAE=",round(mae,3)
    )


# ─────────────────────────────────────────────
# Error Distribution
# ─────────────────────────────────────────────

print("\nError Distribution")

print(val_df["error"].describe().round(3))


# ─────────────────────────────────────────────
# Worst Rows
# ─────────────────────────────────────────────

print("\nWorst Rows\n")

worst = val_df["error"].abs().sort_values(
ascending=False).head(10).index

print(val_df.loc[worst][[

"timestamp",
"trip_id",
"soc_pct",
TARGET,
"pred",
"error",
"battery_current_a",
"output_power_kw",
"stack_voltage_v",
"quality_flag",
"is_charging_current",
"is_parked_charging"

]].sort_values("timestamp").to_string(index=False))