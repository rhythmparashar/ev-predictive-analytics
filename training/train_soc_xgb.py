# train_soc_xgb.py
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────

TRAIN_DATES = [
    "2026-02-23",
    "2026-02-24",
    "2026-02-25",
    "2026-02-26",
]

VAL_DATES = [
    "2026-03-01",
]

TARGET     = "y_soc_t_plus_300s"
VEHICLE_ID = "EV01"

MODEL_OUT  = Path("models/soc_xgb_v2.json")
REPORT_OUT = Path("models/soc_xgb_v2_eval.json")

MODEL_OUT.parent.mkdir(exist_ok=True)

# Micro-trip thresholds (evaluation only)
MIN_TRIP_ROWS      = 60
MIN_TRIP_SOC_RANGE = 3.0


# ───────────────────────────────────────────────
# FEATURES (59)
# ───────────────────────────────────────────────

FEATURES = [

    # SOC
    "soc_pct","soc_pct_lag_60s","soc_pct_lag_300s","soc_pct_lag_600s",
    "soc_pct_roll_mean_60s","soc_pct_roll_mean_300s",
    "soc_pct_roll_std_60s","soc_pct_roll_std_300s",

    # Current
    "battery_current_a","battery_current_a_lag_60s","battery_current_a_lag_300s",
    "battery_current_a_roll_mean_30s","battery_current_a_roll_mean_300s",
    "battery_current_a_roll_std_60s","battery_current_a_roll_max_60s",
    "battery_current_a_roll_min_60s",

    # Voltage
    "stack_voltage_v","stack_voltage_v_lag_60s","stack_voltage_v_lag_300s",
    "stack_voltage_v_roll_mean_60s","stack_voltage_v_roll_mean_300s",
    "stack_voltage_v_roll_std_60s",

    # Cell
    "max_cell_voltage_v","min_cell_voltage_v",
    "cell_voltage_delta_v","cell_voltage_delta_norm",
    "max_cell_voltage_v_roll_mean_60s",
    "min_cell_voltage_v_roll_mean_60s",

    # Power
    "output_power_kw","output_power_kw_lag_60s","output_power_kw_lag_300s",
    "output_power_kw_roll_mean_60s","output_power_kw_roll_mean_300s",
    "output_power_kw_roll_std_60s","output_power_kw_roll_max_60s",
    "elec_power_kw_proxy",

    # Motor speed
    "motor_speed_rpm","motor_speed_rpm_lag_60s",
    "motor_speed_rpm_roll_mean_60s","motor_speed_rpm_roll_mean_300s",
    "motor_speed_rpm_roll_std_60s","motor_speed_rpm_roll_max_60s",

    # Battery temp
    "avg_battery_temp_c","avg_battery_temp_c_lag_60s","avg_battery_temp_c_lag_300s",
    "avg_battery_temp_c_roll_mean_300s","avg_battery_temp_c_roll_std_300s",
    "battery_temp_delta_c","battery_temp_delta_norm",

    # Motor temp
    "motor_temperature_c","motor_temperature_c_lag_300s",
    "motor_temperature_c_roll_mean_300s","motor_temperature_c_roll_std_60s",

    # Energy
    "total_kwh_consumed","total_kwh_consumed_lag_60s",
    "total_kwh_consumed_lag_300s",

    # Fault
    "fault_any",
    "power_proxy_error_kw",
    "power_proxy_ratio",
]


# ───────────────────────────────────────────────
# LOAD GOLD DATA
# ───────────────────────────────────────────────

def load_gold(dates,label):

    frames=[]

    for date in dates:

        base=Path(f"data/gold/window_features/dt={date}/vehicle_id={VEHICLE_ID}")

        files=list(base.glob("*.parquet"))

        if not files:
            print(f"  WARNING: no gold files for {date}")
            continue


        df=pd.concat([
            pd.read_parquet(
                f,
                columns=FEATURES + [TARGET,"label_available","trip_id"]
            )
            for f in files
        ])


        df=df[df["label_available"]==1]

        df=df.dropna(subset=FEATURES+[TARGET])


        print(f"  {date}: {len(df):,} rows, {df.trip_id.nunique()} trips")

        frames.append(df)


    combined=pd.concat(frames).reset_index(drop=True)


    print("  -------------------------------------------")
    print(f"  {label}: {len(combined):,} rows, {combined.trip_id.nunique()} trips")

    return combined



# ───────────────────────────────────────────────
# MICRO TRIP FILTER
# ───────────────────────────────────────────────

def filter_micro_trips(df,label):

    # ensure soc_pct is Series
    soc=df["soc_pct"]

    if isinstance(soc,pd.DataFrame):
        soc=soc.iloc[:,0]


    rows=df.groupby("trip_id").size()

    soc_range=(
        soc.groupby(df["trip_id"]).max()
        -
        soc.groupby(df["trip_id"]).min()
    )


    stats=pd.concat([rows,soc_range],axis=1)

    stats.columns=["rows","soc_range"]


    valid=stats[
        (stats.rows>=MIN_TRIP_ROWS)
        &
        (stats.soc_range>=MIN_TRIP_SOC_RANGE)
    ].index


    skipped=stats.loc[~stats.index.isin(valid)]


    if len(skipped)>0:

        print(f"\nMicro trips skipped ({len(skipped)}):")

        for tid,row in skipped.iterrows():

            print(
                f"  {tid} rows={int(row.rows)} "
                f"soc_range={row.soc_range:.1f}%"
            )


    filtered=df[df.trip_id.isin(valid)]


    print(
        f"{label} after filter: "
        f"{len(filtered):,} rows, "
        f"{filtered.trip_id.nunique()} trips"
    )


    return filtered



# ───────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────

print("\nLoading train...")

train_df=load_gold(TRAIN_DATES,"TRAIN")


print("\nLoading val...")

val_raw=load_gold(VAL_DATES,"VAL")


print("\nFiltering micro trips...")

val_df=filter_micro_trips(val_raw,"VAL")


X_train=train_df[FEATURES].astype("float32")
y_train=train_df[TARGET].astype("float32")

X_val=val_df[FEATURES].astype("float32")
y_val=val_df[TARGET].astype("float32")



# ───────────────────────────────────────────────
# TRAIN
# ───────────────────────────────────────────────

print("\nTraining model...")

model=xgb.XGBRegressor(

    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,

    subsample=0.8,
    colsample_bytree=0.8,

    min_child_weight=10,

    reg_alpha=0.1,
    reg_lambda=1.0,

    n_jobs=-1,
    random_state=42,

    early_stopping_rounds=30,
    eval_metric="mae",
)


model.fit(
    X_train,
    y_train,
    eval_set=[(X_val,y_val)],
    verbose=50
)



# ───────────────────────────────────────────────
# METRICS
# ───────────────────────────────────────────────

pred_train=model.predict(X_train)
pred_val=model.predict(X_val)


def metrics(y_true,y_pred,label):

    mae=mean_absolute_error(y_true,y_pred)

    rmse=np.sqrt(mean_squared_error(y_true,y_pred))


    print(f"\n{label}")

    print("MAE:",round(mae,3))
    print("RMSE:",round(rmse,3))


    return dict(mae=float(mae),rmse=float(rmse))


train_metrics=metrics(y_train,pred_train,"TRAIN")

val_metrics=metrics(y_val,pred_val,"VAL")


print(
    "\nTrain-Val gap:",
    round(val_metrics["mae"]-train_metrics["mae"],3)
)



# ───────────────────────────────────────────────
# FEATURE IMPORTANCE
# ───────────────────────────────────────────────

imp=pd.Series(

    model.feature_importances_,
    index=FEATURES

).sort_values(ascending=False)


print("\nTop features")

print(imp.head(10))



# ───────────────────────────────────────────────
# SAVE
# ───────────────────────────────────────────────

model.save_model(MODEL_OUT)


report=dict(

    train_dates=TRAIN_DATES,
    val_dates=VAL_DATES,

    train_rows=len(X_train),
    val_rows=len(X_val),

    train_metrics=train_metrics,
    val_metrics=val_metrics,

    top_features=imp.head(20).to_dict()
)


REPORT_OUT.write_text(json.dumps(report,indent=2))


print("\nSaved model →",MODEL_OUT)
print("Saved report →",REPORT_OUT)