# tasks/soc_forecast/feature_set.py
#
# SINGLE SOURCE OF TRUTH for the SOC 5-min forecast task.
# Imported by train.py, evaluate.py, and any future serving code.
# Never duplicate this list elsewhere.

TASK    = "soc_forecast"
TARGET  = "y_soc_t_plus_300s"

FEATURES = [
    # SOC state
    "soc_pct",
    "soc_pct_lag_60s",
    "soc_pct_lag_300s",
    "soc_pct_lag_600s",
    "soc_pct_roll_mean_60s",
    "soc_pct_roll_mean_300s",
    "soc_pct_roll_std_60s",
    "soc_pct_roll_std_300s",

    # Battery current
    "battery_current_a",
    "battery_current_a_lag_60s",
    "battery_current_a_lag_300s",
    "battery_current_a_roll_mean_30s",
    "battery_current_a_roll_mean_300s",
    "battery_current_a_roll_std_60s",
    "battery_current_a_roll_max_60s",
    "battery_current_a_roll_min_60s",

    # Stack voltage
    "stack_voltage_v",
    "stack_voltage_v_lag_60s",
    "stack_voltage_v_lag_300s",
    "stack_voltage_v_roll_mean_60s",
    "stack_voltage_v_roll_mean_300s",
    "stack_voltage_v_roll_std_60s",

    # Cell imbalance
    "max_cell_voltage_v",
    "min_cell_voltage_v",
    "cell_voltage_delta_v",
    "cell_voltage_delta_norm",
    "max_cell_voltage_v_roll_mean_60s",
    "min_cell_voltage_v_roll_mean_60s",

    # Output power
    "output_power_kw",
    "output_power_kw_lag_60s",
    "output_power_kw_lag_300s",
    "output_power_kw_roll_mean_60s",
    "output_power_kw_roll_mean_300s",
    "output_power_kw_roll_std_60s",
    "output_power_kw_roll_max_60s",
    "elec_power_kw_proxy",

    # Motor speed
    "motor_speed_rpm",
    "motor_speed_rpm_lag_60s",
    "motor_speed_rpm_roll_mean_60s",
    "motor_speed_rpm_roll_mean_300s",
    "motor_speed_rpm_roll_std_60s",
    "motor_speed_rpm_roll_max_60s",

    # Battery temperature
    "avg_battery_temp_c",
    "avg_battery_temp_c_lag_60s",
    "avg_battery_temp_c_lag_300s",
    "avg_battery_temp_c_roll_mean_300s",
    "avg_battery_temp_c_roll_std_300s",
    "battery_temp_delta_c",
    "battery_temp_delta_norm",

    # Motor temperature
    "motor_temperature_c",
    "motor_temperature_c_lag_300s",
    "motor_temperature_c_roll_mean_300s",
    "motor_temperature_c_roll_std_60s",

    # Energy counters
    "total_kwh_consumed",
    "total_kwh_consumed_lag_60s",
    "total_kwh_consumed_lag_300s",

    # Derived / fault
    "fault_any",
    "power_proxy_error_kw",
    "power_proxy_ratio",

    # Charging context  (engineered in dataset.py from raw signals)
    "is_charging_current",
    "is_parked_charging",
]

# Columns that must exist in Gold — fail fast if absent
REQUIRED_SIGNALS = ["soc_pct", "battery_current_a", "stack_voltage_v"]

# Micro-trip filter thresholds (applied during evaluation, not training)
MIN_TRIP_ROWS      = 60
MIN_TRIP_SOC_RANGE = 3.0
