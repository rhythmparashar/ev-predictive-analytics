from __future__ import annotations

import numpy as np
import pandas as pd


def physics_features(df: pd.DataFrame, defragment: bool = False) -> pd.DataFrame:
    """
    Physics-derived features.

    IMPORTANT:
    - Do NOT copy huge frames unless you must.
    - If defragment=True, returns a copy at end (slower, more RAM).
    """
    out = df
    eps = 1e-9

    if "stack_voltage_v" in out.columns and "battery_current_a" in out.columns:
        out["elec_power_kw_proxy"] = (out["stack_voltage_v"] * out["battery_current_a"]) / 1000.0

    if "output_power_kw" in out.columns and "elec_power_kw_proxy" in out.columns:
        out["power_proxy_error_kw"] = out["output_power_kw"] - out["elec_power_kw_proxy"]
        out["power_proxy_ratio"] = out["output_power_kw"] / (out["elec_power_kw_proxy"].abs() + eps)

    if "max_cell_voltage_v" in out.columns and "min_cell_voltage_v" in out.columns:
        out["cell_voltage_delta_v"] = out["max_cell_voltage_v"] - out["min_cell_voltage_v"]
        if "avg_cell_voltage_v" in out.columns:
            out["cell_voltage_delta_norm"] = out["cell_voltage_delta_v"] / (out["avg_cell_voltage_v"].abs() + eps)

    if "max_battery_temp_c" in out.columns and "min_battery_temp_c" in out.columns:
        out["battery_temp_delta_c"] = out["max_battery_temp_c"] - out["min_battery_temp_c"]
        if "avg_battery_temp_c" in out.columns:
            out["battery_temp_delta_norm"] = out["battery_temp_delta_c"] / (out["avg_battery_temp_c"].abs() + eps)

    torque_col = "motor_torque_value_nm" if "motor_torque_value_nm" in out.columns else None
    if torque_col and "motor_speed_rpm" in out.columns:
        omega = out["motor_speed_rpm"] * (2.0 * np.pi / 60.0)
        out["mech_power_kw_proxy"] = (out[torque_col] * omega) / 1000.0
        if "elec_power_kw_proxy" in out.columns:
            out["eff_proxy"] = out["mech_power_kw_proxy"] / (out["elec_power_kw_proxy"].abs() + eps)

    if "motor_temperature_c" in out.columns and "radiator_temperature_c" in out.columns:
        out["motor_minus_radiator_temp_c"] = out["motor_temperature_c"] - out["radiator_temperature_c"]

    if "mcu_temperature_c" in out.columns and "radiator_temperature_c" in out.columns:
        out["mcu_minus_radiator_temp_c"] = out["mcu_temperature_c"] - out["radiator_temperature_c"]

    if "dcdc_input_voltage_v" in out.columns and "dcdc_input_current_a" in out.columns:
        out["dcdc_input_kw_proxy"] = (out["dcdc_input_voltage_v"] * out["dcdc_input_current_a"]) / 1000.0

    if "dcdc_output_voltage_v" in out.columns and "dcdc_output_current_a" in out.columns:
        out["dcdc_output_kw_proxy"] = (out["dcdc_output_voltage_v"] * out["dcdc_output_current_a"]) / 1000.0

    if "dcdc_input_kw_proxy" in out.columns and "dcdc_output_kw_proxy" in out.columns:
        out["dcdc_eff_proxy"] = out["dcdc_output_kw_proxy"] / (out["dcdc_input_kw_proxy"].abs() + eps)

    return out.copy() if defragment else out