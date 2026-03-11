"""
Ingest pipeline per day:

- Loads raw CSVs for dt
- Standardizes columns to canonical names
- Validates and parses timestamp + range checks
- Writes raw_parquet
- Resamples to 1Hz silver
- Assigns trip_id
- Adds fault flags
- Writes silver

Returns a manifest dict to write to data/reports/
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from ingestion.io import (
    list_raw_csvs,
    vehicle_id_from_filename,
    read_csv,
    parquet_path,
    fault_csv_path,
    write_parquet_atomic,
)
from ingestion.validators import validate
from ingestion.resampler import resample_1hz
from ingestion.trip_segmentor import add_trip_id
from ingestion.faults import load_fault_csv, add_fault_flags


def _parse_hhmmss_to_seconds(x: object) -> float:
    """Convert HH:MM:SS -> seconds (float). Safe on malformed values."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")

    s = str(x).strip()
    if not s:
        return float("nan")

    parts = s.split(":")
    if len(parts) != 3:
        return float("nan")

    try:
        h, m, sec = parts
        return float(int(h) * 3600 + int(m) * 60 + int(sec))
    except (TypeError, ValueError):
        return float("nan")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Timestamp": "timestamp",
        "SOC (%)": "soc_pct",
        "Battery Status": "battery_status",
        "Stack Voltage (V)": "stack_voltage_v",
        "Battery Current (A)": "battery_current_a",
        "Output Power (kW)": "output_power_kw",
        "Charger Current Demand (A)": "charger_current_demand_a",
        "Charger Voltage Demand (V)": "charger_voltage_demand_v",
        "Max Cell Voltage (V)": "max_cell_voltage_v",
        "Min Cell Voltage (V)": "min_cell_voltage_v",
        "Avg Cell Voltage (V)": "avg_cell_voltage_v",
        "Max Battery Temp (°C)": "max_battery_temp_c",
        "Min Battery Temp (°C)": "min_battery_temp_c",
        "Avg Battery Temp (°C)": "avg_battery_temp_c",
        "Motor Torque Limit (Nm)": "motor_torque_limit_nm",
        "Motor Torque Value (Nm)": "motor_torque_value_nm",
        "Motor Speed (RPM)": "motor_speed_rpm",
        "Motor Rotation Direction": "motor_rotation_direction",
        "Motor Operation Mode": "motor_operation_mode",
        "MCU Enable State": "mcu_enable_state",
        "Motor AC Current (A)": "motor_ac_current_a",
        "Motor AC Voltage (V)": "motor_ac_voltage_v",
        "DC Side Voltage (V)": "dc_side_voltage_v",
        "Motor Temperature (°C)": "motor_temperature_c",
        "MCU Temperature (°C)": "mcu_temperature_c",
        "Radiator Temperature (°C)": "radiator_temperature_c",
        "DCDC Pri A MOSFET Temp (°C)": "dcdc_pri_a_mosfet_temp_c",
        "DCDC Sec LS MOSFET Temp (°C)": "dcdc_sec_ls_mosfet_temp_c",
        "DCDC Sec HS MOSFET Temp (°C)": "dcdc_sec_hs_mosfet_temp_c",
        "DCDC Pri C MOSFET Temp (°C)": "dcdc_pri_c_mosfet_temp_c",
        "DCDC Input Voltage (V)": "dcdc_input_voltage_v",
        "DCDC Input Current (A)": "dcdc_input_current_a",
        "DCDC Output Voltage (V)": "dcdc_output_voltage_v",
        "DCDC Output Current (A)": "dcdc_output_current_a",
        "DCDC Overcurrent Count": "dcdc_overcurrent_count",
        "Total Running Hours": "total_running_hours",
        "Last Trip Hours": "last_trip_hours",
        "Total kWh Consumed": "total_kwh_consumed",
        "Last Trip kWh": "last_trip_kwh",
    }

    df = df.rename(columns=rename_map).copy()

    # MCU Enable normalization
    # Unknown values stay null; they are not silently forced to 0.
    if "mcu_enable_state" in df.columns:
        raw = df["mcu_enable_state"].astype(str).str.strip().str.lower()
        mapped = raw.map(
            {
                "enabled": 1,
                "disabled": 0,
                "on": 1,
                "off": 0,
                "1": 1,
                "0": 0,
                "true": 1,
                "false": 0,
            }
        )
        df["mcu_enable_state"] = mapped.astype("Int64")

    # HH:MM:SS → seconds
    if "total_running_hours" in df.columns:
        df["total_running_hours_s"] = df["total_running_hours"].apply(_parse_hhmmss_to_seconds)
        df = df.drop(columns=["total_running_hours"])

    if "last_trip_hours" in df.columns:
        df["last_trip_hours_s"] = df["last_trip_hours"].apply(_parse_hhmmss_to_seconds)
        df = df.drop(columns=["last_trip_hours"])

    return df


def run_ingest_for_day(
    dt: str,
    raw_dir: Path,
    raw_fault_dir: Path,
    raw_parquet_dir: Path,
    silver_dir: Path,
    schema_path: Path,
    ranges_path: Path,
    signal_classes_path: Path,
    quality_flags_path: Path,
    resample_cfg_path: Path,
    trip_cfg_path: Path,
    state_path: Path,
) -> dict:
    csvs = list_raw_csvs(raw_dir, dt)
    if not csvs:
        raise FileNotFoundError(f"No raw CSVs found under {raw_dir}/dt={dt}/")

    manifest = {
        "dt": dt,
        "vehicles_processed": [],
        "per_vehicle": {},
        "status": "success",
    }

    for csv_path in csvs:
        vehicle_id = vehicle_id_from_filename(csv_path)

        df = read_csv(csv_path)
        df = standardize_columns(df)
        df["vehicle_id"] = vehicle_id

        vr = validate(
            df,
            schema_path=schema_path,
            ranges_path=ranges_path,
            quality_flags_path=quality_flags_path,
        )

        data_start_ts = vr.df["timestamp"].min()
        data_end_ts = vr.df["timestamp"].max()

        raw_out = parquet_path(raw_parquet_dir, dt, vehicle_id)
        write_parquet_atomic(vr.df, raw_out)

        silver = resample_1hz(
            vr.df,
            resample_cfg=resample_cfg_path,
            signal_classes=signal_classes_path,
            quality_flags=quality_flags_path,
        )

        # Align faults before trip/session assignment for architectural consistency
        fp = fault_csv_path(raw_fault_dir, dt, vehicle_id)
        faults = load_fault_csv(fp)
        fault_count = int(len(faults))
        silver = add_fault_flags(silver, faults)

        silver = add_trip_id(
            silver,
            trip_cfg_path=trip_cfg_path,
            quality_flags_path=quality_flags_path,
            state_path=state_path,
        )

        silver_out = parquet_path(silver_dir, dt, vehicle_id)
        write_parquet_atomic(silver, silver_out)

        manifest["vehicles_processed"].append(vehicle_id)
        manifest["per_vehicle"][vehicle_id] = {
            "raw_rows": int(len(vr.df)),
            "silver_rows": int(len(silver)),
            "data_start_ts": str(data_start_ts),
            "data_end_ts": str(data_end_ts),
            "raw_parquet_path": str(raw_out),
            "silver_path": str(silver_out),
            "validation": vr.report,
            "silver_duplicate_timestamps": int(silver["timestamp"].duplicated().sum()),
            "num_trips": int(silver["trip_id"].nunique(dropna=True)),
            "num_fault_events": fault_count,
            "fault_active_rows": int(silver["fault_any"].sum()) if "fault_any" in silver.columns else 0,
        }

    return manifest