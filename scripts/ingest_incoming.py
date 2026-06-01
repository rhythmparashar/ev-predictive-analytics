"""
Incoming File Ingestion
=======================
Scans data/incoming/ for new telemetry and cell data files (CSV or Excel),
converts them to the correct format, then triggers the daily pipeline.

Naming convention:
    EV01_TELEMETRY_2026-06-01.xlsx    →  raw_parquet  (telemetry)
    EV01_CELL_VOLTAGE_2026-06-01.xlsx →  raw_cell_voltage  (cell pipeline)
    EV01_CELL_TEMP_2026-06-01.xlsx    →  raw_cell_temp     (cell pipeline)

    Legacy CSV format also supported:
    EV01_2026-06-01.csv               →  raw_parquet  (telemetry)

The script is idempotent — already-processed files live in data/incoming/processed/.

Usage:
    python scripts/ingest_incoming.py           # process all new files + run pipeline
    python scripts/ingest_incoming.py --scan    # only show what would be processed
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import (
    RAW_PARQUET_DIR, RAW_CELL_VOLTAGE_DIR, RAW_CELL_TEMP_DIR, MACHINE_TYPES
)

INCOMING_DIR  = PROJECT_ROOT / "data" / "incoming"
PROCESSED_DIR = INCOMING_DIR / "processed"

FILE_TYPES = ("TELEMETRY", "CELL_VOLTAGE", "CELL_TEMP")

# Required columns for telemetry files
REQUIRED_TELEMETRY_COLS = {
    "timestamp", "soc_pct", "stack_voltage_v", "battery_current_a",
    "output_power_kw", "avg_battery_temp_c", "motor_temperature_c",
    "total_kwh_consumed",
}

# All expected raw_parquet columns — extras filled with NaN
EXPECTED_TELEMETRY_COLS = [
    "timestamp", "soc_pct", "battery_status", "stack_voltage_v",
    "battery_current_a", "output_power_kw", "charger_current_demand_a",
    "charger_voltage_demand_v", "max_cell_voltage_v", "min_cell_voltage_v",
    "avg_cell_voltage_v", "max_battery_temp_c", "min_battery_temp_c",
    "avg_battery_temp_c", "motor_torque_limit_nm", "motor_torque_value_nm",
    "motor_speed_rpm", "motor_rotation_direction", "motor_operation_mode",
    "mcu_enable_state", "motor_ac_current_a", "motor_ac_voltage_v",
    "dc_side_voltage_v", "motor_temperature_c", "mcu_temperature_c",
    "radiator_temperature_c", "dcdc_input_voltage_v", "dcdc_input_current_a",
    "dcdc_output_voltage_v", "dcdc_output_current_a",
    "total_kwh_consumed", "last_trip_kwh", "total_running_hours_s",
    "last_trip_hours_s", "quality_flag",
]


def read_file(path: Path) -> pd.DataFrame:
    """Read CSV or Excel file into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def parse_filename(path: Path) -> tuple[str, str, str] | None:
    """
    Extract (vehicle_id, file_type, date) from filename.

    Supported formats:
      EV01_TELEMETRY_2026-06-01.xlsx    → (EV01, TELEMETRY, 2026-06-01)
      EV01_CELL_VOLTAGE_2026-06-01.xlsx → (EV01, CELL_VOLTAGE, 2026-06-01)
      EV01_CELL_TEMP_2026-06-01.xlsx    → (EV01, CELL_TEMP, 2026-06-01)
      EV01_2026-06-01.csv               → (EV01, TELEMETRY, 2026-06-01)  legacy
    """
    stem = path.stem.upper()

    # New format: EV01_TELEMETRY_2026-06-01 or EV01_CELL_VOLTAGE_2026-06-01
    m = re.match(r"(EV\d+)[_\-](TELEMETRY|CELL_VOLTAGE|CELL_TEMP)[_\-](\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(1), m.group(2), m.group(3)

    # Legacy format: EV01_2026-06-01
    m = re.match(r"(EV\d+)[_\-](\d{4}-\d{2}-\d{2})", stem)
    if m:
        return m.group(1), "TELEMETRY", m.group(2)

    return None


def find_new_files() -> list[Path]:
    """All CSV/Excel files in incoming/ that haven't been moved to processed/."""
    if not INCOMING_DIR.exists():
        return []
    files = []
    for ext in ("*.csv", "*.xlsx", "*.xls"):
        files.extend(
            p for p in INCOMING_DIR.glob(ext)
            if p.is_file() and not p.name.startswith(".")
        )
    return sorted(files)


def ingest_telemetry(df: pd.DataFrame, vehicle_id: str, dt: str) -> tuple[bool, str]:
    """Validate and write telemetry data to raw_parquet."""
    missing = REQUIRED_TELEMETRY_COLS - set(df.columns)
    if missing:
        return False, f"missing columns: {sorted(missing)}"

    df = df.copy()
    df["vehicle_id"] = vehicle_id
    df["dt"]         = dt

    for col in EXPECTED_TELEMETRY_COLS:
        if col not in df.columns:
            df[col] = None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    null_ts = df["timestamp"].isna().sum()
    if null_ts > len(df) * 0.5:
        return False, f"timestamp parse failed on {null_ts}/{len(df)} rows"

    df = df[df["timestamp"].dt.date.astype(str) == dt]
    if df.empty:
        return False, f"no rows for date {dt} (timestamps don't match filename)"

    out_dir = RAW_PARQUET_DIR / f"dt={dt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vehicle_id={vehicle_id}.parquet"
    df.to_parquet(out_path, index=False)
    return True, f"{len(df):,} rows → {out_path}"


def ingest_cell_file(df: pd.DataFrame, vehicle_id: str, dt: str, file_type: str) -> tuple[bool, str]:
    """Write cell voltage or temp data as CSV for the cell ingest pipeline."""
    raw_dir = RAW_CELL_VOLTAGE_DIR if file_type == "CELL_VOLTAGE" else RAW_CELL_TEMP_DIR

    df = df.copy()
    df["vehicle_id"] = vehicle_id

    # Normalise timestamp column name (Excel may export it differently)
    ts_candidates = [c for c in df.columns if c.lower() in ("timestamp", "time", "datetime")]
    if ts_candidates and ts_candidates[0] != "Timestamp":
        df = df.rename(columns={ts_candidates[0]: "Timestamp"})

    if "Timestamp" not in df.columns:
        return False, "no timestamp column found (expected 'Timestamp')"

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce")
    null_ts = df["Timestamp"].isna().sum()
    if null_ts > len(df) * 0.5:
        return False, f"timestamp parse failed on {null_ts}/{len(df)} rows"

    df = df[df["Timestamp"].dt.date.astype(str) == dt]
    if df.empty:
        return False, f"no rows for date {dt}"

    out_dir = raw_dir / f"dt={dt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"vehicle_id={vehicle_id}.csv"
    df.to_csv(out_path, index=False)
    return True, f"{len(df):,} rows → {out_path}"


def move_to_processed(path: Path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    dest = PROCESSED_DIR / f"{path.stem}_{ts}{path.suffix}"
    shutil.move(str(path), str(dest))


def run_pipeline_for_dates(dates: list[str]) -> None:
    import subprocess
    for dt in sorted(set(dates)):
        print(f"\n  Running pipeline for {dt}…")
        result = subprocess.run(
            [sys.executable, "scripts/run_daily_pipeline.py", "--date", dt],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print(f"  Pipeline failed for {dt} (exit {result.returncode})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest incoming telemetry and cell data files")
    ap.add_argument("--scan", action="store_true", help="Show files without processing")
    args = ap.parse_args()

    files = find_new_files()

    if not files:
        print("No new files in data/incoming/")
        print("Expected: EV01_TELEMETRY_2026-06-01.xlsx, EV01_CELL_VOLTAGE_2026-06-01.xlsx, EV01_CELL_TEMP_2026-06-01.xlsx")
        return

    print(f"\nFound {len(files)} file(s) in data/incoming/\n")

    new_dates: list[str] = []

    for f in sorted(files):
        parsed = parse_filename(f)
        if not parsed:
            print(f"  SKIP  {f.name}  (unrecognised name — use EV01_TELEMETRY_2026-06-01.xlsx format)")
            continue

        vehicle_id, file_type, dt = parsed
        machine = MACHINE_TYPES.get(vehicle_id, "Unknown")
        print(f"  {f.name}  →  {vehicle_id} ({machine})  type={file_type}  date={dt}")

        if args.scan:
            continue

        try:
            df = read_file(f)
        except Exception as e:
            print(f"    ✗  failed to read file: {e}")
            continue

        if file_type == "TELEMETRY":
            ok, msg = ingest_telemetry(df, vehicle_id, dt)
        else:
            ok, msg = ingest_cell_file(df, vehicle_id, dt, file_type)

        if ok:
            print(f"    ✓  {msg}")
            move_to_processed(f)
            new_dates.append(dt)
        else:
            print(f"    ✗  {msg}")

    if new_dates and not args.scan:
        run_pipeline_for_dates(new_dates)


if __name__ == "__main__":
    main()
