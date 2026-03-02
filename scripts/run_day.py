"""
Run one day of the pipeline:

- Ingest raw CSV -> raw_parquet -> silver (trip_id + faults)
- Write a run manifest JSON to data/reports/dt=YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path


# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


# ---------------------------------------------------------
# Imports
# ---------------------------------------------------------

from configs.settings import (
    RAW_DIR,
    RAW_FAULT_DIR,
    RAW_PARQUET_DIR,
    SILVER_DIR,
    REPORTS_DIR,
    SCHEMA_DIR,
    CONFIG_DIR,
    STATE_DIR,
)

from ingestion.ingest import run_ingest_for_day


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dt",
        required=True,
        help="Date partition folder name (example: 2026-02-24)",
    )

    args = parser.parse_args()

    dt = args.dt

    t0 = time.time()

    # Run ingestion pipeline

    manifest = run_ingest_for_day(

        dt=dt,

        raw_dir=RAW_DIR,

        raw_fault_dir=RAW_FAULT_DIR,

        raw_parquet_dir=RAW_PARQUET_DIR,

        silver_dir=SILVER_DIR,

        schema_path=SCHEMA_DIR / "telemetry_schema.yaml",

        ranges_path=SCHEMA_DIR / "ranges.yaml",

        signal_classes_path=SCHEMA_DIR / "signal_classes.yaml",

        quality_flags_path=SCHEMA_DIR / "quality_flags.yaml",

        resample_cfg_path=CONFIG_DIR / "resample.yaml",

        trip_cfg_path=CONFIG_DIR / "trip.yaml",

        state_path=STATE_DIR / "open_trips.parquet",
    )

    # Add runtime

    manifest["duration_seconds"] = round(
        time.time() - t0,
        3
    )

    # Write manifest

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    out = REPORTS_DIR / f"dt={dt}.json"

    with open(out, "w", encoding="utf-8") as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    print(f"[OK] wrote report: {out}")


# ---------------------------------------------------------

if __name__ == "__main__":
    main()