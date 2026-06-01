# run_cells.py
"""
Cell pipeline CLI entry point.

Commands:
  cell_ingest   raw_cell_voltage + raw_cell_temp → silver_cells
  cell_gold     silver_cells → gold_cells (cell + module + pack features)
  cell_report   gold_cells → daily_health_reports + terminal dashboard

Usage:
  python run_cells.py cell_ingest --dt 2026-03-10
  python run_cells.py cell_gold   --dt 2026-03-10
  python run_cells.py cell_report --dt 2026-03-10

  # Full pipeline for a new date
  python run_cells.py cell_ingest --dt 2026-03-10
  python run_cells.py cell_gold   --dt 2026-03-10
  python run_cells.py cell_report --dt 2026-03-10

  # Backfill
  python run_cells.py cell_ingest --backfill --start 2026-01-15 --end 2026-03-08
  python run_cells.py cell_gold   --backfill --start 2026-01-15 --end 2026-03-08
  python run_cells.py cell_report --backfill --start 2026-01-15 --end 2026-03-08
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import date, timedelta
from pathlib import Path

from configs.settings import (
    # Cell pipeline dirs
    RAW_CELL_VOLTAGE_DIR,
    RAW_CELL_TEMP_DIR,
    SILVER_CELLS_DIR,
    GOLD_CELL_FEATURES_DIR,
    GOLD_MODULE_FEATURES_DIR,
    GOLD_PACK_FEATURES_DIR,
    GOLD_DAILY_HEALTH_DIR,
    CELL_REPORTS_DIR,
    # Schema + config
    SCHEMA_DIR,
    CONFIG_DIR,
)

from ingestion.cell_ingest import run_cell_ingest_for_day
from features.cell_pipeline import run_cell_gold_for_day
from reports.cell_health_report import run_cell_health_report


# -------------------------------------------------
# Schema + config paths
# -------------------------------------------------

VOLTAGE_SCHEMA_PATH      = SCHEMA_DIR / "cell_voltage_schema.yaml"
TEMP_SCHEMA_PATH         = SCHEMA_DIR / "cell_temp_schema.yaml"
CELL_QUALITY_FLAGS_PATH  = SCHEMA_DIR / "cell_quality_flags.yaml"
CELL_CFG_PATH            = CONFIG_DIR / "cell.yaml"


# -------------------------------------------------
# Date range helper
# -------------------------------------------------

def date_range(start: str, end: str) -> list[str]:
    """Return list of YYYY-MM-DD strings from start to end inclusive."""
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    if s > e:
        raise ValueError(f"start={start} is after end={end}")
    out = []
    current = s
    while current <= e:
        out.append(current.isoformat())
        current += timedelta(days=1)
    return out


# -------------------------------------------------
# Commands
# -------------------------------------------------

def cmd_cell_ingest(dt: str, vehicle_id_filter: str | None = None) -> None:
    print(f"\n── Cell Ingest  dt={dt} {'─' * 40}")
    manifest = run_cell_ingest_for_day(
        dt=dt,
        raw_cell_voltage_dir=RAW_CELL_VOLTAGE_DIR,
        raw_cell_temp_dir=RAW_CELL_TEMP_DIR,
        silver_cells_dir=SILVER_CELLS_DIR,
        voltage_schema_path=VOLTAGE_SCHEMA_PATH,
        temp_schema_path=TEMP_SCHEMA_PATH,
        cell_quality_flags_path=CELL_QUALITY_FLAGS_PATH,
        cell_cfg_path=CELL_CFG_PATH,
        reports_dir=CELL_REPORTS_DIR / "ingest",
        vehicle_id_filter=vehicle_id_filter,
    )
    vehicles = manifest.get("vehicles_processed", [])
    print(f"[OK] cell_ingest dt={dt} vehicles={vehicles}")


def cmd_cell_gold(dt: str, vehicle_id_filter: str | None = None) -> None:
    print(f"\n── Cell Gold  dt={dt} {'─' * 40}")
    manifest = run_cell_gold_for_day(
        dt=dt,
        silver_cells_dir=SILVER_CELLS_DIR,
        gold_cell_features_dir=GOLD_CELL_FEATURES_DIR,
        gold_module_features_dir=GOLD_MODULE_FEATURES_DIR,
        gold_pack_features_dir=GOLD_PACK_FEATURES_DIR,
        voltage_schema_path=VOLTAGE_SCHEMA_PATH,
        temp_schema_path=TEMP_SCHEMA_PATH,
        cell_cfg_path=CELL_CFG_PATH,
        cell_quality_flags_path=CELL_QUALITY_FLAGS_PATH,
        reports_dir=CELL_REPORTS_DIR / "gold",
        vehicle_id_filter=vehicle_id_filter,
    )
    vehicles = manifest.get("vehicles_processed", [])
    print(f"[OK] cell_gold dt={dt} vehicles={vehicles}")


def cmd_cell_report(dt: str, vehicle_id_filter: str | None = None) -> None:
    print(f"\n── Cell Report  dt={dt} {'─' * 40}")
    manifest = run_cell_health_report(
        dt=dt,
        gold_module_features_dir=GOLD_MODULE_FEATURES_DIR,
        gold_pack_features_dir=GOLD_PACK_FEATURES_DIR,
        gold_daily_health_dir=GOLD_DAILY_HEALTH_DIR,
        voltage_schema_path=VOLTAGE_SCHEMA_PATH,
        cell_cfg_path=CELL_CFG_PATH,
        reports_dir=CELL_REPORTS_DIR / "health",
        vehicle_id_filter=vehicle_id_filter,
    )
    vehicles = manifest.get("vehicles_processed", [])
    print(f"[OK] cell_report dt={dt} vehicles={vehicles}")


# -------------------------------------------------
# Backfill runner
# -------------------------------------------------

def run_backfill(
    command: str,
    start: str,
    end: str,
    vehicle_id_filter: str | None = None,
) -> None:
    """
    Run a command over a date range.
    Skips dates where source data is missing.
    Continues on error — reports failures at end.
    """

    dates   = date_range(start, end)
    failed  = []
    skipped = []

    cmd_fn = {
        "cell_ingest": cmd_cell_ingest,
        "cell_gold":   cmd_cell_gold,
        "cell_report": cmd_cell_report,
    }[command]

    print(f"\nBackfill: {command}  {start} → {end}  ({len(dates)} dates)"
          + (f"  vehicle={vehicle_id_filter}" if vehicle_id_filter else ""))

    for dt in dates:
        try:
            cmd_fn(dt, vehicle_id_filter=vehicle_id_filter)
        except FileNotFoundError as e:
            skipped.append((dt, str(e)))
            print(f"  [SKIP] {dt} — {e}")
        except Exception as e:
            failed.append((dt, str(e)))
            print(f"  [FAIL] {dt} — {e}")
            traceback.print_exc()

    print(f"\nBackfill complete: "
          f"{len(dates) - len(failed) - len(skipped)} ok  "
          f"{len(skipped)} skipped  "
          f"{len(failed)} failed")

    if failed:
        print("\nFailed dates:")
        for dt, reason in failed:
            print(f"  {dt}: {reason}")
        sys.exit(1)


# -------------------------------------------------
# Argument parser
# -------------------------------------------------

def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog="run_cells.py",
        description="Cell health pipeline CLI",
    )

    parser.add_argument(
        "command",
        choices=["cell_ingest", "cell_gold", "cell_report"],
        help="Pipeline stage to run",
    )

    # Single date
    parser.add_argument(
        "--dt",
        type=str,
        default=None,
        help="Date to process: YYYY-MM-DD",
    )

    # Backfill
    parser.add_argument(
        "--backfill",
        action="store_true",
        default=False,
        help="Run over a date range (requires --start and --end)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Backfill start date: YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Backfill end date: YYYY-MM-DD",
    )
    parser.add_argument(
        "--vehicle-id",
        type=str,
        default=None,
        dest="vehicle_id",
        help="Only process this vehicle (e.g. EV02). If omitted, all vehicles are processed.",
    )

    return parser


# -------------------------------------------------
# Main
# -------------------------------------------------

def main() -> None:

    parser = build_parser()
    args   = parser.parse_args()

    # -------------------------------------------------
    # Backfill mode
    # -------------------------------------------------

    if args.backfill:
        if not args.start or not args.end:
            parser.error("--backfill requires --start and --end")
        run_backfill(args.command, args.start, args.end, vehicle_id_filter=args.vehicle_id)
        return

    # -------------------------------------------------
    # Single date mode
    # -------------------------------------------------

    if not args.dt:
        parser.error("--dt is required unless --backfill is set")

    if args.command == "cell_ingest":
        cmd_cell_ingest(args.dt, vehicle_id_filter=args.vehicle_id)

    elif args.command == "cell_gold":
        cmd_cell_gold(args.dt, vehicle_id_filter=args.vehicle_id)

    elif args.command == "cell_report":
        cmd_cell_report(args.dt, vehicle_id_filter=args.vehicle_id)


if __name__ == "__main__":
    main()