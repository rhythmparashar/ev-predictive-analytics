"""
Global settings (paths + environment switches).
All other modules import paths from here so you never hardcode folder paths.
"""

from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_PARQUET_DIR = DATA_DIR / "raw_parquet"
RAW_FAULT_DIR = DATA_DIR / "raw_faults"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
REPORTS_DIR = DATA_DIR / "reports"
STATE_DIR = DATA_DIR / "state"

SCHEMA_DIR = PROJECT_ROOT / "schema"
CONFIG_DIR = PROJECT_ROOT / "configs"

RUN_MODE = os.getenv("RUN_MODE", "dev")  # dev | prod
DEFAULT_TIMEZONE = os.getenv("TELEMETRY_TIMEZONE", "UTC")