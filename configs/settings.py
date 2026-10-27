# configs/settings.py
"""
Global settings (paths + environment switches).
All other modules import paths from here so you never hardcode folder paths.
"""

from __future__ import annotations
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

# -------------------------------------------------
# VCU pipeline (unchanged)
# -------------------------------------------------
RAW_DIR         = DATA_DIR / "raw"
RAW_PARQUET_DIR = DATA_DIR / "raw_parquet"
RAW_FAULT_DIR   = DATA_DIR / "raw_faults"
SILVER_DIR      = DATA_DIR / "silver"
GOLD_DIR        = DATA_DIR / "gold"
REPORTS_DIR     = DATA_DIR / "reports"
STATE_DIR       = DATA_DIR / "state"

# -------------------------------------------------
# Cell pipeline (new)
# -------------------------------------------------
RAW_CELL_VOLTAGE_DIR  = DATA_DIR / "raw_cell_voltage"
RAW_CELL_TEMP_DIR     = DATA_DIR / "raw_cell_temp"
SILVER_CELLS_DIR      = DATA_DIR / "silver_cells"
GOLD_CELLS_DIR        = DATA_DIR / "gold_cells"

GOLD_CELL_FEATURES_DIR       = GOLD_CELLS_DIR / "cell_features"
GOLD_MODULE_FEATURES_DIR     = GOLD_CELLS_DIR / "module_features"
GOLD_PACK_FEATURES_DIR       = GOLD_CELLS_DIR / "pack_features"
GOLD_DAILY_HEALTH_DIR        = GOLD_CELLS_DIR / "daily_health_reports"

CELL_REPORTS_DIR      = REPORTS_DIR / "cell"

# -------------------------------------------------
# Schema + config
# -------------------------------------------------
SCHEMA_DIR = PROJECT_ROOT / "schema"
CONFIG_DIR = PROJECT_ROOT / "configs"

# -------------------------------------------------
# Fleet — vehicle → machine type registry
# -------------------------------------------------
MACHINE_TYPES: dict[str, str] = {
    "EV01": "Loader",
    "EV02": "Excavator",
}

# Merge in any vehicles added via the dashboard onboarding UI
_VEHICLE_REGISTRY_PATH = DATA_DIR / "vehicle_registry.json"
if _VEHICLE_REGISTRY_PATH.exists():
    try:
        import json as _json
        _extra: dict[str, str] = _json.loads(_VEHICLE_REGISTRY_PATH.read_text())
        MACHINE_TYPES.update(_extra)
    except Exception:
        pass

# -------------------------------------------------
# Environment
# -------------------------------------------------
RUN_MODE         = os.getenv("RUN_MODE", "dev")   # dev | prod
DEFAULT_TIMEZONE = os.getenv("TELEMETRY_TIMEZONE", "UTC")

# -------------------------------------------------
# Pydantic operational settings
# Every field is overridable via environment variable.
# Example: MLFLOW_URI=http://remote:5001 python scripts/train_soc_ev01.py
# -------------------------------------------------
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    mlflow_uri: str             = Field("http://localhost:5001", alias="MLFLOW_URI")
    soc_mae_alert_threshold: float = Field(1.5, alias="SOC_MAE_ALERT_THRESHOLD")
    soc_mae_retrain_threshold: float = Field(1.0, alias="SOC_MAE_RETRAIN_THRESHOLD")
    drop_quality_mask: int      = Field(52,  alias="DROP_QUALITY_MASK")
    log_level: str              = Field("INFO", alias="LOG_LEVEL")

    model_config = {"populate_by_name": True, "env_file": ".env", "extra": "ignore"}


settings = Settings()
