from pathlib import Path
import pytest

SCHEMA_DIR = Path(__file__).parents[1] / "schema"
CONFIG_DIR = Path(__file__).parents[1] / "configs"


@pytest.fixture()
def schema_path():
    return SCHEMA_DIR / "telemetry_schema.yaml"


@pytest.fixture()
def ranges_path():
    return SCHEMA_DIR / "ranges.yaml"


@pytest.fixture()
def quality_flags_path():
    return SCHEMA_DIR / "quality_flags.yaml"


@pytest.fixture()
def signal_classes_path():
    return SCHEMA_DIR / "signal_classes.yaml"


@pytest.fixture()
def resample_cfg():
    return CONFIG_DIR / "resample.yaml"


@pytest.fixture()
def trip_cfg_path():
    return CONFIG_DIR / "trip.yaml"
