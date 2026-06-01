"""
All configurable thresholds for Battery Intelligence Engine Modules 1 and 2.
Import THRESHOLDS, PENALTY_VALUES, RULE_THRESHOLDS, MODULE2_THRESHOLDS from here.

Voltage values in master dataset are in VOLTS.
Convert to millivolts (×1000) before comparing against mV thresholds defined here.
"""

from __future__ import annotations

MODULES = [1, 2, 3, 4, 5]

# Cell quality flag bitmasks (from schema/cell_quality_flags.yaml)
CELL_FLAG = {
    "CELL_FFILLED": 1,
    "MODULE_GAP": 2,
    "PACK_GAP": 4,
    "CELL_SOFT_BREACH": 8,
    "CELL_HARD_BREACH": 16,
    "CELL_FLATLINE": 32,
    "CELL_OUTLIER": 64,
}

# Penalty step thresholds
THRESHOLDS: dict = {
    # max_battery_temp_c (°C)
    "thermal": {
        "warn": 45.0,
        "high": 50.0,
        "critical": 55.0,
    },
    # pack_voltage_range × 1000 (mV)
    # Data: median=36mV, p90=80mV, p97.6=120mV — low raised from 40→50 (was firing 49% of time)
    "voltage_imbalance_mv": {
        "low": 50.0,
        "medium": 80.0,
        "high": 120.0,
    },
    # pack_temp_range (°C)
    # Data: median=10°C, p86=18°C, p93=22°C — all bands raised; old low=6 fired 70% of time
    "temp_imbalance": {
        "low": 10.0,
        "medium": 16.0,
        "high": 22.0,
    },
    # abs_current_a (A)
    # Data: p95=147A, p99=165A — high lowered from 180→150 (180A fired on only 13 rows)
    "current_stress": {
        "low": 50.0,
        "medium": 100.0,
        "high": 150.0,
    },
    # per-module voltage range × 1000 (mV)
    "weak_module_voltage_mv": {
        "warn": 80.0,
        "alert": 120.0,
    },
    # per-module temp range (°C)
    "weak_module_temp": {
        "warn": 12.0,
        "alert": 18.0,
    },
}

PENALTY_VALUES: dict = {
    # thermal
    "thermal_warn": 10,
    "thermal_high": 20,
    "thermal_critical": 35,
    # voltage imbalance
    "voltage_imbalance_low": 10,
    "voltage_imbalance_medium": 20,
    "voltage_imbalance_high": 35,
    # temp imbalance
    "temp_imbalance_low": 10,
    "temp_imbalance_medium": 20,
    "temp_imbalance_high": 35,
    # current stress
    "current_low": 8,
    "current_medium": 15,
    "current_high": 25,
    # weak module
    "weak_module_voltage_warn": 10,
    "weak_module_voltage_alert": 20,
    "weak_module_temp_warn": 10,
    "weak_module_temp_alert": 20,
    "weak_module_cap": 30,
    # data quality
    "data_quality_not_clean": 20,
    "data_quality_pack_gap": 15,
    "data_quality_hard_breach": 15,
    "data_quality_module_gap": 8,
    "data_quality_soft_breach": 5,
    "data_quality_vcu_flag": 10,
    "data_quality_outlier": 5,
    # fault
    "fault": 25,
}

# Health status score bands
HEALTH_STATUS: dict = {
    "EXCELLENT": 90,
    "HEALTHY": 75,
    "WARNING": 60,
    "CRITICAL": 40,
    "SEVERE": 0,
}

# Status priority for worst-case comparisons (lower = worse)
HEALTH_STATUS_PRIORITY: dict = {
    "SEVERE": 0,
    "CRITICAL": 1,
    "WARNING": 2,
    "HEALTHY": 3,
    "EXCELLENT": 4,
    "NO_DATA": 5,
}

# Module 2 — Weak Cell / Weak Module / Hot Sensor Intelligence thresholds
MODULE2_THRESHOLDS: dict = {
    # A cell appearing as the lowest voltage in its module more than this % is chronic
    "chronic_weak_cell_pct": 15.0,
    # A sensor appearing as the hottest in its module more than this % is chronic
    "chronic_hot_sensor_pct": 15.0,
    # A module appearing as the weakest (highest voltage range) more than this % is chronic
    "chronic_weak_module_pct": 30.0,
    # A module appearing as the hottest (hottest_module_id) more than this % is chronic
    "chronic_hot_module_pct": 30.0,
    # Average module voltage range (mV) above which the module is flagged as chronically imbalanced
    "chronic_module_voltage_mv": 80.0,
    # Average module temp range (°C) above which the module is flagged as chronically hot
    "chronic_module_temp_c": 10.0,
    # Cell delta (deviation from module mean, mV) below which cell is considered undervoltage
    "cell_weak_delta_mv": -20.0,
    # Cell outlier frequency (% of timestamps) above which cell is flagged as unstable
    "cell_outlier_pct": 5.0,
    # Module risk score thresholds (0–100)
    "module_risk_monitor": 20,
    "module_risk_critical": 50,
}

# Rule engine alert thresholds
RULE_THRESHOLDS: dict = {
    "temp_warning_c": 45.0,
    "temp_critical_c": 50.0,
    # voltage imbalance in mV (pack_voltage_range × 1000)
    "voltage_imbalance_warning_mv": 90.0,
    "voltage_imbalance_critical_mv": 130.0,
    "temp_imbalance_warning_c": 16.0,
    "temp_imbalance_critical_c": 22.0,
    "high_current_a": 100.0,
    "combined_current_a": 80.0,
    "combined_temp_c": 43.0,
    "weak_module_voltage_mv": 120.0,
    "weak_module_temp_c": 15.0,
}
