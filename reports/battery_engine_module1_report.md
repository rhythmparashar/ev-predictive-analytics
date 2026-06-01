# Battery Engine Module 1 Report
> Generated: 2026-05-18 12:23 UTC  
> Vehicle: EV01  

---

## 1. Rows Processed

| Metric | Value |
|--------|-------|
| Total rows | 1,643,119 |
| Date range | 2026-01-22 → 2026-05-17 |
| Total alert rows | 1,674,976 |
| Days in daily summary | 89 |

---

## 2. Date Range

**2026-01-22 → 2026-05-17**

---

## 3. Columns Used

48 columns available from master dataset used in scoring:

```
abs_current_a, avg_battery_temp_c, avg_cell_voltage_v, battery_current_a, cell_quality_flag, cell_spread_v, current_direction, dt, fault_any, hottest_module_id, imbalance_flag, is_clean, max_battery_temp_c, max_cell_voltage_v, min_battery_temp_c, min_cell_voltage_v, module_1_hottest_sensor, module_1_lowest_cell, module_1_temp_range, module_1_voltage_range, module_2_hottest_sensor, module_2_lowest_cell, module_2_temp_range, module_2_voltage_range, module_3_hottest_sensor, module_3_lowest_cell, module_3_temp_range, module_3_voltage_range, module_4_hottest_sensor, module_4_lowest_cell, module_4_temp_range, module_4_voltage_range, module_5_hottest_sensor, module_5_lowest_cell, module_5_temp_range, module_5_voltage_range, operating_regime, output_power_kw, pack_temp_range, pack_voltage_range, pack_voltage_std, soc_pct, stack_voltage_v, timestamp, trip_id, vcu_quality_flag, vehicle_id, worst_cell_id
```

---

## 4. Missing Optional Columns

All expected columns present.

---

## 5. Health Score Distribution

| Stat | Value |
|------|-------|
| Mean | 72.66 |
| Std  | 21.54 |
| Min  | 0.00 |
| 25%  | 60.00 |
| 50%  | 80.00 |
| 75%  | 90.00 |
| Max  | 100.00 |

---

## 6. Status Distribution

| Status | Row Count | % of Total |
|--------|-----------|------------|
| SEVERE | 139,861 | 8.5% |
| CRITICAL | 210,126 | 12.8% |
| WARNING | 390,289 | 23.8% |
| HEALTHY | 362,692 | 22.1% |
| EXCELLENT | 540,151 | 32.9% |

---

## 7. Top Alert Types

| Rule ID | Alert Count |
|---------|-------------|
| WEAK_MODULE_THERMAL | 392,682 |
| TEMP_IMBALANCE_WARNING | 240,069 |
| HIGH_CURRENT_STRESS | 236,800 |
| BATTERY_TEMP_WARNING | 207,151 |
| HIGH_CURRENT_WITH_HIGH_TEMP | 127,805 |
| BMS_OR_CELL_DATA_QUALITY_ISSUE | 105,273 |
| VOLTAGE_IMBALANCE_WARNING | 99,237 |
| TEMP_IMBALANCE_CRITICAL | 97,126 |
| BATTERY_FAULT_PRESENT | 79,437 |
| WEAK_MODULE_VOLTAGE | 33,680 |

---

## 8. Worst Battery Health Days

| dt | vehicle_id | avg_health_score | min_health_score | worst_health_status | top_issue |
| --- | --- | --- | --- | --- | --- |
| 2026-01-27 | EV01 | 24.14 | 0.0 | SEVERE | Critical thermal gradient across modules |
| 2026-01-24 | EV01 | 29.28 | 0.0 | SEVERE | Critical thermal gradient across modules |
| 2026-01-22 | EV01 | 30.88 | 0.0 | SEVERE | Critical thermal gradient across modules |
| 2026-01-25 | EV01 | 35.66 | 12.0 | SEVERE | Critical thermal gradient across modules |
| 2026-01-23 | EV01 | 36.21 | 0.0 | SEVERE | Critical thermal gradient across modules |

---

## 9. Weakest Modules Summary

| dt | vehicle_id | module_id | max_voltage_range_mv | max_temp_range_c | module_risk_status | module_risk_score |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-01-22 | EV01 | 1 | 166.0 | 20.0 | Critical | 80 |
| 2026-01-23 | EV01 | 3 | 179.0 | 60.0 | Critical | 80 |
| 2026-01-24 | EV01 | 2 | 174.0 | 34.0 | Critical | 80 |
| 2026-01-24 | EV01 | 5 | 146.0 | 60.0 | Critical | 80 |
| 2026-01-27 | EV01 | 1 | 3299.0 | 117.0 | Critical | 80 |
| 2026-01-27 | EV01 | 2 | 3290.0 | 117.0 | Critical | 80 |
| 2026-01-27 | EV01 | 3 | 3290.0 | 117.0 | Critical | 80 |
| 2026-01-27 | EV01 | 4 | 3291.0 | 60.0 | Critical | 80 |
| 2026-01-27 | EV01 | 5 | 3290.0 | 60.0 | Critical | 80 |
| 2026-01-28 | EV01 | 1 | 3333.0 | 41.0 | Critical | 80 |

---

## 10. Example Battery Twin State

```json
{
  "vehicle_id": "EV01",
  "timestamp": "2026-05-17 23:59:59",
  "battery": {
    "health_score": 100.0,
    "status": "EXCELLENT",
    "soc_pct": 69.0,
    "max_temp_c": 40.0,
    "voltage_imbalance_mv": 24.0,
    "temp_imbalance_c": 2.0,
    "abs_current_a": 4.0,
    "output_power_kw": 2.36,
    "imbalance_flag": false,
    "fault_active": false,
    "weakest_module_by_voltage": "2",
    "weakest_module_by_temp": "1",
    "top_issue": "Battery operating normally",
    "recommended_action": "Continue normal monitoring. No action required.",
    "business_impact": "None \u2014 battery is within all healthy operating thresholds."
  },
  "penalties": {
    "thermal": 0,
    "voltage_imbalance": 0,
    "temperature_imbalance": 0,
    "current_stress": 0,
    "weak_module": 0,
    "data_quality": 0,
    "fault": 0
  },
  "modules": [
    {
      "module_id": 1,
      "voltage_range_mv": 8.0,
      "temp_range_c": 1.0,
      "lowest_cell": "M1_C1",
      "hottest_sensor": "M1_T1"
    },
    {
      "module_id": 2,
      "voltage_range_mv": 19.0,
      "temp_range_c": 1.0,
      "lowest_cell": "M2_C1",
      "hottest_sensor": "M2_T7"
    },
    {
      "module_id": 3,
      "voltage_range_mv": 8.0,
      "temp_range_c": 1.0,
      "lowest_cell": "M3_C1",
      "hottest_sensor": "M3_T7"
    },
    {
      "module_id": 4,
      "voltage_range_mv": 11.0,
      "temp_range_c": 1.0,
      "lowest_cell": "M4_C25",
      "hottest_sensor": "M4_T3"
    },
    {
      "module_id": 5,
      "voltage_range_mv": 15.0,
      "temp_range_c": 1.0,
      "lowest_cell": "M5_C31",
      "hottest_sensor": "M5_T8"
    }
  ]
}
```

---

## 11. Output File Paths

| File | Description |
|------|-------------|
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_health_scores.parquet` | Full health score + penalties per timestamp |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_alerts.parquet` | Long-format rule alerts (triggered rows only) |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_daily_summary.csv` | Daily aggregated health metrics per vehicle |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_module_summary.csv` | Per-module daily risk scores |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_engine_module1_summary.json` | Run metadata and key stats |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_twin_state_sample.json` | Latest-timestamp twin state snapshot |
