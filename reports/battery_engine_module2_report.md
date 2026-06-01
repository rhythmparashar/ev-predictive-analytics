# Battery Intelligence Engine — Module 2 Report
> Generated: 2026-05-18 11:51 UTC
> Vehicle: EV01

---

## Summary

| Metric | Value |
|--------|-------|
| Master dataset rows analysed | 1,643,119 |
| Cell feature partitions loaded | 88 |
| Chronically weak cells | 10 |
| Chronically hot sensors | 15 |
| Modules at Critical chronic risk | 0 |
| Cells with Critical delta weakness | 13 |
| Most frequently weakest cell | M1_C25 |
| Most frequently hottest sensor | M1_T8 |
| Riskiest module | Module 1 |

---

## 1. Module Chronic Risk Ranking

Modules ranked by chronic voltage and thermal risk across the full date range.

| module_id | avg_voltage_range_mv | max_voltage_range_mv | avg_temp_range_c | max_temp_range_c | pct_time_weakest_voltage | pct_time_hottest | chronic_voltage_risk | chronic_thermal_risk | module_chronic_risk_score | module_chronic_risk_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 37.74 | 3337.0 | 9.35 | 117.0 | 56.65 | 66.14 | True | True | 35 | Monitor |
| 2 | 35.7 | 3337.0 | 9.73 | 117.0 | 20.04 | 3.48 | False | False | 25 | Monitor |
| 3 | 33.86 | 3337.0 | 8.95 | 117.0 | 8.83 | 19.76 | False | False | 20 | Monitor |
| 4 | 34.15 | 3337.0 | 9.0 | 60.0 | 8.36 | 10.25 | False | False | 20 | Monitor |
| 5 | 33.85 | 3337.0 | 8.87 | 117.0 | 6.13 | 0.37 | False | False | 15 | Healthy |

---

## 2. Chronically Weak Cells (Lowest Voltage Frequency)

Cells appearing as the lowest voltage in their module > 15.0% of all timestamps.
Total chronic weak cells: **10**

| cell_id | module_id | times_lowest_in_module | pct_lowest_in_module | times_worst_in_pack | pct_worst_in_pack | weakness_status |
| --- | --- | --- | --- | --- | --- | --- |
| M1_C25 | 1 | 982018 | 59.77 | 703712 | 42.83 | Critical |
| M3_C25 | 3 | 866525 | 52.74 | 74372 | 4.53 | Critical |
| M5_C25 | 5 | 817038 | 49.72 | 55391 | 3.37 | Critical |
| M4_C25 | 4 | 812954 | 49.48 | 61342 | 3.73 | Critical |
| M2_C25 | 2 | 805341 | 49.01 | 107554 | 6.55 | Critical |
| M2_C1 | 2 | 586505 | 35.69 | 112154 | 6.83 | Monitor |
| M4_C1 | 4 | 562921 | 34.26 | 14763 | 0.9 | Monitor |
| M5_C1 | 5 | 537637 | 32.72 | 15736 | 0.96 | Monitor |
| M3_C1 | 3 | 525414 | 31.98 | 18564 | 1.13 | Monitor |
| M1_C1 | 1 | 454816 | 27.68 | 285722 | 17.39 | Watch |

---

## 3. Chronically Hot Sensors (Hottest Sensor Frequency)

Sensors appearing as the hottest in their module > 15.0% of all timestamps.
Total chronic hot sensors: **15**

| sensor_id | module_id | times_hottest_in_module | pct_hottest_in_module | heat_status |
| --- | --- | --- | --- | --- |
| M1_T8 | 1 | 499032 | 30.37 | Monitor |
| M4_T8 | 4 | 492156 | 29.95 | Watch |
| M2_T1 | 2 | 491468 | 29.91 | Watch |
| M1_T1 | 1 | 486255 | 29.59 | Watch |
| M3_T1 | 3 | 472414 | 28.75 | Watch |
| M4_T1 | 4 | 466642 | 28.4 | Watch |
| M5_T1 | 5 | 455522 | 27.72 | Watch |
| M5_T8 | 5 | 431850 | 26.28 | Watch |
| M3_T7 | 3 | 406582 | 24.74 | Watch |
| M1_T7 | 1 | 387532 | 23.59 | Watch |

---

## 4. Cell Voltage Delta Profile (Weakest Cells by Deviation)

Per-cell mean deviation from module mean voltage across all timestamps.
Negative mean_delta_mv = cell consistently below its module mean (weaker cell).

| cell_id | module_id | cell_num | mean_delta_mv | std_delta_mv | outlier_frequency_pct | weakness_score | weakness_status | rank_in_module |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M1_C25 | 1 | 25 | -28.1933 | 26.944 | 68.68 | 100.0 | Critical | 1 |
| M1_C26 | 1 | 26 | -24.5274 | 24.9131 | 56.12 | 100.0 | Critical | 2 |
| M1_C1 | 1 | 1 | -1.1674 | 5.9906 | 9.89 | 32.0 | Monitor | 3 |
| M1_C13 | 1 | 13 | 0.0173 | 6.7821 | 1.86 | 5.58 | Healthy | 4 |
| M1_C17 | 1 | 17 | 0.2129 | 6.3661 | 0.08 | 0.24 | Healthy | 5 |
| M1_C23 | 1 | 23 | 0.7151 | 6.1966 | 0.14 | 0.42 | Healthy | 6 |
| M1_C11 | 1 | 11 | 0.7387 | 5.8497 | 0.09 | 0.27 | Healthy | 7 |
| M1_C7 | 1 | 7 | 0.747 | 6.5056 | 0.13 | 0.39 | Healthy | 8 |
| M1_C8 | 1 | 8 | 0.866 | 4.7378 | 0.13 | 0.39 | Healthy | 9 |
| M1_C19 | 1 | 19 | 1.028 | 4.8349 | 0.08 | 0.24 | Healthy | 10 |

---

## 5. Output Files

| File | Description |
|------|-------------|
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/weak_cell_profile.csv` | Per-cell frequency as module lowest — all cells |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/hot_sensor_profile.csv` | Per-sensor frequency as module hottest — all sensors |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/module_chronic_risk.csv` | Per-module chronic risk scores and flags |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/cell_delta_profile.csv` | Per-cell mean delta, std, outlier frequency from gold features |
| `/Users/rhythmparashar/Desktop/ev-predictive-analytics/outputs/battery_engine/battery_engine_module2_summary.json` | Run metadata |
| `reports/battery_engine_module2_report.md` | This report |
