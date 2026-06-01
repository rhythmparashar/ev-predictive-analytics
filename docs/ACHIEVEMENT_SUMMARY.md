# ACHIEVEMENT_SUMMARY.md
## EV Predictive Analytics — Factual Handoff Summary
> Generated: 2026-05-18 | Vehicle: EV01 | Inspected by: Claude Code

---

## 1. Project Overview

This folder contains a fully functional **EV telemetry predictive analytics engine** built for a single vehicle (`EV01`). It is part of a larger AI-powered EV Digital Twin and Fleet Intelligence Platform. The project covers:

- **Raw telemetry ingestion** from VCU (vehicle control unit) CSVs and cell-level CSVs
- **Silver pipeline**: Cleaning, validation, 1-Hz resampling, fault flagging, trip segmentation
- **Gold pipeline**: Feature engineering (rolling, lag, physics, trip aggregations, cell-level aggregations)
- **Cell analytics pipeline**: A separate pipeline for 180-cell voltage + 90-sensor temperature data
- **Master dataset**: A merged analysis-ready dataset combining VCU silver + cell pack and module features
- **SOC Forecasting**: Trained XGBoost and LightGBM models predicting SOC 5 minutes ahead
- **Drift Monitoring**: Production-grade feature drift detection comparing live data to training baseline
- **Reports and charts**: Fault visualizations, battery analysis CSVs, telemetry trip PDF, cell health dashboards

---

## 2. Folder Structure

```
ev-predictive-analytics/
│
├── Makefile                          # CLI entry points for all pipeline phases
├── README.md                         # One-line project goal
├── requirements.txt                  # Core Python dependencies
├── run.py                            # VCU pipeline runner (Phase 1+2)
├── run_cells.py                      # Cell pipeline CLI (ingest / gold / report)
├── build_master_dataset.py           # Builds analysis/master/ by joining VCU + cell features
│
├── configs/
│   ├── settings.py                   # Global paths (PROJECT_ROOT, all data dirs)
│   ├── gold.yaml                     # Rolling windows, lag config, target definition
│   ├── train_soc.yaml                # SOC model config (LightGBM run)
│   ├── cell.yaml                     # Cell health thresholds (calibrated from data)
│   ├── resample.yaml                 # Resampler config
│   └── trip.yaml                     # Trip segmentation config
│
├── schema/
│   ├── telemetry_schema.yaml         # VCU column names and types
│   ├── ranges.yaml                   # Signal soft/hard range bounds
│   ├── quality_flags.yaml            # VCU quality flag bitmask definitions
│   ├── cell_voltage_schema.yaml      # Cell voltage schema: 5 modules x 36 cells
│   ├── cell_temp_schema.yaml         # Cell temp schema: 5 modules x 18 sensors
│   ├── cell_quality_flags.yaml       # Cell quality bitmask (CELL_FFILLED, PACK_GAP, etc.)
│   ├── signal_classes.yaml           # Signal classification
│   └── units.yaml                    # Physical units reference
│
├── ingestion/
│   ├── ingest.py                     # VCU raw CSV → silver pipeline
│   ├── validators.py                 # Schema/range validation + quality flags
│   ├── faults.py                     # Fault CSV ingestion + binary flag columns
│   ├── resampler.py                  # Resample to 1 Hz
│   ├── trip_segmentor.py             # Trip ID assignment
│   ├── io.py                         # Parquet I/O helpers
│   ├── cell_ingest.py                # Cell voltage + temp CSV → silver_cells
│   └── cell_validators.py            # Cell-specific null/range/flatline validation
│
├── features/
│   ├── pipeline.py                   # VCU gold pipeline (rolling + physics + lag + trip agg)
│   ├── rolling.py                    # Rolling window features (30s/60s/300s/600s)
│   ├── lags.py                       # Lag features (60s/300s/600s)
│   ├── physics.py                    # Physics-derived features (power, efficiency, spread)
│   ├── trip_agg.py                   # Per-trip aggregations
│   ├── utils.py                      # Shared helpers
│   ├── cell_health.py                # Per-cell delta, z-score, outlier features (180 cells)
│   ├── cell_agg.py                   # Module-level and pack-level aggregations
│   └── cell_pipeline.py              # Cell gold pipeline entry point
│
├── training/
│   ├── train_soc_xgb.py              # XGBoost SOC trainer (v1/v2)
│   ├── run_phase3_debug.py           # Phase 3 debug runner
│   ├── eval_per_trip.py              # Per-trip evaluation
│   ├── dataset.py                    # Dataset builder
│   ├── artifacts.py                  # Model artifact saving
│   ├── splitter.py                   # Train/val splitter
│   └── plot_phase3_debug.py          # Plots for debug runs
│
├── scripts/
│   ├── run_day.py                    # Silver pipeline: one day
│   ├── backfill.py                   # Silver pipeline: date range backfill
│   ├── run_gold_day.py               # Gold pipeline: one day
│   ├── backfill_gold.py              # Gold pipeline: date range backfill
│   ├── train_soc.py                  # LightGBM SOC train script
│   ├── eval_soc.py                   # SOC evaluation script
│   ├── model_report.py               # Model report generation
│   └── split_csv_by_day.py           # Utility to split large CSVs by day
│
├── evaluation/
│   └── metrics.py                    # Evaluation metrics (MAE, RMSE, MAPE)
│
├── monitoring/
│   └── drift.py                      # Drift detection: PSI, KS test, mean shift, null rate
│
├── reports/
│   └── cell_health_report.py         # Cell health daily report (per-module status labels)
│
├── analysis/
│   ├── cell_health_baseline.py       # 180-cell degradation scan across all dates
│   └── ev01_cell_vcu_analysis_stage1_to_5.ipynb  # Cell-VCU combined analysis notebook
│
├── dashboard/
│   ├── app.py                        # Dashboard app
│   └── soc_charts.py                 # SOC chart components
│
├── data/
│   ├── silver/                       # Cleaned 1-Hz VCU data (Parquet, partitioned by dt/vehicle)
│   ├── gold/
│   │   ├── window_features/          # Rolling+lag+physics features (Parquet per trip)
│   │   ├── trip_features/            # Per-trip aggregations
│   │   └── daily_stats/              # Daily summary stats (2026-01-15 to 2026-03-17)
│   ├── gold_cells/
│   │   ├── cell_features/            # Per-cell delta/zscore/outlier features
│   │   ├── module_features/          # Per-module voltage+temp aggregations
│   │   └── pack_features/            # Pack-level aggregations + imbalance flags
│   ├── analysis/master/              # Master merged dataset (39 partitions)
│   └── drift_reports/                # Drift JSON reports (2026-03-04 to 2026-03-15)
│
├── models/
│   ├── soc_xgb_baseline.json         # XGBoost baseline model (891 KB)
│   ├── soc_xgb_baseline_eval.json    # Baseline eval report
│   ├── soc_xgb_v2.json               # XGBoost v2 model (1.3 MB)
│   ├── soc_xgb_v2_eval.json          # v2 eval report
│   └── soc_forecast/
│       ├── v1__2026-03-01__e50f8b21/ # LightGBM run v1 (model + drift_baseline + eval)
│       └── v2__2026-03-13__fc0cd5ed/ # LightGBM run v2 (model + drift_baseline + eval)
│
├── battery_analysis_outputs/         # Cell/module voltage & temp summary CSVs + problem lists
├── fault_report_outputs/             # 50+ PNG fault charts
├── telemetry_output/                 # Merged CSV, trip report PDF, high-RPM PNG plots
│
├── raw_telemetry_1_2026-03-11_to_2026-03-19.csv   # 28 MB raw VCU telemetry
├── vehicle_1_cell_voltages (2).csv   # 108 MB raw cell voltage (180 cells)
├── vehicle_1_temperature_sensors (1).csv           # 28 MB raw temp (90 sensors)
├── clean_timestamps.csv              # 88 KB intermediate timestamp file
├── min_max_per_timestamp.csv         # 200 KB per-timestamp voltage min/max
├── detailed_mismatch_table.csv       # 72 KB mismatch analysis table
│
├── EV_Phase3_Dataset_Exploration.ipynb
├── ev01_analysis_final.ipynb
├── ev01_battery_diagnostic_report.ipynb
├── ev01_founder_review_battery_diagnostic.ipynb
├── fault_month_report.ipynb
├── telemetry_merge_trip_graphs_high_rpm_only.ipynb
├── cell.ipynb
├── cell_threshold_tuning_notebook.ipynb
└── test.ipynb
```

---

## 3. Data Files Found

### Raw Input Data

| File | Type | Size | Description |
|---|---|---|---|
| `vehicle_1_cell_voltages (2).csv` | CSV | 108 MB | 180-cell voltage readings (5 modules × 36 cells, columns M1_C1…M5_C36) |
| `vehicle_1_temperature_sensors (1).csv` | CSV | 28 MB | 90-sensor temperature readings (5 modules × 18 sensors, columns M1_T1…M5_T18) |
| `raw_telemetry_1_2026-03-11_to_2026-03-19.csv` | CSV | 28 MB | VCU telemetry: Mar 11–19, 2026 |

### Processed / Intermediate Data

| File | Type | Size | Description |
|---|---|---|---|
| `clean_timestamps.csv` | CSV | 88 KB | Cleaned timestamp reference file |
| `min_max_per_timestamp.csv` | CSV | 200 KB | Per-timestamp voltage min/max lookup |
| `detailed_mismatch_table.csv` | CSV | 72 KB | Timestamp/row mismatch analysis |

### Gold Data Layer

- **`data/gold/daily_stats/`** — Daily stats partitions: `dt=2026-01-15` to `dt=2026-03-17` (one parquet per date per vehicle)
- **`data/gold/trip_features/`** — Trip-level feature partitions: `dt=2026-01-15` to `dt=2026-03-17`
- **`data/gold/window_features/`** — Rolling/lag/physics features per trip (multi-part parquet)

### Master Analysis Dataset

- **`data/analysis/master/`** — 39 date partitions, `dt=2026-01-22` to `dt=2026-03-08`, one parquet per date for `vehicle_id=EV01`
- Total merged rows across all partitions: **~685,000** (based on manifest)
- Date range: **2026-01-22 to 2026-03-08**
- All 39 partitions have `clean_pct = 100%`

### Battery Analysis Output CSVs

| File | Size | Content |
|---|---|---|
| `battery_analysis_outputs/cell_voltage_summary.csv` | 24 KB | Per-cell voltage statistics |
| `battery_analysis_outputs/cell_temp_summary.csv` | 14 KB | Per-sensor temperature statistics |
| `battery_analysis_outputs/module_voltage_summary.csv` | 975 B | Per-module voltage summary |
| `battery_analysis_outputs/module_temp_summary.csv` | 1.1 KB | Per-module temperature summary |
| `battery_analysis_outputs/top_voltage_problem_cells.csv` | 3 KB | Ranked list of worst voltage cells |
| `battery_analysis_outputs/top_temp_problem_sensors.csv` | 3.4 KB | Ranked list of hottest temperature sensors |

---

## 4. Dataset Creation Work Completed

### VCU Silver Pipeline
- **Script**: `ingestion/ingest.py` + `scripts/run_day.py` / `scripts/backfill.py`
- **What it does**: Reads raw VCU CSVs, renames 30+ columns to canonical names, validates schema, parses timestamps to UTC, applies soft/hard range checks, adds quality flag bitmask, resamples to 1 Hz, assigns trip IDs, merges fault flags, writes to `data/silver/dt={date}/vehicle_id=EV01.parquet`

### VCU Gold Pipeline
- **Script**: `features/pipeline.py` + `scripts/run_gold_day.py` / `scripts/backfill_gold.py`
- **What it does**: Reads silver, applies quality filter, computes rolling features, physics features, lag features, and trip aggregations per trip group. Writes window features, trip features, and daily stats to `data/gold/`

### Cell Pipeline (Silver → Gold)
- **Scripts**: `run_cells.py` with commands `cell_ingest`, `cell_gold`, `cell_report`
- **What it does**:
  - `cell_ingest`: Reads raw cell voltage (180 columns) and temperature (90 columns) CSVs, validates and applies quality flags, forward-fills dropout gaps, writes `data/silver_cells/`
  - `cell_gold`: Computes per-cell features (delta, z-score, outlier), module-level aggregations, pack-level aggregations. Writes `data/gold_cells/`
  - `cell_report`: Reads gold_cells, assigns module health status labels (Healthy / Monitor / Imbalance Rising / Thermal Hotspot / Degraded Cell / Critical), writes `data/gold_cells/daily_health_reports/`

### Master Dataset Builder
- **Script**: `build_master_dataset.py`
- **What it does**: Finds shared date partitions across VCU silver, cell pack_features, and cell module_features. Deduplicates on timestamp+vehicle_id, performs inner merge on the three sources, adds derived columns (`cell_spread_v`, `abs_current_a`, `current_direction`, `soc_band`, `operating_regime`, `is_clean`), writes to `data/analysis/master/` with a manifest JSON.

---

## 5. Master Dataset Created

| Property | Value |
|---|---|
| **Path** | `data/analysis/master/dt={date}/vehicle_id=EV01.parquet` |
| **Format** | Parquet, partitioned by date |
| **Date Range** | 2026-01-22 to 2026-03-08 |
| **Number of Partitions** | 39 |
| **Total Merged Rows** | ~685,000 |
| **All Partitions Clean** | Yes (clean_pct = 100% for all 39) |
| **Built At** | 2026-03-16 |
| **Build Unit** | Shared partition folder |

### Columns in Master Dataset

**From VCU Silver** (19 columns):
`timestamp`, `vehicle_id`, `trip_id`, `soc_pct`, `battery_current_a`, `stack_voltage_v`, `output_power_kw`, `max_cell_voltage_v`, `min_cell_voltage_v`, `avg_cell_voltage_v`, `avg_battery_temp_c`, `max_battery_temp_c`, `min_battery_temp_c`, `motor_speed_rpm`, `motor_torque_value_nm`, `motor_temperature_c`, `battery_status`, `fault_any`, `vcu_quality_flag`

**From Cell Pack Features** (7 columns):
`cell_quality_flag`, `pack_voltage_range`, `pack_voltage_std`, `worst_cell_id`, `imbalance_flag`, `pack_temp_range`, `hottest_module_id`

**From Cell Module Features** (35 columns across 5 modules):
`module_1_voltage_mean`, `module_1_voltage_std`, `module_1_voltage_range`, `module_1_lowest_cell`, `module_1_temp_mean`, `module_1_temp_range`, `module_1_hottest_sensor` … (same 7 columns × 5 modules)

**Derived Columns** (7 columns):
`dt`, `cell_spread_v`, `abs_current_a`, `current_direction`, `soc_band`, `operating_regime`, `is_clean`

**Target column**: `soc_pct` (source); `y_soc_t_plus_300s` (in gold window_features)

---

## 6. Feature Engineering Already Built

### Rolling Window Features (`features/rolling.py`)
Applied across **4 windows**: 30s, 60s, 300s, 600s with **4 aggregations**: mean, std, min, max.

Base signals: `soc_pct`, `stack_voltage_v`, `battery_current_a`, `output_power_kw`, `motor_speed_rpm`, `dc_side_voltage_v`, `avg_battery_temp_c`, `max_battery_temp_c`, `min_battery_temp_c`, `avg_cell_voltage_v`, `max_cell_voltage_v`, `min_cell_voltage_v`, `motor_temperature_c`, `mcu_temperature_c`, `radiator_temperature_c`, `total_kwh_consumed`, `last_trip_kwh`

Examples: `soc_pct_roll_mean_60s`, `battery_current_a_roll_std_300s`, `motor_speed_rpm_roll_max_60s`

### Lag Features (`features/lags.py`)
Lag offsets: **60s, 300s, 600s** applied to the same base signals.

Examples: `soc_pct_lag_60s`, `soc_pct_lag_300s`, `battery_current_a_lag_300s`

### Physics-Derived Features (`features/physics.py`)
| Feature | Formula |
|---|---|
| `elec_power_kw_proxy` | stack_voltage_v × battery_current_a / 1000 |
| `power_proxy_error_kw` | output_power_kw − elec_power_kw_proxy |
| `power_proxy_ratio` | output_power_kw / elec_power_kw_proxy (safe, clipped) |
| `cell_voltage_delta_v` | max_cell_voltage_v − min_cell_voltage_v |
| `cell_voltage_delta_norm` | cell_voltage_delta_v / avg_cell_voltage_v |
| `battery_temp_delta_c` | max_battery_temp_c − min_battery_temp_c |
| `battery_temp_delta_norm` | battery_temp_delta_c / avg_battery_temp_c |
| `mech_power_kw_proxy` | (motor_torque × ω) / 1000 |
| `eff_proxy` | mech_power / elec_power |
| `motor_minus_radiator_temp_c` | motor_temperature_c − radiator_temperature_c |
| `dcdc_input_kw_proxy` | dcdc_input_v × dcdc_input_a / 1000 |
| `dcdc_output_kw_proxy` | dcdc_output_v × dcdc_output_a / 1000 |

### Cell-Level Features (`features/cell_health.py`)
For each of the **180 cells** (M1_C1 to M5_C36), per timestamp:
- `{cell}_delta` — deviation from module mean voltage
- `{cell}_zscore` — z-score within module (threshold: 2.5)
- `{cell}_is_outlier` — binary flag if |zscore| > 2.5

### Module-Level Aggregations (`features/cell_agg.py`)
For each of **5 modules**, per timestamp:
- `module_{m}_voltage_mean`, `_std`, `_range`, `_lowest_cell`
- `module_{m}_temp_mean`, `_temp_range`, `_hottest_sensor`

### Pack-Level Aggregations (`features/cell_agg.py`)
- `pack_voltage_range` — max−min across all 180 cells
- `pack_voltage_std` — std across all 180 cells
- `pack_temp_range` — max−min across all 90 sensors
- `worst_cell_id` — column name of lowest voltage cell
- `hottest_module_id` — module number with highest mean temp
- `imbalance_flag` — 1 if pack_voltage_range > 130mV (p99 threshold)

### Master Dataset Derived Features (`build_master_dataset.py`)
- `cell_spread_v` — max_cell_voltage_v − min_cell_voltage_v (pack spread)
- `abs_current_a` — |battery_current_a|
- `current_direction` — categorical: charge / idle / discharge (thresholds: ±10 A)
- `soc_band` — categorical: 0-20 / 20-40 / 40-60 / 60-80 / 80-100
- `operating_regime` — categorical: heavy_regen / light_regen / idle / light_discharge / heavy_discharge (thresholds: ±10 A idle, ±100 A heavy)
- `is_clean` — boolean: cell_quality_flag passes PACK_GAP + CELL_HARD_BREACH mask

### SOC Model Feature Set (59 features in XGBoost v2)
Grouped by signal family:
- **SOC** (8): current value + 3 lags + 4 rolling stats
- **Current** (8): current value + 2 lags + 6 rolling stats
- **Voltage** (6): stack_voltage + 2 lags + 3 rolling stats
- **Cell** (6): max/min cell voltage + spread + norm + rolling means
- **Power** (8): output_power + 2 lags + 4 rolling + elec_proxy
- **Motor speed** (6): RPM + lag + 4 rolling stats
- **Battery temp** (7): avg_temp + 2 lags + rolling + spread + norm
- **Motor temp** (4): motor_temp + lag + rolling mean + rolling std
- **Energy** (3): total_kwh + 2 lags
- **Other** (3): `fault_any`, `power_proxy_error_kw`, `power_proxy_ratio`

---

## 7. SOC Prediction Work Completed

### XGBoost Baseline Model
| Property | Value |
|---|---|
| **File** | `models/soc_xgb_baseline.json` (891 KB) |
| **Eval report** | `models/soc_xgb_baseline_eval.json` |
| **Train date** | 2026-02-23 |
| **Val date** | 2026-02-25 |
| **Train rows** | 174,131 |
| **Val rows** | 762 |
| **Target** | `y_soc_t_plus_300s` (SOC 5 minutes ahead) |
| **n_features** | 59 |
| **Best iteration** | 92 |
| **Train MAE** | 0.352% SOC |
| **Val MAE** | 0.599% SOC |
| **Val RMSE** | 0.820% SOC |
| **Val MAPE** | 0.733% |

### XGBoost v2 Model
| Property | Value |
|---|---|
| **File** | `models/soc_xgb_v2.json` (1.3 MB) |
| **Eval report** | `models/soc_xgb_v2_eval.json` |
| **Train dates** | 2026-02-23, 24, 25, 26 |
| **Val dates** | 2026-03-01 |
| **Train rows** | 176,949 |
| **Val rows** | 6,768 |
| **Config** | 500 estimators, max_depth=6, lr=0.05, early_stopping=30 |
| **Train MAE** | 0.265% SOC |
| **Val MAE** | 0.516% SOC |
| **Val RMSE** | 0.697% SOC |

### LightGBM v1 Run (soc_forecast)
| Property | Value |
|---|---|
| **Directory** | `models/soc_forecast/v1__2026-03-01__e50f8b21/` |
| **Files** | `model.json` (1.3 MB), `eval_report.json`, `drift_baseline.parquet`, `data_fingerprint.json`, `feature_set.json`, `config.yaml` |

### LightGBM v2 Run (soc_forecast) — Current Best
| Property | Value |
|---|---|
| **Directory** | `models/soc_forecast/v2__2026-03-13__fc0cd5ed/` |
| **Files** | `model.json` (1.4 MB), `eval_report.json`, `drift_baseline.parquet`, `data_fingerprint.json`, `feature_set.json`, `config.yaml` |
| **Status** | EXCELLENT |
| **Target** | `y_soc_t_plus_300s` (5-minute ahead SOC %) |
| **Train MAE** | 0.306% SOC |
| **Val MAE** | 0.441% SOC |
| **Val RMSE** | 0.563% SOC |
| **Train-Val Gap** | 0.135 |
| **Persistence baseline MAE** | 0.822% (model beats by 46%) |
| **Rolling baseline MAE** | 0.909% |
| **Best iteration** | 180 |
| **Per-trip val** | 8 trips evaluated; best trip MAE = 0.336%, worst = 0.597% |
| **SOC bucket 60–80%** | MAE = 0.439% |
| **SOC bucket 80–100%** | MAE = 0.443% |
| **Max abs error** | 1.933% SOC |
| **Error distribution** | mean = −0.008, std = 0.563 (near-unbiased) |

**Top features** (v2 LightGBM):
1. `soc_pct` (49.97%)
2. `soc_pct_roll_mean_60s` (34.41%)
3. `soc_pct_lag_60s` (11.15%)
4. `stack_voltage_v` (0.84%)
5. `output_power_kw` (0.66%)

---

## 8. Other Battery Analytics Completed

### Fault Analysis and Visualization
- **Notebook**: `fault_month_report.ipynb` (12 MB — extensive)
- **Output**: 50+ PNG charts in `fault_report_outputs/`, one chart per fault event
- **Fault types detected**: busbar_undervoltage_fault, bus_overvoltage_fault, hardware_overvoltage_fault, total_hardware_failure, ac_hall_failure, module_over_temperature_warning, temperature_difference_failure, low_voltage_undervoltage_fault, software_overcurrent_fault
- **Code**: `ingestion/faults.py` — binary fault flag columns added to every silver row using cumulative diff algorithm on start/end timestamps

### Battery Current, Voltage, Temperature Analysis
- **Notebook**: `ev01_analysis_final.ipynb` (1 MB)
- **Notebook**: `ev01_founder_review_battery_diagnostic.ipynb` (2.5 MB) — Founder-review diagnostic
- **Notebook**: `ev01_battery_diagnostic_report.ipynb`
- **Output**: `battery_analysis_outputs/` with cell_voltage_summary.csv, cell_temp_summary.csv, module summaries, top problem cells and sensors

### Cell-Level Analysis (180-cell scan)
- **Script**: `analysis/cell_health_baseline.py`
- **Purpose**: Full 180-cell degradation scan on clean master + cell_features data, identifies chronically weak cells, tracks degradation over time
- **Notebook**: `analysis/ev01_cell_vcu_analysis_stage1_to_5.ipynb` — Staged cell + VCU combined analysis

### Cell Health Reports (Module Status Labels)
- **Script**: `reports/cell_health_report.py`
- **Output**: `data/gold_cells/daily_health_reports/` — per-module, per-day status
- **Module status labels**: Healthy, Monitor, Imbalance Rising, Thermal Hotspot, Degraded Cell, Critical
- **Metrics per module per day**: `avg_voltage_std`, `max_voltage_range`, `avg_temp_range`, `max_temp_spike`, `worst_cell_id`, `pct_time_imbalanced`, `trend_direction`
- **Cell config** (calibrated thresholds in `configs/cell.yaml`):
  - Voltage imbalance warn: 90 mV (p90), alert: 130 mV (p99)
  - Temperature hotspot delta: 15°C (p95)
  - Flatline window: 1800s (30 min)
  - Outlier z-score: 2.5

### Trip Analysis and High-RPM Telemetry
- **Notebook**: `telemetry_merge_trip_graphs_high_rpm_only.ipynb`
- **Output** in `telemetry_output/`:
  - `merged_sorted_trip_segmented.csv` (23 MB) — merged multi-file telemetry with trip IDs
  - `telemetry_trip_report.pdf` (1.4 MB) — full trip report PDF
  - `trip_summary.csv` (6.3 KB) — trip-level summary table
  - `all_data_rpm_vs_battery_current_high_rpm.png` — high-RPM scatter
  - `all_data_rpm_vs_motor_ac_current_high_rpm.png` — motor AC current at high RPM
  - `plots_high_rpm_only/` — per-trip high-RPM plots (multiple sub-folders by date)

### Dataset Exploration (Phase 3)
- **Notebook**: `EV_Phase3_Dataset_Exploration.ipynb` (529 KB) — Explores gold dataset distributions, feature statistics, label availability

### Drift Monitoring (Production)
- **Script**: `monitoring/drift.py`
- **Metrics**: PSI (Population Stability Index), Kolmogorov-Smirnov test, mean shift %, null rate change
- **PSI thresholds**: < 0.1 = STABLE, 0.1–0.2 = MONITOR, > 0.2 = DRIFT
- **Reports generated** in `data/drift_reports/`:
  - `dt=2026-03-04.json` + `_eval.json`
  - `dt=2026-03-08.json` + `_eval.json`
  - `dt=2026-03-10.json` + `_eval.json`
  - `dt=2026-03-11.json` + `_eval.json`
  - `dt=2026-03-12.json` + `_eval.json`
  - `dt=2026-03-13.json` + `_eval.json`
  - `dt=2026-03-14.json` + `_eval.json`
  - `dt=2026-03-15.json` + `_eval.json`
- Each model run folder contains a `drift_baseline.parquet` with stored reference distribution stats (mean, std, p10, p25, p50, p75, p90, null_rate) per feature.

---

## 9. Scripts and How They Are Run

### VCU Pipeline (Phase 1 — Silver)
```bash
# Single day
python -m scripts.run_day --dt 2026-03-10

# Date range backfill
python -m scripts.backfill --start 2026-01-15 --end 2026-03-17
```

### VCU Pipeline (Phase 2 — Gold)
```bash
# Single day
python -m scripts.run_gold_day --dt 2026-03-10
python -m scripts.run_gold_day --dt 2026-03-10 --vehicle_id EV01

# Date range backfill
python -m scripts.backfill_gold --start 2026-01-15 --end 2026-03-17
```

### Cell Pipeline
```bash
# Single day — all three stages
python run_cells.py cell_ingest --dt 2026-03-10
python run_cells.py cell_gold   --dt 2026-03-10
python run_cells.py cell_report --dt 2026-03-10

# Backfill
python run_cells.py cell_ingest --backfill --start 2026-01-15 --end 2026-03-08
python run_cells.py cell_gold   --backfill --start 2026-01-15 --end 2026-03-08
python run_cells.py cell_report --backfill --start 2026-01-15 --end 2026-03-08
```

### Master Dataset Build
```bash
python build_master_dataset.py
```

### SOC Training
```bash
# XGBoost script
python training/train_soc_xgb.py

# LightGBM via Makefile
make train dt=2026-03-13 vehicle_id=EV01
# equivalent to:
python -m scripts.train_soc --dt 2026-03-13 --vehicle_id EV01
```

### SOC Evaluation
```bash
make eval dt=2026-03-13 vehicle_id=EV01
# equivalent to:
python -m scripts.eval_soc --dt 2026-03-13 --vehicle_id EV01
```

### Model Report
```bash
make report dt=2026-03-13 vehicle=EV01 model=soc_forecast
```

### Testing
```bash
make test
# equivalent to: pytest -q
```

### CSV Splitting Utility
```bash
python scripts/split_csv_by_day.py
```

---

## 10. Outputs Generated

### Trained Models
| File | Size | Description |
|---|---|---|
| `models/soc_xgb_baseline.json` | 891 KB | XGBoost baseline (92 trees) |
| `models/soc_xgb_v2.json` | 1.3 MB | XGBoost v2 (500 trees, 4-day train) |
| `models/soc_forecast/v1__2026-03-01__e50f8b21/model.json` | 1.3 MB | LightGBM v1 |
| `models/soc_forecast/v2__2026-03-13__fc0cd5ed/model.json` | 1.4 MB | LightGBM v2 (current best) |

### Eval Reports
| File | Description |
|---|---|
| `models/soc_xgb_baseline_eval.json` | Baseline metrics: val MAE=0.599% |
| `models/soc_xgb_v2_eval.json` | v2 metrics: val MAE=0.516%, top 20 features |
| `models/soc_forecast/v2__*/eval_report.json` | Full eval: EXCELLENT, MAE=0.441%, per-trip, SOC buckets, error distribution, worst predictions |

### Drift Reports
- 8 drift report pairs (`dt=YYYY-MM-DD.json` + `_eval.json`) in `data/drift_reports/` covering 2026-03-04 to 2026-03-15
- Each report: per-feature PSI, KS p-value, mean shift %, null rate change, overall status (STABLE/MONITOR/DRIFT)

### Drift Baselines
- `models/soc_forecast/v1__*/drift_baseline.parquet` — stored reference distribution for v1
- `models/soc_forecast/v2__*/drift_baseline.parquet` — stored reference distribution for v2

### Master Dataset
- 39 Parquet partitions in `data/analysis/master/` + `manifest.json`

### Gold Data
- `data/gold/window_features/` — per-trip multi-part Parquet (rolling+lag+physics features)
- `data/gold/trip_features/` — trip-level aggregations
- `data/gold/daily_stats/` — daily summary stats, 50 partitions (2026-01-15 to 2026-03-17)

### Cell Gold Data
- `data/gold_cells/cell_features/` — per-cell delta/zscore/outlier (180 features per row)
- `data/gold_cells/module_features/` — module aggregations
- `data/gold_cells/pack_features/` — pack aggregations + imbalance flags
- `data/gold_cells/daily_health_reports/` — per-module daily health status

### Battery Analysis CSVs
- `battery_analysis_outputs/cell_voltage_summary.csv` (24 KB)
- `battery_analysis_outputs/cell_temp_summary.csv` (14 KB)
- `battery_analysis_outputs/module_voltage_summary.csv` (975 B)
- `battery_analysis_outputs/module_temp_summary.csv` (1.1 KB)
- `battery_analysis_outputs/top_voltage_problem_cells.csv` (3 KB)
- `battery_analysis_outputs/top_temp_problem_sensors.csv` (3.4 KB)

### Charts and Reports
- `fault_report_outputs/` — 50+ PNG charts per fault event
- `telemetry_output/telemetry_trip_report.pdf` — 1.4 MB trip report PDF
- `telemetry_output/trip_summary.csv` — trip-level summary
- `telemetry_output/all_data_rpm_vs_battery_current_high_rpm.png`
- `telemetry_output/all_data_rpm_vs_motor_ac_current_high_rpm.png`
- `telemetry_output/plots_high_rpm_only/` — per-trip high-RPM plots

### Intermediate Files
- `clean_timestamps.csv` (88 KB)
- `min_max_per_timestamp.csv` (200 KB)
- `detailed_mismatch_table.csv` (72 KB)
- `telemetry_output/merged_sorted_trip_segmented.csv` (23 MB)

---

## 11. Libraries and Tech Used

Based on imports across all source files:

| Library | Use |
|---|---|
| `pandas` 2.2.2 | All data manipulation |
| `numpy` 1.26.4 | Numeric computation, bitmask ops |
| `pyarrow` 16.1.0 | Parquet read/write |
| `xgboost` | SOC regression (XGBoost baseline and v2) |
| `lightgbm` | SOC regression (soc_forecast v1 and v2) |
| `scikit-learn` | MAE, RMSE metrics |
| `scipy.stats` | KS test in drift monitoring |
| `PyYAML` 6.0.1 | Config/schema loading |
| `python-dateutil` 2.9 | Date parsing |
| `tqdm` 4.66.4 | Progress bars |
| `pytest` 8.2.2 | Testing |
| `matplotlib` | Fault and telemetry charts (from notebook outputs) |

---

## 12. What I Can Share With ChatGPT

**Best files to copy-paste or describe:**

1. **`configs/gold.yaml`** — Defines all feature engineering parameters (rolling windows, lags, target definition)
2. **`models/soc_forecast/v2__2026-03-13__fc0cd5ed/eval_report.json`** — Full SOC model evaluation (EXCELLENT, MAE=0.441%, baselines, per-trip, error distribution)
3. **`data/analysis/master/manifest.json`** — Shows master dataset structure, date range, column lists, partition row counts
4. **`models/soc_xgb_v2_eval.json`** — XGBoost v2 metrics and top 20 feature importances
5. **`build_master_dataset.py`** (header/config section) — Shows exactly what columns are in the master dataset
6. **`training/train_soc_xgb.py`** (FEATURES list, lines 41–93) — The exact 59 features used for SOC training
7. **`configs/cell.yaml`** — Calibrated cell health thresholds
8. **`schema/cell_voltage_schema.yaml`** — Battery pack structure (5 modules × 36 cells = 180)
9. **`schema/cell_quality_flags.yaml`** — Quality bitmask definitions
10. **`monitoring/drift.py`** (function signatures) — Drift monitoring implementation overview
