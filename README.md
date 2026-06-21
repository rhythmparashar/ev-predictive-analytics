# EV Predictive Analytics

End-to-end battery intelligence and SOC forecasting engine for electric construction equipment — built on real CAN/BMS telemetry from a 2-vehicle fleet.

---

## What It Does

| Capability | Detail |
|---|---|
| **SOC Forecasting** | XGBoost model predicts State of Charge 5 minutes ahead; **0.54% MAE** on held-out data vs 1.78% persistence baseline |
| **Battery Health Scoring** | Rule-based engine scores each timestamp 0–100 across 7 penalty dimensions (thermal, voltage imbalance, current stress, weak module, faults, data quality) |
| **Automated Daily Pipeline** | Raw CAN data → silver → gold → predictions → health scoring → PDF/email report, fully automated with state tracking |
| **Drift Monitoring** | PSI + KS test across 61 features per day; alerts when feature distributions shift vs training baseline |
| **Cell-Level Intelligence** | Per-cell voltage gap tracking (180 cells per pack), degradation trend detection, module-level thermal hotspot analysis |

---

## Architecture

```
data/raw_parquet/              ← raw CAN telemetry (1 folder per day per vehicle)
data/raw_cell_voltage/         ← per-cell voltage (180 cells × 5 modules)
data/raw_cell_temp/            ← per-cell temperature
data/raw_faults/               ← motor-controller fault events
        │
        ▼  ingestion/  (validate → 1Hz resample → trip segment → quality flags)
data/silver/                   ← clean 1Hz telemetry, trip-labelled
data/silver_cells/             ← merged cell voltage + temperature
        │
        ▼  features/  (rolling windows, lag features, physics-derived)
data/gold/window_features/     ← 61-feature ML windows  ←── XGBoost SOC model
data/gold/trip_features/       ← trip-level aggregates
data/gold_cells/               ← per-cell gap, outlier, module health labels
        │
        ▼  build_master_dataset.py
data/analysis/master/          ← merged telemetry + cell + module dataset
        │
        ▼  battery_intelligence/
outputs/battery_health/        ← health scores + alerts per day
outputs/soc_scores/            ← SOC predictions per day
reports/                       ← auto-generated PDF and markdown reports
```

---

## Key Results

### SOC Forecast Model (v4 — XGBoost, 61 features)

| Metric | Value |
|---|---|
| Validation MAE | **0.54%** |
| Validation RMSE | 0.80% |
| Persistence baseline MAE | 1.78% |
| Improvement vs baseline | **3.3×** |
| Train dates | 25 days (balanced pre/post firmware) |
| Val dates | 4 held-out days (May 29 – Jun 1) |

Top features by importance: `soc_pct` (50%), `soc_pct_roll_mean_60s` (29%), `soc_pct_lag_60s` (15%), `stack_voltage_v`, `output_power_kw`

### Battery Health Engine

| Metric | Value |
|---|---|
| Rows scored | 1,643,119 |
| Date range | Jan 22 – May 17, 2026 |
| Mean health score | 72.7 / 100 |
| Alert events detected | 1,674,976 |
| Alert types | 12 |

Most frequent alerts: `WEAK_MODULE_THERMAL` (392K events), `TEMP_IMBALANCE_WARNING` (240K), `HIGH_CURRENT_STRESS` (237K)

### Notable Finding — Firmware Update Impact

A firmware update on **March 27–30, 2026** caused cell M1_C25 voltage gap to drop from **−43 mV → −4 mV** overnight. v2 model (trained only on pre-April data) showed MAE degrading to **1.53%** on post-April data — triggering a retrain. v3/v4 were trained on balanced pre+post data.

---

## Project Structure

```
ev-predictive-analytics/
├── ingestion/               # Raw → silver (resample, validate, trip segment, faults)
├── features/                # Silver → gold (rolling, lag, physics features, cell pipeline)
├── battery_intelligence/    # Health scoring, rule engine, cell intelligence, DuckDB loaders
├── ml/                      # XGBoost training wrapper, drift, metrics, model registry
├── monitoring/              # Production drift monitoring (PSI, KS test, mean shift)
├── scripts/                 # Runnable pipeline scripts
│   ├── run_daily_pipeline.py    # Full automated daily run
│   ├── run_day.py               # Process one day raw → silver
│   ├── run_gold_day.py          # Build gold features for one day
│   ├── train_soc_v3.py          # Train SOC model (current)
│   └── eval_soc_post_april.py   # Pre/post firmware eval
├── configs/                 # settings.py (all paths), yaml configs per pipeline stage
├── schema/                  # YAML schemas for telemetry validation
├── docs/                    # Architecture docs, model cards, design notes
├── notebooks/               # Exploratory analysis and diagnostic reports
├── tasks/soc_forecast/      # Task definition (config, feature set, train/eval)
├── models/soc_forecast/     # Versioned model artifacts (model.json, drift baseline, eval)
├── evaluation/              # Standalone metrics module
└── monitoring/              # Drift monitoring
```

---

## Quick Start

**Requirements:** Python 3.10+, `venv/bin/python3`

```bash
# Install dependencies
python -m venv venv
pip install -r requirements.txt

# Process one raw day through silver
python scripts/run_day.py --dt 2026-05-01

# Build gold features for that day
python scripts/run_gold_day.py --dt 2026-05-01

# Run the full automated daily pipeline (processes all new dates)
python scripts/run_daily_pipeline.py

# Train a new SOC model
venv/bin/python3 scripts/train_soc_v3.py

# Run battery health engine
python scripts/run_battery_engine_module1.py
```

**First-time setup** (mark historical dates as already processed):
```bash
python scripts/run_daily_pipeline.py --mark-existing
```

---

## Fleet

| Vehicle | Type | Data Range | Battery |
|---|---|---|---|
| EV01 | Loader | Jan 15 – May 17, 2026 | 5 modules × 36 cells = 180 cells |
| EV02 | Excavator | Jan 15 – May 14, 2026 | 5 modules × 36 cells = 180 cells |

SOC models are **per machine type** — Loader and Excavator have different duty cycles and require separate models.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, PyArrow, DuckDB |
| ML | XGBoost, scikit-learn, SHAP |
| Drift monitoring | scipy (KS test), PSI |
| Experiment tracking | MLflow |
| Reporting | reportlab (PDF), plotly, seaborn |
| Cloud sync | AWS S3 (boto3) |
| Pipeline orchestration | Custom state-machine (`data/state/pipeline_state.json`) |

---

## Model Versions

| Version | Date | Val MAE | Notes |
|---|---|---|---|
| v1 | Mar 1, 2026 | — | First baseline |
| v2 | Mar 13, 2026 | 0.44% (train-time val) | Pre-April data only; degraded to 1.53% post-firmware |
| v3 | May 20, 2026 | 0.51% | Balanced pre+post firmware data |
| **v4** | Jun 2, 2026 | **0.54%** | **Current production** — 25 training days, 4-day holdout |
