# EV Telemetry ML — Phase 1: Data Foundation
> **Status: Implemented and Verified**  
> The ingestion and silver pipeline is stable, deterministic, and reproducible.  
> Phase 2 Gold feature engineering is implemented and verified.

---

## What Phase 1 Currently Builds

```
Raw CSV files
      ↓
Column Standardization
      ↓
Validation + Quality Flags
      ↓
1 Hz Resampling
      ↓
Fault Alignment
      ↓
Trip Segmentation
      ↓
Silver Parquet
      ↓
Gold Feature Engineering
      ↓
Window Features + Trip Features + Daily Stats
```

**Currently implemented outputs:**
```
raw CSV
 → raw_parquet
 → silver parquet (trip + fault labeled)
 → gold window features
 → gold trip features
 → gold daily stats
 → reports/manifest
```

---

## Full Structure

```
ev-telemetry-ml/
│
├── ingestion/
│   ├── ingest.py
│   ├── validators.py
│   ├── resampler.py
│   ├── faults.py
│   ├── trip_segmentor.py
│   ├── io.py
│   └── tests/
│       ├── test_validators.py
│       ├── test_resampler.py
│       ├── test_trip_segmentor.py
│       └── test_io.py
│
├── features/
│   ├── rolling.py
│   ├── physics.py
│   ├── lags.py
│   ├── trip_agg.py
│   ├── pipeline.py
│   └── tests/
│       ├── test_rolling.py
│       ├── test_physics.py
│       └── test_pipeline.py
│
├── data/                       ← never committed to git
│   ├── raw/
│   │   └── dt=2026-02-24/
│   │       ├── vehicle_id=EV01.csv
│   │       └── vehicle_id=EV02.csv
│   ├── raw_parquet/
│   │   └── dt=2026-02-24/
│   │       ├── vehicle_id=EV01.parquet
│   │       └── vehicle_id=EV02.parquet
│   ├── silver/
│   │   └── dt=2026-02-24/
│   │       ├── vehicle_id=EV01.parquet
│   │       └── vehicle_id=EV02.parquet
│   ├── gold/
│   │   ├── window_features/    ← 1 Hz ML feature dataset
│   │   ├── trip_features/      ← one row per trip
│   │   └── daily_stats/        ← one row per vehicle per day
│   ├── state/
│   │   └── open_trips.parquet
│   ├── reports/
│   │   └── dt=2026-02-24.json
│   └── samples/
│       ├── sample_raw.csv
│       └── sample_expected_silver.parquet
│
├── schema/
│   ├── telemetry_schema.yaml
│   ├── ranges.yaml
│   ├── units.yaml
│   ├── signal_classes.yaml
│   └── quality_flags.yaml
│
├── configs/
│   ├── settings.py
│   ├── resample.yaml
│   └── trip.yaml
│
├── scripts/
│   ├── run_day.py
│   └── backfill.py
│
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_check.ipynb
│
├── tests/
│   └── test_end_to_end.py
│
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

---

## `ingestion/`

The daily ETL engine. Produces the trusted Silver dataset from raw CSVs.

---

### `ingest.py`
Main pipeline entry point for one vehicle, one date.

**Flow:**
```
CSV
 → Standardize column names
 → Convert HH:MM:SS timestamps → seconds → UTC datetime
 → Add vehicle_id
 → Validate telemetry (sets quality_flag)
 → Write raw_parquet
 → Resample to 1 Hz
 → Align hardware faults
 → Assign trip_id
 → Write silver
 → Return manifest dict
```

---

### `validators.py`
Validates raw telemetry before resampling. Sets `quality_flag` bitmask on every row.

| Check | Description |
|---|---|
| Required columns | Must exist per `schema/telemetry_schema.yaml` |
| Timestamp parsing | Converts to UTC datetime; flags failures |
| Duplicate timestamps | Reported in manifest; allowed (2 Hz telemetry) |
| Range validation | Soft breach → flag; Hard breach → flag + exclude |
| Null rates | Reported per signal in manifest |

---

### `resampler.py`
Converts telemetry to a fixed 1 Hz timeline.

**Steps:**
1. Sort by timestamp
2. Aggregate duplicate timestamps (2 Hz → 1 Hz via mean)
3. Create full 1 Hz index from first to last timestamp
4. Reindex and fill by signal class

| Signal Class | Fill Strategy |
|---|---|
| `fast` | Time interpolation |
| `slow` | Forward fill |
| `status` | Forward fill (last known value) |

**Gap enforcement:** Gaps longer than `max_gap_s` (per `resample.yaml`) are not filled — rows are inserted with `GAP_INSERTED` flag and null values.

**Quality flags set by resampler:**
- `GAP_INSERTED` — missing source row
- `INTERPOLATED_FAST` — fast signal gap filled by interpolation
- `FORWARD_FILLED_SLOW` — slow/status signal gap filled by forward fill

---

### `trip_segmentor.py`
Detects driving trips and assigns a unique `trip_id` to every row.

**Trip Start condition:**
```
MCU Enable State == ON
AND (motor_speed > threshold OR output_power > threshold)
```

**Trip End condition:**
```
MCU Enable State == OFF
AND motor_speed < threshold
AND output_power < threshold
```

**Additional logic:**
- Start debounce — avoids false starts from brief MCU enable spikes
- Idle gap detection — ends trip if vehicle idles beyond `idle_gap_s`
- Long telemetry gap detection — ends trip if data gap exceeds threshold
- Cross-day trips — open trips carried forward via state file
- Minimum trip duration — trips shorter than `min_trip_duration_s` discarded

**Trip ID format:**
```
EV01_000001
EV01_000002
```

**State file** (`data/state/open_trips.parquet`):

| Column | Description |
|---|---|
| `vehicle_id` | One row per vehicle |
| `trip_seq` | Incrementing trip counter |
| `open_trip_id` | Trip ID in progress |
| `open_trip_start_ts` | Timestamp trip started |

---

### `faults.py`
Aligns hardware fault logs with telemetry timestamps and produces binary fault indicator columns in the Silver dataset.

Fault logs are provided as a separate CSV alongside telemetry, containing:

```
Activated At
Fixed At
Code
Description
```

Each row describes a time window where a hardware fault was active on the vehicle.

**Fault alignment runs after resampling and before trip segmentation.**

For each fault window, every telemetry row where `activated_at ≤ timestamp ≤ fixed_at` receives:
```
fault_<fault_code> = 1
```
All other rows receive:
```
fault_<fault_code> = 0
```

**Fault columns produced:**

| Column | Fault |
|---|---|
| `fault_busbar_undervoltage_fault` | Bus bar undervoltage |
| `fault_bus_overvoltage_fault` | Bus overvoltage |
| `fault_hardware_overvoltage_fault` | Hardware overvoltage |
| `fault_total_hardware_failure` | Total hardware failure |
| `fault_ac_hall_failure` | AC Hall sensor failure |
| `fault_module_over_temperature_warning` | Module over-temperature |
| `fault_temperature_difference_failure` | Temperature difference failure |
| `fault_low_voltage_undervoltage_fault` | Low voltage undervoltage |
| `fault_software_overcurrent_fault` | Software overcurrent |

**Aggregate indicator:**

```python
fault_any = max(all fault_* columns)  # 1 if any fault active, 0 otherwise
```

| `fault_any` | Meaning |
|---|---|
| `0` | No fault active at this timestamp |
| `1` | At least one fault active at this timestamp |

**Why fault alignment belongs in ingestion (not features):**
Faults are labels, not features. Aligning them here means Silver is the single trusted dataset for both training and evaluation. No downstream code needs to re-join fault logs — the alignment is done once, deterministically, at ingest time.

---

### `io.py`
All file I/O in one place. No other module reads or writes files directly.

**Functions:**
```python
list_raw_csvs(dt)              # Find all CSVs for a date
vehicle_id_from_filename()     # Parse vehicle_id from filename
read_csv()                     # Read with correct dtypes
parquet_path()                 # Build partition path: dt= / vehicle_id=
write_parquet_atomic()         # Write to tmp → rename (never leaves corrupt files)
```

**Atomic write pattern:**
```
write → data/.tmp/file.parquet
rename → data/silver/dt=.../vehicle_id=EV01.parquet
```
If the run fails mid-write, no partial file is left in the output directory.

---

## `features/`

**Status: Implemented and Verified**

The features layer converts Silver telemetry into ML-ready Gold datasets. The `features/` folder is designed as a shared library used by training pipelines, offline experiments, and future model serving. Silver is never modified — Gold is derived deterministically.

| File | Purpose |
|---|---|
| `rolling.py` | Rolling mean/std/min/max over 30s, 1min, 5min, 10min windows |
| `physics.py` | Physics-derived variables: voltage imbalance, thermal gradient, efficiency proxies |
| `lags.py` | Lag features at t−1min, t−5min, t−10min |
| `trip_agg.py` | One row per trip: duration, SOC change, energy use |
| `pipeline.py` | Silver → Gold feature pipeline; applies quality filter `(quality_flag & (16|32)) == 0` |

---

## `data/`

Never committed to git (except `data/samples/`).

| Folder | Contents | Produced by |
|---|---|---|
| `raw/` | Original CSVs, never modified. Drop zone only. | Manual / upstream transfer |
| `raw_parquet/` | Typed parquet. Schema enforced. Column names standardized. | `ingest.py` |
| `silver/` | Clean + resampled + `trip_id` + `quality_flag` + `fault_*` columns. The trusted dataset. | `ingest.py` |
| `gold/` | ML-ready datasets. `window_features/` (1 Hz feature rows), `trip_features/` (one row per trip), `daily_stats/` (one row per vehicle per day). | `features/pipeline.py` |
| `state/` | `open_trips.parquet` — 1 row per vehicle mid-trip at end of day. | `trip_segmentor.py` |
| `reports/` | `dt=YYYY-MM-DD.json` — run manifest with row counts, quality stats, trip counts. | `run_day.py` |
| `samples/` | Tiny anonymized fixtures for tests. Only data committed to git. | Hand-crafted |

---

## `schema/`

Single source of truth for all signal definitions. Every module imports from here. Nothing is hardcoded elsewhere.

| File | Defines |
|---|---|
| `telemetry_schema.yaml` | Every column name, dtype, required flag |
| `ranges.yaml` | Per-signal hard gate and soft flag thresholds |
| `units.yaml` | Unit for every signal (V, A, °C, kW, Nm, RPM) |
| `signal_classes.yaml` | Every signal as `fast`, `slow`, or `status` — controls resampler fill strategy |
| `quality_flags.yaml` | Bitmask bit definitions — imported by validators, resampler, and trip_segmentor |

**`quality_flag` bitmask:**

| Bit | Value | Meaning |
|---|---|---|
| 0 | 1 | Interpolated (fast signal) |
| 1 | 2 | Forward-filled (slow/status signal) |
| 2 | 4 | Gap inserted (missing source row) |
| 3 | 8 | Soft range breach |
| 4 | 16 | Hard range breach |
| 5 | 32 | Time anomaly (duplicate / reversal) |
| 6 | 64 | Sensor flatline window |

**Model-safe filter (used in Phase 2 `features/pipeline.py`):**
```python
df = df[(df["quality_flag"] & (16 | 32)) == 0]
```

---

## `configs/`

Runtime behavior controlled by config, not code.

### `settings.py`
Loads all paths from environment variables. Every other module imports paths from here.
```python
DATA_DIR, RAW_DIR, RAW_PARQUET_DIR, SILVER_DIR,
GOLD_DIR, STATE_DIR, REPORTS_DIR
```

### `resample.yaml`
```yaml
freq: 1Hz
max_gap_s:
  fast: 5
  slow: 60
  status: 300
```

### `trip.yaml`
```yaml
debounce_s: 5
idle_gap_s: 120
min_trip_duration_s: 60
speed_threshold: 2.0
power_threshold_kw: 0.5
```

---

## `scripts/`

### `run_day.py`
One command runs the full pipeline for one date.

```bash
make run dt=2026-02-24
```

**Steps executed:**
1. Find all CSVs in `data/raw/dt=2026-02-24/`
2. For each vehicle: ingest → validate → resample → fault align → trip segment
3. Write `raw_parquet` and `silver`
4. Write manifest to `data/reports/dt=2026-02-24.json`

```bash
make gold dt=2026-02-24
```

**Steps executed:**
1. Load Silver parquet
2. Apply quality filter
3. Remove non-trip rows
4. Compute rolling features
5. Compute physics features
6. Compute lag features
7. Create SOC target labels
8. Write `window_features`
9. Write `trip_features`
10. Write `daily_stats`
11. Write manifest

### `backfill.py`
Loops `run_day.py` over a date range. Not yet exercised in production.
```bash
make backfill start=2026-01-01 end=2026-02-24
```

---

## `tests/`

### `test_end_to_end.py`
Runs the full implemented pipeline on `data/samples/sample_raw.csv`.

**Asserts:**
- Silver parquet exists at expected path
- Timestamps are monotonically increasing per vehicle
- Every row has a `trip_id`
- `quality_flag` is set correctly on known-bad rows
- Manifest JSON is written with expected fields
- Run is idempotent (run twice → identical output)

---

## Root Files

| File | Purpose |
|---|---|
| `.env.example` | Template for every env var `settings.py` reads. Copy to `.env` to run locally. Never commit `.env`. |
| `requirements.txt` | `pandas`, `numpy`, `pyarrow`, `pyyaml`, `python-dotenv`, `pytest` |
| `Makefile` | `make run dt=...` / `make backfill start=... end=...` / `make test` / `make lint` |
| `README.md` | Setup, how to run, how to add a new vehicle, where data lives |

---

## Current Execution — What Runs Today

```bash
make run dt=2026-02-24
```

```
Step 1   Load CSVs from data/raw/dt=2026-02-24/
Step 2   Standardize column names
Step 3   Validate → set quality_flag
Step 4   Write raw_parquet
Step 5   Resample to 1 Hz by signal class
Step 6   Align hardware faults with telemetry timestamps
Step 7   Assign trip_id; update open_trips state
Step 8   Write silver
Step 9   Write manifest
```

**Outputs produced:**
```
data/raw_parquet/dt=2026-02-24/vehicle_id=EV01.parquet

data/silver/dt=2026-02-24/vehicle_id=EV01.parquet
    ├── telemetry signals
    ├── quality_flag
    ├── trip_id
    ├── fault_* columns (one per fault code)
    └── fault_any

data/gold/window_features/dt=2026-02-24/vehicle_id=EV01.parquet
    ├── telemetry signals
    ├── rolling features
    ├── lag features
    ├── physics features
    ├── quality_flag
    ├── trip_id
    ├── fault_any
    ├── y_soc_t_plus_300s
    └── label_available

data/gold/trip_features/dt=2026-02-24/vehicle_id=EV01.parquet
    ├── trip_id
    ├── duration_s
    ├── start_soc
    ├── end_soc
    └── energy metrics

data/gold/daily_stats/dt=2026-02-24/vehicle_id=EV01.parquet
    ├── dt
    ├── vehicle_id
    ├── row counts
    └── trip counts

data/reports/dt=2026-02-24.json
data/state/open_trips.parquet                           ← updated
```

---

## Pipeline Properties

| Property | Status |
|---|---|
| Stable | ✅ |
| Deterministic | ✅ |
| Reproducible | ✅ |
| Idempotent | ✅ (atomic writes; rerun = same output) |
| Cross-day trips | ✅ (via state file) |
| New vehicle = zero code changes | ✅ (drop CSV → pipeline picks it up) |
| Fault alignment | ✅ (binary fault columns in silver) |
| Gold features | ✅ |
| Model training | ⏳ Phase 3 |

---

## What's Next — Phase 3: Modeling

```
Phase 1 (done)   raw → silver (cleaned + validated + fault-aligned + trip-labeled)
Phase 2 (done)   silver → gold (window features + trip features + daily stats)
Phase 3 (next)   gold → ML models
```

Phase 3 will implement:
- Training dataset builder from Gold window features
- Trip-based train/validation split (walk-forward, no random shuffle)
- SOC 5-minute-ahead forecasting model
- Model evaluation pipeline (MAE, MAPE, RMSE)
- Model artifacts and versioning

**First model:**
```
SOC(t + 5 minutes)
```
Regression using Gold `window_features` — rolling stats, lag features, physics variables as inputs.