# EV Telemetry ML — Phase 1: Data Foundation
> **Status: Locked. Ready to build.**  
> Every folder and file below is intentional. This document explains what each piece does and why it exists.

---

## What Phase 1 Builds

A pipeline that runs every night and does exactly one thing reliably:

```
Raw CSV files  →  Validated  →  Resampled  →  Trip-labeled  →  Feature-engineered  →  Parquet on disk
```

No models. No serving. No dashboards. Just clean, versioned, trustworthy data that every future model depends on.

---

## Full Structure

```
ev-telemetry-ml/
│
├── ingestion/
│   ├── ingest.py
│   ├── validators.py
│   ├── resampler.py
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
├── data/
│   ├── raw/
│   │   └── dt=2026-02-21/
│   │       ├── vehicle_id=EV01.csv
│   │       └── vehicle_id=EV02.csv
│   ├── raw_parquet/
│   │   └── dt=2026-02-21/
│   │       ├── vehicle_id=EV01.parquet
│   │       └── vehicle_id=EV02.parquet
│   ├── silver/
│   │   └── dt=2026-02-21/
│   │       ├── vehicle_id=EV01.parquet
│   │       └── vehicle_id=EV02.parquet
│   ├── gold/
│   │   ├── window_features/
│   │   │   └── dt=2026-02-21/
│   │   │       ├── vehicle_id=EV01.parquet
│   │   │       └── vehicle_id=EV02.parquet
│   │   ├── trip_features/
│   │   │   └── dt=2026-02-21/
│   │   │       ├── vehicle_id=EV01.parquet
│   │   │       └── vehicle_id=EV02.parquet
│   │   └── daily_stats/
│   │       └── dt=2026-02-21.parquet
│   ├── state/
│   │   └── open_trips.parquet
│   ├── reports/
│   │   └── dt=2026-02-21.json
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

## Folder-by-Folder Breakdown

---

### `ingestion/`
**What:** The daily ETL engine. Runs every night on new raw data.  
**Why:** Raw telemetry is messy — wrong types, stuck sensors, missing timestamps, duplicate rows. Nothing downstream should ever touch raw data directly. This folder is the gatekeeper.

| File | What it does | Why it exists |
|---|---|---|
| `ingest.py` | Orchestrates the full flow: raw CSV → raw_parquet → silver. Calls validators, resampler, trip_segmentor, io in order. | Single entry point so the pipeline is one function call, not a sequence of manual steps. |
| `validators.py` | Checks schema, null rates, out-of-range values (hard + soft), time reversals, duplicate timestamps, sensor flatlines. Sets `quality_flag` bits on each row. | Catches data problems at the source before they silently corrupt features and models. |
| `resampler.py` | Aligns all signals to 1 Hz. Interpolates fast signals (current, voltage), forward-fills slow signals (temp, SOC), last-values status signals (relay state, mode). Respects max gap limits — never fills gaps longer than configured threshold. | Telemetry arrives at irregular intervals. Models need uniform time steps. Different signals need different fill strategies or you introduce false physics. |
| `trip_segmentor.py` | Detects trip start and end from MCU Enable State + Motor Status Word + idle gap duration. Assigns a unique `trip_id` to every row. Writes unfinished trips to `state/open_trips.parquet` and picks them up the next day. | Every model we build in Phase 2+ needs to reason about trips, not raw time-series. Segmentation here means it never needs to be re-done. |
| `io.py` | Reads raw CSVs with correct types. Writes parquet with atomic overwrite (write to temp → rename). Builds partition paths. | Keeps all file I/O in one place. Atomic writes mean a failed run never leaves corrupt partial files. |

**`ingestion/tests/`**

| File | What it tests |
|---|---|
| `test_validators.py` | Known bad inputs → correct quality_flag bits. Edge cases: all nulls, reversed timestamps, stuck sensors. |
| `test_resampler.py` | Gap insertion, fast vs slow vs status fill, max-gap constraint respected, quality_flag set correctly on filled rows. |
| `test_trip_segmentor.py` | Trip start debounce, idle-gap end detection, cross-midnight trips stay one trip, open_trip state writes and reads correctly. |
| `test_io.py` | Partition path correctness, schema casting (strings → floats), parquet roundtrip fidelity, idempotency (run twice → same output). |

---

### `features/`
**What:** Computes all derived features from silver data and writes them to gold.  
**Why:** This is a shared library — imported by both the training pipeline (Phase 2) and the serving layer (Phase 3). Building it correctly now as a standalone library prevents feature skew, which is the #1 cause of production model failures.

| File | What it does | Why it exists |
|---|---|---|
| `rolling.py` | Rolling mean, std, min, max over configurable windows (30s, 1min, 5min, 10min) for current, voltage, temperature, SOC. | Rolling stats capture temporal dynamics that point-in-time values miss. These are the most important inputs for short-term forecasting models. |
| `physics.py` | Computes voltage imbalance (max−min cell voltage), thermal gradient (max−min battery temp), BTMS delta-T (outlet−inlet), C-rate (current / nominal capacity), drivetrain efficiency (output power / DC input power). | Physics features encode domain knowledge. Without them, models must learn relationships like "high C-rate accelerates degradation" from data alone, which needs far more samples. |
| `lags.py` | Generates lag features at t−1min, t−5min, t−10min for key signals (SOC, battery current, stack voltage, temperature). | Lag features give non-sequence models (XGBoost, Random Forest) the temporal context they need. Without lags, XGBoost sees each row as independent. |
| `trip_agg.py` | Aggregates closed trips into one row per trip: total kWh consumed, max temperature, avg C-rate, trip duration, start/end SOC, etc. | Trip-level rows are what trip energy prediction and fault classification models train on. Only processes trips marked as closed in `state/open_trips.parquet`. |
| `pipeline.py` | Orchestrates the full silver → gold transformation. Applies quality filter `(quality_flag & (16|32)) == 0` before any feature computation. Calls rolling, physics, lags, trip_agg and writes to correct gold partitions. | One entry point for feature computation. The quality filter here means no model ever trains on hard-breach or time-anomaly rows. |

**`features/tests/`**

| File | What it tests |
|---|---|
| `test_rolling.py` | Known input series → expected rolling output. Edge cases: window larger than series, all-null window. |
| `test_physics.py` | Known voltage/current/temp values → correct imbalance, delta-T, efficiency. Division-by-zero handling. |
| `test_pipeline.py` | Quality filter removes correct rows. Output parquet has expected columns and shape. |

---

### `data/`
**What:** All data on disk, organized by layer and partition.  
**Why:** The medallion architecture (raw → raw_parquet → silver → gold) means every transformation is visible and reversible. If a bug is found in `resampler.py`, you reprocess from raw_parquet — you never re-download from source.

**Never commit real telemetry to git.** `data/` is in `.gitignore` except for `data/samples/`.

| Folder | Contains | Partition key | Produced by |
|---|---|---|---|
| `raw/` | Original CSV files exactly as received. Never modified. | `dt=` / `vehicle_id=` | Manual drop or upstream transfer |
| `raw_parquet/` | Same data as raw, converted to typed parquet. Schema enforced, column names standardized. | `dt=` / `vehicle_id=` | `ingestion/io.py` |
| `silver/` | Clean, resampled, trip-labeled data with `quality_flag` column. The trusted layer. | `dt=` / `vehicle_id=` | `ingestion/ingest.py` |
| `gold/window_features/` | Row-level feature set ready for short-term forecasting models. One row per second per vehicle. | `dt=` / `vehicle_id=` | `features/pipeline.py` |
| `gold/trip_features/` | One row per closed trip. Partitioned by trip END date. | `dt=` (end date) / `vehicle_id=` | `features/trip_agg.py` |
| `gold/daily_stats/` | Per-vehicle daily summary: avg SOC, total kWh, trips completed, quality stats. Single file covers all vehicles. | `dt=` | `features/pipeline.py` |
| `state/open_trips.parquet` | One row per vehicle currently mid-trip at end of day. Carried forward to next day's ingestion run. | — (vehicle_id column) | `ingestion/trip_segmentor.py` |
| `reports/` | JSON manifest written after each successful run. Records what was processed, quality counts, pipeline versions. | `dt=` | `scripts/run_day.py` |
| `samples/` | Tiny anonymized fixtures (< 1000 rows). The only data committed to git. Used by all tests. | — | Hand-crafted |

---

### `schema/`
**What:** The single source of truth for all signal definitions. Every other module imports from here — nothing is hardcoded anywhere else.  
**Why:** When a column name changes or a new signal is added, you change one file. Not five files scattered across ingestion and features.

| File | What it defines |
|---|---|
| `telemetry_schema.yaml` | Every column name, its dtype, and whether it's required. This is what `validators.py` and `io.py` enforce on every CSV. |
| `ranges.yaml` | Per-signal valid ranges: hard gate (reject / flag invalid) and soft flag (flag suspect, keep). E.g., SOC hard gate: [0, 100], soft flag: [2, 98]. |
| `units.yaml` | Unit for every signal (V, A, °C, kW, Nm, RPM, etc.). Used in reports and feature naming. |
| `signal_classes.yaml` | Every signal classified as `fast`, `slow`, or `status`. Controls which fill strategy `resampler.py` applies. |
| `quality_flags.yaml` | Bitmask bit definitions (bit 0=interpolated, bit 4=hard breach, bit 5=time anomaly, etc.). Imported by validators, resampler, and features/pipeline. |

---

### `configs/`
**What:** Runtime configuration. Controls behavior without touching code.  
**Why:** You will tune resampling gaps, trip detection thresholds, and file paths constantly early on. Config files mean zero code changes for behavior changes.

| File | What it configures |
|---|---|
| `settings.py` | Loads all paths and environment variables from `.env`. The only file that reads env vars — everything else imports from here. |
| `resample.yaml` | Target frequency (1 Hz), max gap per signal class before refusing to fill, fill strategy overrides for specific signals. |
| `trip.yaml` | Idle gap seconds before declaring trip end, minimum trip duration to count as a real trip, debounce seconds for trip start, signals used for detection. |

---

### `scripts/`
**What:** The operational entry points you actually run.  
**Why:** Nobody should run `python ingestion/ingest.py` directly. Scripts wire everything together into one safe, logged, manifest-writing command.

| File | What it does |
|---|---|
| `run_day.py` | Runs the full pipeline for one date: ingest → validate → resample → segment → features → write manifest. Usage: `python scripts/run_day.py --dt 2026-02-21` |
| `backfill.py` | Loops `run_day.py` over a date range. Usage: `python scripts/backfill.py --start 2026-01-01 --end 2026-02-21` |

---

### `notebooks/`
**What:** Scratch space for exploration only.  
**Why:** You need a place to look at data interactively. But notebook code never goes to production — all real logic lives in the modules above.

| File | Purpose |
|---|---|
| `01_eda.ipynb` | Load raw CSVs, check signal distributions, identify bad sensors, understand gap patterns before writing validators. |
| `02_feature_check.ipynb` | Load gold parquet, plot rolling features and physics variables, sanity-check that computed values look physically reasonable. |

---

### `tests/`

| File | What it tests |
|---|---|
| `test_end_to_end.py` | Runs the full pipeline on `data/samples/sample_raw.csv`. Asserts: silver has monotonic timestamps per vehicle, every row has a `trip_id`, gold files exist with expected columns, `quality_flag` is set on known-bad rows, manifest is written with correct fields. |

---

### Root Files

| File | Purpose |
|---|---|
| `.env.example` | Template listing every environment variable `settings.py` reads. Copy to `.env` to run locally. Never commit `.env`. |
| `requirements.txt` | Pinned dependencies. Phase 1 needs: `polars`, `pyarrow`, `pyyaml`, `pytest`, `python-dotenv`. |
| `Makefile` | `make run dt=2026-02-21` / `make test` / `make backfill start=... end=...` / `make lint` |
| `README.md` | Setup instructions, how to run the pipeline, how to add a new vehicle, where data lives. |

---

## Build Order

Follow this exactly. Each step is fully testable before the next begins.

```
Step 1   schema/signal_classes.yaml + quality_flags.yaml + telemetry_schema.yaml + ranges.yaml
         → Define every signal, its class, and its valid range. Everything else imports from here.

Step 2   ingestion/io.py + test_io.py
         → CSV read, typed parquet write, atomic overwrite, partition path helpers.
         → Test: roundtrip, idempotency, partition correctness.

Step 3   ingestion/validators.py + test_validators.py
         → Schema enforcement, range checks, quality_flag bit assignment.
         → Test: known bad inputs produce correct quality_flag values.

Step 4   ingestion/resampler.py + test_resampler.py
         → 1Hz resample by signal class. Gap insertion. Fill strategies. Max-gap constraint.
         → Test: fast/slow/status fill, quality_flag set on filled rows, gaps refused at limit.

Step 5   ingestion/trip_segmentor.py + test_trip_segmentor.py
         → Trip boundary detection. open_trips state read/write.
         → Test: debounce, idle-gap end, cross-midnight trip, state persistence.

Step 6   ingestion/ingest.py
         → Wire steps 2–5. raw CSV → raw_parquet → silver.
         → Manually run on sample data. Inspect silver output.

Step 7   features/rolling.py + test_rolling.py
         → Rolling stats. Test with known series.

Step 8   features/physics.py + test_physics.py
         → Physics variables. Test with known values. Check division-by-zero handling.

Step 9   features/lags.py
         → Lag features. Straightforward — test with a simple time series.

Step 10  features/trip_agg.py
         → Trip aggregations. Only runs on closed trips. Test with sample silver + state.

Step 11  features/pipeline.py + test_pipeline.py
         → silver → gold. Quality filter applied. Calls steps 7–10.
         → Test: filtered rows absent from gold, expected columns present.

Step 12  scripts/run_day.py
         → End-to-end for one date. Writes manifest to reports/.

Step 13  tests/test_end_to_end.py
         → Full pipeline on samples/. Assert everything. This is your regression test.
```

---

## What "Done" Looks Like for Phase 1

- `make run dt=<any date>` completes without errors
- Silver parquet has: correct dtypes, monotonic timestamps per vehicle, `trip_id` on every row, `quality_flag` set correctly
- Gold parquet has: window features, trip features (closed trips only), daily stats
- `data/reports/dt=<date>.json` written with row counts, quality counts, pipeline version
- `make test` passes with 100% of assertions green
- A new vehicle can be added by dropping its CSV in `data/raw/dt=.../` — no code changes needed