# CLAUDE.md — EV Dashboard

## What this project is
A 3D interactive battery intelligence dashboard for heavy-duty electric machines
(excavators, loaders). The user clicks on a machine, sees its battery pack rendered
in 3D with all 180 cells visible and color-coded by health. Clicking any cell opens
a detail panel showing that cell's live stats, derived health metrics, and predictions.

This dashboard is the frontend for an existing analytics pipeline at:
`../ev-predictive-analytics/`

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript |
| 3D rendering | React Three Fiber + Three.js + @react-three/drei |
| 2D charts | Recharts |
| Styling | Tailwind CSS |
| State | Zustand |
| Backend | FastAPI (Python) |
| Data queries | DuckDB (reads parquet files directly) |
| Model inference | LightGBM (SOC forecast) |

---

## Architecture

```
React Frontend
     │  REST (JSON)
     ▼
FastAPI Backend  (backend/)
     │  DuckDB queries
     ▼
Parquet files  (read-only, from pipeline)
     ▲
     │  written daily by pipeline
../ev-predictive-analytics/
```

The backend never writes data — it only reads pipeline outputs.

---

## Data sources (all read-only parquet)

| What | Path (relative to ev-predictive-analytics/) |
|---|---|
| Master telemetry | `data/analysis/master/**/*.parquet` |
| Cell health (per-cell gap, anomaly) | `data/gold_cells/cell_features/**/*.parquet` |
| Module features (temp, spread) | `data/gold_cells/module_features/**/*.parquet` |
| Daily health reports | `data/gold_cells/daily_health_reports/**/*.parquet` |
| Battery engine outputs | `outputs/battery_engine/` |
| SOC model | `models/soc_forecast/v3__2026-05-20__7b14ac62/model.json` |

DuckDB loaders already exist at:
`../ev-predictive-analytics/battery_intelligence/loader.py`
— import and reuse these in the FastAPI backend service layer.

---

## Battery pack structure

```
5 modules × 36 cells = 180 cells total
5 modules × 18 temp sensors = 90 sensors total

Cell IDs:  M1_C1  → M1_C36
           M2_C1  → M2_C36
           M3_C1  → M3_C36
           M4_C1  → M4_C36
           M5_C1  → M5_C36

Temp IDs:  M1_T1  → M1_T18  (per module)
```

**Physical layout assumption for 3D model:**
- 5 modules arranged in a row (X axis)
- Each module has 36 cells arranged in a 6×6 grid (Y × Z axis)
- Each cell rendered as a rectangular box (cuboid)
- Cells colored by health status (see color scheme below)

---

## Cell health color scheme

| Color | Meaning | Condition |
|---|---|---|
| Green `#2E8B57` | Healthy | gap > −10 mV, anomaly < 10% |
| Yellow `#F4A261` | Monitor | gap −10 to −30 mV OR anomaly 10–50% |
| Orange `#E07000` | Degraded | gap −30 to −60 mV OR anomaly 50–80% |
| Red `#E63946` | Critical | gap < −60 mV OR anomaly > 80% |
| Grey `#6C757D` | No data | null / missing |

---

## Key data columns to know

**From master (telemetry):**
- `timestamp`, `vehicle_id`
- `soc_pct` — State of Charge %
- `battery_current_a`, `abs_current_a`, `current_direction` (charge/idle/discharge)
- `output_power_kw`
- `avg_battery_temp_c`, `max_battery_temp_c`, `min_battery_temp_c`
- `module_1_temp_mean`, `module_1_temp_range`
- `pack_voltage_range` — pack imbalance in V
- `fault_any` — binary fault flag
- `trip_id`
- `total_kwh_consumed`, `total_running_hours_s`, `last_trip_kwh` (from raw parquet)

**From gold_cells/cell_features (per cell per timestamp):**
- `M{n}_C{m}_delta` — cell voltage deviation from module mean (V, multiply ×1000 for mV)
- `M{n}_C{m}_is_outlier` — binary anomaly flag (1 = anomalous)
- `M{n}_C{m}_zscore` — z-score vs module mean

**From gold_cells/module_features:**
- `module_{n}_temp_mean`, `module_{n}_temp_range`
- `module_{n}_voltage_mean`, `module_{n}_voltage_range`
- `module_{n}_lowest_cell` — which cell is weakest in this module

**From daily_health_reports:**
- Per-module daily status: `Healthy` / `Monitor` / `Imbalance Rising` /
  `Thermal Hotspot` / `Degraded Cell` / `Critical`

---

## Focus cell — M1_C25

The most analytically significant cell. Key findings:
- Pre-April (before March 27): avg gap −40 mV, anomaly rate 90–100%
- Post-April (after March 31): avg gap −3 to −5 mV, anomaly rate 17–25%
- A maintenance event occurred March 27–30 (BMS recalibration / firmware update)
- At cold temps (<32°C) under high load (>50A) the gap still spikes to −50/−60 mV
- Root cause: elevated internal resistance — temperature-sensitive weakness
- Battery temperature is the #1 external predictor (Pearson r = 0.50 for M1 temp)

---

## API endpoints to build (FastAPI)

```
GET /api/vehicles
    → list of vehicle IDs

GET /api/vehicle/{vehicle_id}/latest
    → latest telemetry snapshot (SOC, current, temp, fault, power)

GET /api/vehicle/{vehicle_id}/cells/latest
    → latest health status for all 180 cells (gap_mv, anomaly_flag, health_label)
    → used to color the 3D model

GET /api/vehicle/{vehicle_id}/cell/{cell_id}/history?days=7
    → time series for one cell (timestamp, gap_mv, is_outlier, temp)
    → shown in the cell detail panel when user clicks a cell

GET /api/vehicle/{vehicle_id}/modules/latest
    → per-module health (temp_mean, voltage_range, health_label, lowest_cell)

GET /api/vehicle/{vehicle_id}/soc/forecast
    → run LightGBM inference → return current SOC + predicted SOC in 5 min

GET /api/vehicle/{vehicle_id}/metrics/derived
    → health score, anomaly rate, lifetime kWh, running hours,
      worst cell ID + gap, pack voltage spread
```

---

## Metrics panel — what to show when user clicks the battery

**LIVE** (updates every ~5s from /latest endpoint)
- SOC %
- Pack voltage (V)
- Current (A) + direction badge (charging / idle / discharging)
- Output power (kW)
- Avg battery temperature (°C)
- Fault status (green OK / red FAULT)

**DERIVED** (updates daily, from pipeline outputs)
- Battery health score (0–100)
- Worst cell: ID + gap in mV
- Module health: 5 status badges (M1–M5)
- Cell anomaly rate % (last 24h)
- Pack voltage spread (mV)
- Lifetime energy consumed (kWh)
- Total running hours

**PREDICTED** (from model inference)
- SOC in 5 minutes
- Temp risk flag: fires when avg_temp < 32°C AND abs_current > 50A
- Cell gap trend: 7-day rolling slope on M1_C25_delta (improving / stable / worsening)

---

## Cell detail panel — shown when user clicks a single cell in 3D

- Cell ID (e.g. M1_C25)
- Current gap (mV) + health label
- Anomaly flag (yes/no)
- 7-day gap trend chart (line chart)
- 7-day anomaly rate chart (bar chart)
- Module it belongs to + module health
- Nearest temp sensor value

---

## Alert conditions (show banner / highlight cell red)

1. `avg_battery_temp_c < 32 AND abs_current_a > 50` → Cold + high load warning
2. `fault_any == 1` → Active fault
3. Any cell gap < −60 mV → Critical cell alert
4. Module health label == "Critical" → Module critical alert

---

## Build order

```
1. FastAPI backend skeleton + /latest endpoint
2. DuckDB service layer (reuse loader.py from pipeline)
3. React project scaffold (Vite + TS + Tailwind + R3F)
4. Metrics panel component (live + derived + predicted, no 3D yet)
5. Wire frontend to backend API
6. 3D battery pack — 180 cells as colored cuboids
7. Cell click → detail panel + history charts
8. Alert banner system
9. Polish: animations, loading states, responsive layout
```

---

## Fleet

| Vehicle | Machine Type | Data Range | Notes |
|---|---|---|---|
| `EV01` | **Loader** | 2026-01-22 → 2026-05-17 | Firmware update March 27–30, 2026 |
| `EV02` | **Excavator** | 2026-01-22 → 2026-05-14 | Different duty cycle — needs own SOC model |

Both vehicles share the same battery pack: 5-module lithium, 180 cells, 90 temp sensors.
SOC models are **per machine type** — Loader and Excavator are not interchangeable.
