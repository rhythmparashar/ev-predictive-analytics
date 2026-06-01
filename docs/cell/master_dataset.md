## Master Dataset Build — Summary

### What you built
A clean, analysis-ready master dataset joining VCU silver + cell module features + cell pack features for EV01.

**Location:** `data/analysis/master/dt={partition}/vehicle_id=EV01.parquet`

**Coverage:** Jan 22 → Mar 08, 2026 — 39 partitions, 744,618 rows, 100% clean, 68 columns per row.

---

### What's in each row
- **VCU signals** — SOC, current, voltage, temperature, motor speed/torque, battery status, fault flags, trip ID
- **Pack features** — pack voltage range/std, worst cell, imbalance flag, temp range, hottest module
- **Module features** — per-module voltage mean/std/range, lowest cell, temp mean/range, hottest sensor (all 5 modules)
- **Derived columns** — cell_spread_v, abs_current_a, current_direction, soc_band, operating_regime, is_clean, dt, actual_date

---

### Issues found and fixed along the way

| Issue | Fix |
|---|---|
| Cell pack/module features had massive duplicate timestamps (up to 188k rows) | `drop_duplicates` in build script before merge |
| VCU silver Feb 01-12 had wrong timestamps (day/month swapped) | Fixed `dayfirst=True` → `dayfirst=False` in `validators.py`, re-ingested those dates |
| Cell partitions span multiple calendar days (Mar 01 covers Feb 27-Mar 01) | Build script uses partition key as-is, `actual_date` derived from timestamp |
| Feb 01-12 built 0 rows due to timestamp mismatch | Resolved by fixing validators.py, all 11 dates now OK |

---

### Known issues still open

| Issue | Location |
|---|---|
| Cell pipeline writing duplicate rows for Jan dates (up to 82k dupes on Jan 30) | `cell_pipeline.py` — fix before next cell gold rebuild |
| VCU retention is 17-47% per day | Expected — cell data only covers active machine hours, not full 24h |
| `quality_flags.yaml` MODEL_SAFE_MASK=52 but code uses `& 48` | VCU pipeline — fix before next retrain |
| `total_kwh_consumed` and `motor_temperature_c` should be dropped before next SOC retrain | SOC model — Phase 4 |

---

Ready to move to baseline analysis. What do you want to start with?