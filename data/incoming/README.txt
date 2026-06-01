DROP ZONE — How to add new data
================================

1. Download the CSV from the web dashboard.

2. Rename the file using this format:
      EV01_2026-06-01.csv        ← EV01 (Loader), date 2026-06-01
      EV02_2026-06-01.csv        ← EV02 (Excavator), date 2026-06-01

3. Drop the file into this folder (data/incoming/).

4. The pipeline picks it up automatically at 6am.
   To process immediately: python scripts/ingest_incoming.py

That's it. Processed files move to data/incoming/processed/ automatically.

REQUIRED COLUMNS IN THE CSV:
  timestamp, soc_pct, stack_voltage_v, battery_current_a, output_power_kw,
  max_cell_voltage_v, min_cell_voltage_v, avg_cell_voltage_v,
  max_battery_temp_c, min_battery_temp_c, avg_battery_temp_c,
  motor_speed_rpm, motor_temperature_c, motor_torque_value_nm,
  total_kwh_consumed, total_running_hours_s
