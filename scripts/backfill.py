"""
Backfill over a date range by calling run_day.py logic for each date.
You can enhance later with parallelism.
"""

from __future__ import annotations
import argparse
from datetime import date, timedelta

from scripts.run_day import main as run_day_main  # simple reuse


def daterange(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--vehicle-id", default=None, help="Only process this vehicle (e.g. EV02)")
    args = p.parse_args()

    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)

    skipped = []
    for d in daterange(s, e):
        import sys
        cmd = ["run_day.py", "--dt", d.isoformat()]
        if args.vehicle_id:
            cmd += ["--vehicle-id", args.vehicle_id]
        sys.argv = cmd
        try:
            run_day_main()
        except FileNotFoundError:
            skipped.append(d.isoformat())

    if skipped:
        print(f"[SKIP] No raw data for {len(skipped)} dates: {', '.join(skipped)}")


if __name__ == "__main__":
    main()