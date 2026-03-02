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
    args = p.parse_args()

    s = date.fromisoformat(args.start)
    e = date.fromisoformat(args.end)

    for d in daterange(s, e):
        # This simplistic approach relies on run_day.py args parsing;
        # you can refactor later into a shared function.
        import sys
        sys.argv = ["run_day.py", "--dt", d.isoformat()]
        run_day_main()


if __name__ == "__main__":
    main()