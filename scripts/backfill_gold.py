# scripts/backfill_gold.py
from __future__ import annotations

import argparse
from datetime import date, timedelta

from scripts.run_gold_day import main as run_one_day  # type: ignore


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    for d in daterange(start, end):
        # Reuse run_gold_day.py CLI by faking argv would be messy;
        # just print the date and instruct to call build from pipeline directly if you prefer.
        from features.pipeline import build_gold_for_vehicle_day
        from pathlib import Path

        # list vehicles in silver
        try:
            from configs.settings import SILVER_DIR  # type: ignore
            silver_dir = Path(SILVER_DIR)
        except Exception:
            silver_dir = Path("data/silver")

        part = silver_dir / f"dt={d.isoformat()}"
        if not part.exists():
            print(f"[gold] skip dt={d.isoformat()} (no silver)")
            continue

        vehicles = sorted([p.name.replace("vehicle_id=", "").replace(".parquet", "") for p in part.glob("vehicle_id=*.parquet")])
        for vid in vehicles:
            build_gold_for_vehicle_day(dt=d.isoformat(), vehicle_id=vid)

        print(f"[gold] done dt={d.isoformat()} vehicles={len(vehicles)}")


if __name__ == "__main__":
    main()