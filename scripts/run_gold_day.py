from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from features.pipeline import build_gold_for_vehicle_day


def _list_vehicles_from_silver(dt: str) -> List[str]:
    try:
        from configs.settings import SILVER_DIR  # type: ignore
        silver_dir = Path(SILVER_DIR)
    except Exception:
        silver_dir = Path("data/silver")

    part = silver_dir / f"dt={dt}"
    if not part.exists():
        return []

    vehicles = []
    for p in part.glob("vehicle_id=*.parquet"):
        name = p.name.replace("vehicle_id=", "").replace(".parquet", "")
        vehicles.append(name)
    return sorted(vehicles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", required=True, help="Date partition YYYY-MM-DD")
    ap.add_argument("--vehicle_id", default=None, help="Optional single vehicle id")
    args = ap.parse_args()

    dt = args.dt
    vehicles = [args.vehicle_id] if args.vehicle_id else _list_vehicles_from_silver(dt)

    if not vehicles:
        raise SystemExit(f"No vehicles found in silver for dt={dt}")

    manifests = []
    for vid in vehicles:
        manifests.append(build_gold_for_vehicle_day(dt=dt, vehicle_id=vid))

    total_rows = sum(m["counts"]["window_rows"] for m in manifests)
    total_parts = sum(m["counts"].get("window_parts", 0) for m in manifests)
    print(f"[gold] dt={dt} vehicles={len(vehicles)} total_window_rows={total_rows} total_parts={total_parts}")


if __name__ == "__main__":
    main()