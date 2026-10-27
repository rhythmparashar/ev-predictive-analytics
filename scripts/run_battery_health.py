"""
Compute and write battery health scores for a single vehicle.

Usage:
    python scripts/run_battery_health.py --vehicle-id EV03
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import DATA_DIR
from battery_intelligence.loader import load_master
from battery_intelligence.health_score import compute_health_scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vehicle-id", required=True)
    args = ap.parse_args()

    vehicle_id = args.vehicle_id

    master_df = load_master(DATA_DIR)
    if "vehicle_id" in master_df.columns:
        master_df = master_df[master_df["vehicle_id"] == vehicle_id]

    if master_df.empty:
        print(f"[battery_health] no master data for {vehicle_id} — skipping")
        sys.exit(0)

    scored = compute_health_scores(master_df)
    out_dir = PROJECT_ROOT / "outputs" / "battery_health"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scores_{vehicle_id}.parquet"
    scored.to_parquet(out_path, index=False)
    print(f"[battery_health] vehicle={vehicle_id} rows={len(scored):,} → {out_path}")


if __name__ == "__main__":
    main()
