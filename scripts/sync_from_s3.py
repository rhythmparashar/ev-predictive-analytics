"""
S3 → Incoming Sync
==================
Polls an S3 bucket for new telemetry CSV files, downloads any not yet seen,
then triggers the ingestion pipeline for each new file.

Expected S3 key format:
    telemetry/EV01_2026-06-01.csv
    telemetry/EV02_2026-06-01.csv

The script tracks which S3 keys it has already downloaded in
data/state/s3_sync_state.json so it never re-downloads the same file.

Usage:
    python scripts/sync_from_s3.py                        # uses env or ~/.aws/credentials
    python scripts/sync_from_s3.py --bucket my-bucket
    python scripts/sync_from_s3.py --bucket my-bucket --prefix telemetry/

Set env vars to override credentials:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import STATE_DIR

INCOMING_DIR   = PROJECT_ROOT / "data" / "incoming"
S3_STATE_FILE  = STATE_DIR / "s3_sync_state.json"

# ── Defaults (override via CLI args or env vars) ───────────────────────────
DEFAULT_BUCKET = os.getenv("EV_S3_BUCKET", "")
DEFAULT_PREFIX = os.getenv("EV_S3_PREFIX", "telemetry/")


def load_s3_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if S3_STATE_FILE.exists():
        return json.loads(S3_STATE_FILE.read_text())
    return {"downloaded_keys": []}


def save_s3_state(state: dict) -> None:
    S3_STATE_FILE.write_text(json.dumps(state, indent=2))


def list_bucket_files(s3, bucket: str, prefix: str) -> list[str]:
    """Return all .csv / .xlsx / .xls keys under the prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".csv", ".xlsx", ".xls")):
                keys.append(key)
    return keys


def download_key(s3, bucket: str, key: str) -> Path:
    """Download an S3 key to data/incoming/ and return the local path."""
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    filename = Path(key).name
    local    = INCOMING_DIR / filename
    s3.download_file(bucket, key, str(local))
    return local


def main() -> None:
    ap = argparse.ArgumentParser(description="Sync new CSVs from S3 to incoming/")
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX, help="Key prefix (default: telemetry/)")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()

    if not args.bucket:
        print("ERROR: bucket name required. Use --bucket or set EV_S3_BUCKET env var.")
        sys.exit(1)

    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    s3    = boto3.client("s3")
    state = load_s3_state()
    done  = set(state["downloaded_keys"])

    print(f"\nChecking s3://{args.bucket}/{args.prefix}")
    keys = list_bucket_files(s3, args.bucket, args.prefix)

    new_keys = [k for k in keys if k not in done]
    print(f"  Total files in bucket : {len(keys)}")
    print(f"  Already downloaded    : {len(done)}")
    print(f"  New to download       : {len(new_keys)}")

    if not new_keys:
        print("  Nothing new — exiting.")
        return

    downloaded: list[Path] = []
    for key in sorted(new_keys):
        filename = Path(key).name
        print(f"\n  ↓ {key}")

        if args.dry_run:
            print(f"    DRY RUN — would download to data/incoming/{filename}")
            continue

        try:
            local = download_key(s3, args.bucket, key)
            state["downloaded_keys"].append(key)
            save_s3_state(state)
            downloaded.append(local)
            print(f"    ✓ saved to {local}")
        except Exception as e:
            print(f"    ✗ failed: {e}")

    if downloaded and not args.dry_run:
        print(f"\n  {len(downloaded)} file(s) downloaded → running ingestion pipeline…\n")
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/ingest_incoming.py"],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            print("  Pipeline returned non-zero exit code — check logs/ingest.log")


if __name__ == "__main__":
    main()
