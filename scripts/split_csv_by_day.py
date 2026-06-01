#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd


def split_csv_by_day(
    input_csv: str,
    output_root: str,
    timestamp_col: str,
    fixed_vehicle_id: str,
    chunksize: int = 20000,
    start_date: str | None = None,
    end_date: str | None = None,
):
    input_csv = Path(input_csv)
    output_root = Path(output_root)

    rows_written = 0
    bad_timestamp_rows = 0
    bad_output_file = output_root / "bad_rows.csv"

    start_ts = pd.to_datetime(start_date).date() if start_date else None
    end_ts = pd.to_datetime(end_date).date() if end_date else None

    reader = pd.read_csv(
        input_csv,
        chunksize=chunksize,
        engine="python",
        on_bad_lines="warn",
    )

    for chunk_id, chunk in enumerate(reader, start=1):
        if timestamp_col not in chunk.columns:
            raise ValueError(f"{timestamp_col} column not found")

        chunk = chunk.copy()
        original_timestamps = chunk[timestamp_col].copy()

        cleaned_timestamps = (
            chunk[timestamp_col]
            .astype(str)
            .str.replace("\t", "", regex=False)
            .str.strip()
        )

        chunk[timestamp_col] = pd.to_datetime(
            cleaned_timestamps,
            format="%d-%m-%Y %I:%M:%S %p",
            errors="coerce",
        )

        bad_mask = chunk[timestamp_col].isna()
        bad_rows = chunk[bad_mask].copy()

        if len(bad_rows) > 0:
            bad_rows["bad_timestamp_original"] = original_timestamps[bad_mask]

            print(f"\n[chunk {chunk_id}] bad timestamp rows: {len(bad_rows)}")
            print(
                bad_rows[[timestamp_col, "bad_timestamp_original"]]
                .head(5)
                .to_string(index=False)
            )

            write_header = not bad_output_file.exists()
            bad_rows.to_csv(
                bad_output_file,
                mode="a",
                header=write_header,
                index=False,
            )

        valid = chunk.dropna(subset=[timestamp_col]).copy()
        dropped = len(chunk) - len(valid)
        bad_timestamp_rows += dropped

        if len(valid) == 0:
            print(f"[chunk {chunk_id}] skipped: no valid timestamp rows")
            continue

        if start_ts is not None:
            valid = valid[valid[timestamp_col].dt.date >= start_ts]

        if end_ts is not None:
            valid = valid[valid[timestamp_col].dt.date <= end_ts]

        if len(valid) == 0:
            print(f"[chunk {chunk_id}] skipped: no rows inside requested date range")
            continue

        valid["_dt"] = valid[timestamp_col].dt.strftime("%Y-%m-%d")

        for dt, df_day in valid.groupby("_dt", sort=False):
            out_dir = output_root / f"dt={dt}"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_file = out_dir / f"vehicle_id={fixed_vehicle_id}.csv"
            write_header = not out_file.exists()

            # Drop the internal _dt column — dt is encoded in the directory name,
            # vehicle_id is encoded in the filename. ingest.py sets both from those.
            df_day.drop(columns=["_dt"]).to_csv(
                out_file,
                mode="a",
                header=write_header,
                index=False,
            )

            rows_written += len(df_day)
            print(f"[chunk {chunk_id}] wrote {len(df_day)} rows -> {out_file}")

    print("\nDone")
    print(f"Rows written: {rows_written}")
    print(f"Bad timestamp rows dropped: {bad_timestamp_rows}")
    print("Malformed CSV rows shown as warnings: yes")
    if rows_written > 0:
        print(f"Bad timestamp % vs written rows: {(bad_timestamp_rows / rows_written) * 100:.2f}%")
    print(f"Bad rows saved to: {bad_output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--timestamp-col", default="Timestamp")
    parser.add_argument("--fixed-vehicle-id", required=True)
    parser.add_argument("--chunksize", type=int, default=20000)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    split_csv_by_day(
        input_csv=args.input,
        output_root=args.output_root,
        timestamp_col=args.timestamp_col,
        fixed_vehicle_id=args.fixed_vehicle_id,
        chunksize=args.chunksize,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()