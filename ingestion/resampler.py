"""
Resampler:
- Converts irregular/hi-frequency telemetry to fixed 1Hz timeline (silver layer)
- IMPORTANT: If timestamps are second-resolution and data is 2Hz, we will see duplicate timestamps.
  We handle this by aggregating all rows within each timestamp (second) BEFORE resampling:
    - numeric columns -> mean
    - categorical/status columns -> last non-null
    - quality_flag -> bitwise OR
- Applies fill strategy per signal class:
  fast   -> time interpolation (limited by max_gap)
  slow   -> forward fill (limited by max_gap)
  status -> forward fill (no interpolation)
- Adds quality_flag bits for inserted rows and filled values
"""

from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _bitwise_or_reduce(x: pd.Series) -> int:
    """Bitwise OR reducer for quality_flag across duplicates."""
    vals = x.fillna(0).astype("int64").tolist()
    out = 0
    for v in vals:
        out |= int(v)
    return int(out)


def _last_non_null(x: pd.Series):
    """Return last non-null value (good for categorical/status columns)."""
    x = x.dropna()
    return x.iloc[-1] if len(x) else None


def aggregate_duplicates_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate timestamps (typical when raw is 2Hz but timestamp has no milliseconds).

    Rules:
    - numeric columns -> mean (except quality_flag)
    - quality_flag -> bitwise OR
    - non-numeric columns -> last non-null
    """
    if "timestamp" not in df.columns:
        raise ValueError("aggregate_duplicates_per_timestamp: missing 'timestamp'")

    if not df["timestamp"].duplicated().any():
        return df

    df = df.sort_values("timestamp").copy()

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in df.columns if c not in num_cols and c != "timestamp"]

    agg = {}

    # Numeric columns
    for c in num_cols:
        if c == "quality_flag":
            agg[c] = _bitwise_or_reduce
        else:
            agg[c] = "mean"

    # Non-numeric columns
    for c in non_num_cols:
        agg[c] = _last_non_null

    out = df.groupby("timestamp", as_index=False).agg(agg)
    return out


def resample_1hz(
    df: pd.DataFrame,
    resample_cfg: Path,
    signal_classes: Path,
    quality_flags: Path,
) -> pd.DataFrame:
    cfg = load_yaml(resample_cfg)
    classes = load_yaml(signal_classes)
    qf = load_yaml(quality_flags)

    freq = cfg["freq"]        # recommended: "1s"
    max_gap = cfg["max_gap_s"]

    # Sort
    df = df.sort_values("timestamp").copy()

    # Aggregate duplicates per second (2Hz -> 1Hz collapse)
    df = aggregate_duplicates_per_timestamp(df)

    # Index by timestamp for reindexing
    df = df.set_index("timestamp")

    # Build full index from min to max at 1Hz
    full_index = pd.date_range(
        df.index.min().floor("s"),
        df.index.max().ceil("s"),
        freq=freq,
        tz="UTC",
    )
    out = df.reindex(full_index)

    # Ensure quality_flag exists and is int64
    if "quality_flag" not in out.columns:
        out["quality_flag"] = 0
    out["quality_flag"] = out["quality_flag"].fillna(0).astype("int64")

    # Mark inserted rows (where original data had no sample at that second)
    inserted = out["vehicle_id"].isna()
    out.loc[inserted, "quality_flag"] |= int(qf["GAP_INSERTED"])

    # Fill vehicle_id (constant per file)
    out["vehicle_id"] = out["vehicle_id"].ffill().bfill()

    fast = classes.get("fast", [])
    slow = classes.get("slow", [])
    status = classes.get("status", [])

    def apply_limited_interpolate(col: str, limit_s: int) -> None:
        """Interpolate values only if gap from last valid sample <= limit_s."""
        s = out[col]
        valid = s.notna()

        last_valid_time = pd.Series(out.index.where(valid), index=out.index).ffill()
        gap_s = (pd.Series(out.index, index=out.index) - last_valid_time).dt.total_seconds()

        interp = s.interpolate(method="time")
        can_fill = (~valid) & (gap_s <= limit_s)

        out.loc[can_fill, col] = interp.loc[can_fill]
        if can_fill.any():
            out.loc[can_fill, "quality_flag"] |= int(qf["INTERPOLATED_FAST"])

    def apply_limited_ffill(col: str, limit_s: int, flag_bit: int) -> None:
        """Forward-fill only if gap from last valid sample <= limit_s."""
        s = out[col]
        valid = s.notna()

        last_valid_time = pd.Series(out.index.where(valid), index=out.index).ffill()
        gap_s = (pd.Series(out.index, index=out.index) - last_valid_time).dt.total_seconds()

        ffilled = s.ffill()
        can_fill = (~valid) & (gap_s <= limit_s)

        out.loc[can_fill, col] = ffilled.loc[can_fill]
        if can_fill.any() and flag_bit != 0:
            out.loc[can_fill, "quality_flag"] |= int(flag_bit)

    # Apply fill strategies
    for col in fast:
        if col in out.columns:
            apply_limited_interpolate(col, int(max_gap["fast"]))

    for col in slow:
        if col in out.columns:
            apply_limited_ffill(col, int(max_gap["slow"]), int(qf["FORWARD_FILLED_SLOW"]))

    for col in status:
        if col in out.columns:
            # status forward-fill allowed; do not mark as slow fill by default
            apply_limited_ffill(col, int(max_gap["status"]), 0)

    # Restore timestamp column
    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out