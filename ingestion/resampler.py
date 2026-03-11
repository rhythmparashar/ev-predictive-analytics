"""
Resampler:
- Converts irregular/hi-frequency telemetry to fixed 1Hz timeline (silver layer)
- Handles duplicate timestamps before resampling:
    - numeric columns -> mean
    - categorical/status columns -> last non-null
    - quality_flag -> bitwise OR
- Applies fill strategy per signal class:
  fast   -> time interpolation (limited by prev/next valid distance)
  slow   -> forward fill (limited by max_gap)
  status -> forward fill (limited by max_gap)
- Adds quality_flag bits for inserted rows and filled values
"""

from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _bitwise_or_reduce(x: pd.Series) -> int:
    vals = x.fillna(0).astype("int64").tolist()
    out = 0
    for v in vals:
        out |= int(v)
    return int(out)


def _last_non_null(x: pd.Series):
    x = x.dropna()
    return x.iloc[-1] if len(x) else None


def aggregate_duplicates_per_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse duplicate timestamps (typical when raw is 2Hz but timestamp has no milliseconds).
    """
    if "timestamp" not in df.columns:
        raise ValueError("aggregate_duplicates_per_timestamp: missing 'timestamp'")

    if not df["timestamp"].duplicated().any():
        return df

    df = df.sort_values("timestamp").copy()

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    non_num_cols = [c for c in df.columns if c not in num_cols and c != "timestamp"]

    agg: dict[str, object] = {}

    for c in num_cols:
        agg[c] = _bitwise_or_reduce if c == "quality_flag" else "mean"

    for c in non_num_cols:
        agg[c] = _last_non_null

    return df.groupby("timestamp", as_index=False).agg(agg)


def _prev_next_gap_seconds(index: pd.DatetimeIndex, valid: pd.Series) -> tuple[pd.Series, pd.Series]:
    idx_series = pd.Series(index, index=index)

    prev_valid_time = pd.Series(index.where(valid), index=index).ffill()
    next_valid_time = pd.Series(index.where(valid), index=index).bfill()

    prev_gap_s = (idx_series - prev_valid_time).dt.total_seconds()
    next_gap_s = (next_valid_time - idx_series).dt.total_seconds()

    return prev_gap_s, next_gap_s


def resample_1hz(
    df: pd.DataFrame,
    resample_cfg: Path,
    signal_classes: Path,
    quality_flags: Path,
) -> pd.DataFrame:
    cfg = load_yaml(resample_cfg)
    classes = load_yaml(signal_classes)
    qf = load_yaml(quality_flags)

    freq = cfg["freq"]  # "1s"
    max_gap = cfg["max_gap_s"]

    df = df.sort_values("timestamp").copy()
    df = aggregate_duplicates_per_timestamp(df)
    df = df.set_index("timestamp")

    full_index = pd.date_range(
        df.index.min().floor("s"),
        df.index.max().ceil("s"),
        freq=freq,
        tz="UTC",
    )
    out = df.reindex(full_index)

    if "quality_flag" not in out.columns:
        out["quality_flag"] = 0
    out["quality_flag"] = out["quality_flag"].fillna(0).astype("int64")

    inserted = out["vehicle_id"].isna()
    out.loc[inserted, "quality_flag"] |= int(qf["GAP_INSERTED"])

    out["vehicle_id"] = out["vehicle_id"].ffill().bfill()

    fast = classes.get("fast", [])
    slow = classes.get("slow", [])
    status = classes.get("status", [])
    counter = classes.get("counter", [])

    def apply_limited_interpolate(col: str, limit_s: int) -> None:
        s = out[col]
        valid = s.notna()
        if valid.all() or (~valid).all():
            return

        prev_gap_s, next_gap_s = _prev_next_gap_seconds(out.index, valid)
        interp = s.interpolate(method="time")

        can_fill = (~valid) & prev_gap_s.le(limit_s) & next_gap_s.le(limit_s)
        if can_fill.any():
            out.loc[can_fill, col] = interp.loc[can_fill]
            out.loc[can_fill, "quality_flag"] |= int(qf["INTERPOLATED_FAST"])

    def apply_limited_ffill(col: str, limit_s: int, flag_bit: int) -> None:
        s = out[col]
        valid = s.notna()
        if valid.all() or (~valid).all():
            return

        idx_series = pd.Series(out.index, index=out.index)
        last_valid_time = pd.Series(out.index.where(valid), index=out.index).ffill()
        gap_s = (idx_series - last_valid_time).dt.total_seconds()

        ffilled = s.ffill()
        can_fill = (~valid) & gap_s.le(limit_s)

        if can_fill.any():
            out.loc[can_fill, col] = ffilled.loc[can_fill]
            if flag_bit:
                out.loc[can_fill, "quality_flag"] |= int(flag_bit)

    for col in fast:
        if col in out.columns:
            apply_limited_interpolate(col, int(max_gap["fast"]))

    for col in slow:
        if col in out.columns:
            apply_limited_ffill(col, int(max_gap["slow"]), int(qf["FORWARD_FILLED"]))

    for col in status:
        if col in out.columns:
            apply_limited_ffill(col, int(max_gap["status"]), int(qf["FORWARD_FILLED"]))

    for col in counter:
        if col in out.columns:
            # Counter-like signals should never be interpolated
            apply_limited_ffill(col, int(max_gap.get("slow", 60)), int(qf["FORWARD_FILLED"]))

    out = out.reset_index().rename(columns={"index": "timestamp"})
    return out