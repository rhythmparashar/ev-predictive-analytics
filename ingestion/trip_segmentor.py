from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import yaml


# -----------------------------------------------------
# Config
# -----------------------------------------------------

def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# -----------------------------------------------------
# Persistent State
# -----------------------------------------------------

@dataclass
class TripState:
    vehicle_id: str
    trip_seq: int
    open_trip_id: str | None
    open_trip_start_ts: pd.Timestamp | None


def _read_state(state_path: Path) -> dict[str, TripState]:

    if not state_path.exists():
        return {}

    st = pd.read_parquet(state_path)

    out: dict[str, TripState] = {}

    for _, r in st.iterrows():

        out[str(r["vehicle_id"])] = TripState(
            vehicle_id=str(r["vehicle_id"]),
            trip_seq=int(r.get("trip_seq", 0)),
            open_trip_id=None if pd.isna(r.get("open_trip_id"))
            else str(r.get("open_trip_id")),
            open_trip_start_ts=None
            if pd.isna(r.get("open_trip_start_ts"))
            else pd.Timestamp(r.get("open_trip_start_ts")),
        )

    return out


def _write_state(state: dict[str, TripState], state_path: Path) -> None:

    state_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for vid, s in state.items():

        rows.append(
            {
                "vehicle_id": vid,
                "trip_seq": int(s.trip_seq),
                "open_trip_id": s.open_trip_id,
                "open_trip_start_ts": s.open_trip_start_ts,
            }
        )

    pd.DataFrame(rows).to_parquet(state_path, index=False)


# -----------------------------------------------------
# Helper
# -----------------------------------------------------

def _consecutive_true(x: pd.Series, n: int) -> pd.Series:
    """
    True when last n values were True.
    """
    if n <= 1:
        return x.fillna(False)

    return (
        x.fillna(False)
        .rolling(window=n, min_periods=n)
        .sum()
        .eq(n)
    )


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def add_trip_id(
    df: pd.DataFrame,
    trip_cfg_path: Path,
    quality_flags_path: Path,
    state_path: Path,
) -> pd.DataFrame:
    """
    Session segmentation on Silver (1Hz).

    Definition:

    START
    -----
    First real telemetry row after a gap.

    STOP
    ----
    GAP_INSERTED continues for gap_break_s seconds.

    Properties:

    ✔ Cross-day sessions allowed
    ✔ Midnight does NOT split sessions
    ✔ Only real telemetry gaps split sessions
    ✔ Works correctly on resampled 1Hz Silver
    ✔ Deterministic
    ✔ Memory efficient
    """

    cfg = load_yaml(trip_cfg_path)

    gap_break_s = int(cfg.get("gap_break_s", 300))
    min_trip_duration_s = int(cfg.get("min_trip_duration_s", 60))

    df = df.sort_values(["vehicle_id", "timestamp"]).copy()

    df["trip_id"] = pd.NA

    state = _read_state(state_path)

    # -------------------------------------------------

    for vid, g in df.groupby("vehicle_id", sort=False):

        g = g.copy()

        st = state.get(vid) or TripState(
            vehicle_id=vid,
            trip_seq=0,
            open_trip_id=None,
            open_trip_start_ts=None,
        )

        if "quality_flag" not in g.columns:
            raise ValueError("trip_segmentor requires 'quality_flag' column")

        # -------------------------------------------------
        # GAP detection
        # -------------------------------------------------

        qf = g["quality_flag"].fillna(0).astype("int64")

        gap_inserted = (qf & 4) != 0

        present = ~gap_inserted

        # -------------------------------------------------
        # START condition
        # first real row after gap
        # -------------------------------------------------

        start_fire = present & (~present.shift(1, fill_value=False))

        # -------------------------------------------------
        # STOP condition
        # gap run reaches threshold
        # -------------------------------------------------

        gap_run = gap_inserted.groupby(
            (~gap_inserted).cumsum()
        ).cumsum()

        stop_fire = gap_inserted & (gap_run == gap_break_s)

        # -------------------------------------------------
        # Assign trip ids
        # -------------------------------------------------

        open_trip_id = st.open_trip_id
        open_trip_start = st.open_trip_start_ts
        seq = st.trip_seq

        trip_id_col = g["trip_id"].copy()

        for i in range(len(g)):

            # STOP

            if bool(stop_fire.iloc[i]):

                open_trip_id = None
                open_trip_start = None

            # START

            if open_trip_id is None and bool(start_fire.iloc[i]):

                seq += 1

                open_trip_id = f"{vid}_{seq:06d}"

                open_trip_start = pd.Timestamp(
                    g["timestamp"].iloc[i]
                )

            if open_trip_id is not None:

                trip_id_col.iloc[i] = open_trip_id

        g["trip_id"] = trip_id_col

        # -------------------------------------------------
        # Fill gap rows inside sessions
        # -------------------------------------------------

        g["trip_id"] = g["trip_id"].ffill()

        # -------------------------------------------------
        # Min duration enforcement
        # -------------------------------------------------

        sizes = g["trip_id"].value_counts(dropna=True)

        short = sizes[
            sizes < min_trip_duration_s
        ].index

        if len(short) > 0:

            g.loc[
                g["trip_id"].isin(short),
                "trip_id",
            ] = pd.NA

        df.loc[g.index, "trip_id"] = g["trip_id"]

        # -------------------------------------------------
        # Save state
        # -------------------------------------------------

        state[vid] = TripState(

            vehicle_id=vid,

            trip_seq=seq,

            open_trip_id=open_trip_id,

            open_trip_start_ts=open_trip_start,
        )

    _write_state(state, state_path)

    return df