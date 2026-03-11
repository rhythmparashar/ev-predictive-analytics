# training/splitter.py
#
# Trip-level chronological train/val split.
# Shared by all tasks — never splits mid-trip.
#
# Why trip-level?
#   Splitting mid-trip leaks context: the model sees the start of a trip
#   during training and its end during validation, inflating val metrics.
#   All trips from the training window go to train; val trips are later.

from __future__ import annotations

from typing import Tuple

import pandas as pd


def split_by_dates(
    df: pd.DataFrame,
    train_dates: list[str],
    val_dates: list[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by date partition — all rows whose trip started on a train_date go
    to train, val_date rows go to val.

    Requires 'timestamp' and 'trip_id' columns.
    A trip that spans midnight is assigned to whichever date it started on.
    """
    trip_start_date = (
        df.groupby("trip_id")["timestamp"]
        .min()
        .dt.date
        .astype(str)
    )

    train_trip_ids = trip_start_date[trip_start_date.isin(train_dates)].index
    val_trip_ids   = trip_start_date[trip_start_date.isin(val_dates)].index

    train_df = df[df["trip_id"].isin(train_trip_ids)].copy()
    val_df   = df[df["trip_id"].isin(val_trip_ids)].copy()

    _print_split_summary(df, train_df, val_df, train_trip_ids, val_trip_ids)

    return train_df, val_df


def split_last_n_trips(
    df: pd.DataFrame,
    n_val_trips: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological split: last N trips by start time go to val, rest to train.
    Used when val_dates aren't specified (single-date evaluation mode).
    """
    trip_start = (
        df.groupby("trip_id")["timestamp"]
        .min()
        .sort_values()
    )

    val_trip_ids   = trip_start.tail(n_val_trips).index
    train_trip_ids = trip_start.head(len(trip_start) - n_val_trips).index

    train_df = df[df["trip_id"].isin(train_trip_ids)].copy()
    val_df   = df[df["trip_id"].isin(val_trip_ids)].copy()

    _print_split_summary(df, train_df, val_df, train_trip_ids, val_trip_ids)

    return train_df, val_df


def _print_split_summary(df, train_df, val_df, train_ids, val_ids):
    trip_start = df.groupby("trip_id")["timestamp"].min()

    print(f"\n  Split summary")
    print(f"    Train: {len(train_df):,} rows, {len(train_ids)} trips  "
          f"[{trip_start.loc[train_ids].min()} → {trip_start.loc[train_ids].max()}]")
    print(f"    Val:   {len(val_df):,} rows, {len(val_ids)} trips  "
          f"[{trip_start.loc[val_ids].min()} → {trip_start.loc[val_ids].max()}]")
