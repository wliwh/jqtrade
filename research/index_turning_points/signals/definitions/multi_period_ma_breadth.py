"""Causal all-A multi-period moving-average breadth signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "multi_period_ma_breadth_v1_20120104_20260814"
REQUESTED_START_DATE = pd.Timestamp("2012-01-01")
MA_WINDOWS = (20, 60, 120)
CHANGE_LOOKBACK = 5
EXTREME_THRESHOLD = 0.70
CHANGE_THRESHOLD = 0.05
SIGNAL_SPECS = (
    ("multi_period_ma_breadth_top", "top"),
    ("multi_period_ma_breadth_bottom", "bottom"),
)


def build_multi_period_ma_breadth_signals(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build symmetric top/bottom signals from MA20/60/120 breadth.

    The composite and its five-trading-day change use only the current and
    earlier rows. The first five rows remain inactive because the change is not
    yet available.
    """

    market = _validate_daily_features(daily_features, start_date=start_date)
    breadth_columns = [f"breadth_ma{window}" for window in MA_WINDOWS]
    valid_columns = [f"valid_count_ma{window}" for window in MA_WINDOWS]
    market["breadth_composite"] = market[breadth_columns].mean(axis=1)
    market["breadth_change_5d"] = market["breadth_composite"].diff(
        CHANGE_LOOKBACK
    )
    market["change_available"] = market["breadth_change_5d"].notna()
    market["minimum_ma_valid_count"] = market[valid_columns].min(axis=1).astype(int)

    top_triggered = (
        market["breadth_composite"].ge(EXTREME_THRESHOLD)
        & market["breadth_change_5d"].le(-CHANGE_THRESHOLD)
    )
    bottom_triggered = (
        market["breadth_composite"].le(1.0 - EXTREME_THRESHOLD)
        & market["breadth_change_5d"].ge(CHANGE_THRESHOLD)
    )
    triggers = {
        "multi_period_ma_breadth_top": top_triggered,
        "multi_period_ma_breadth_bottom": bottom_triggered,
    }

    shared = {
        "date": market["date"],
        "raw_value": market["breadth_composite"],
        "universe_size": market["universe_size"],
        "valid_count": market["minimum_ma_valid_count"],
        "version": version,
        "breadth_composite": market["breadth_composite"],
        "breadth_change_5d": market["breadth_change_5d"],
        "change_available": market["change_available"],
    }
    for window in MA_WINDOWS:
        for prefix in ("above_count", "valid_count", "breadth"):
            column = f"{prefix}_ma{window}"
            shared[column] = market[column]

    signal_frames = []
    for signal_id, direction in SIGNAL_SPECS:
        frame = pd.DataFrame(shared)
        frame.insert(1, "signal_id", signal_id)
        frame.insert(2, "direction", direction)
        frame.insert(4, "triggered", triggers[signal_id])
        signal_frames.append(frame)

    source = pd.concat(signal_frames, ignore_index=True)
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    episodes_by_direction = (
        episodes.groupby("direction").size().astype(int).to_dict()
    )
    triggered_by_direction = (
        daily.groupby("direction")["triggered"].sum().astype(int).to_dict()
    )
    metadata = {
        "signal_version": version,
        "requested_start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "comparison_start_date": market["date"].min().strftime("%Y-%m-%d"),
        "comparison_end_date": market["date"].max().strftime("%Y-%m-%d"),
        "trade_dates": len(market),
        "daily_rows": len(daily),
        "signal_series": len(SIGNAL_SPECS),
        "triggered_days_by_direction": triggered_by_direction,
        "episodes_by_direction": episodes_by_direction,
        "change_available_from": market.loc[
            market["change_available"], "date"
        ].iloc[0].strftime("%Y-%m-%d"),
    }
    return daily, episodes, metadata


def _validate_daily_features(
    frame: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("daily_features must be a pandas DataFrame")
    required = {"date", "universe_size"}
    for window in MA_WINDOWS:
        required.update(
            {
                f"above_count_ma{window}",
                f"valid_count_ma{window}",
                f"breadth_ma{window}",
            }
        )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily_features is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("daily_features must not be empty")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily_features contains an invalid date")
    result = result[result["date"].ge(pd.Timestamp(start_date))].copy()
    if result.empty:
        raise ValueError("daily_features has no rows on or after start_date")
    if result["date"].duplicated().any():
        raise ValueError("daily_features contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("daily_features dates must be strictly increasing")

    integer_columns = ["universe_size"]
    for window in MA_WINDOWS:
        integer_columns.extend(
            [f"above_count_ma{window}", f"valid_count_ma{window}"]
        )
    for column in integer_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if (
            values.isna().any()
            or values.lt(0).any()
            or np.not_equal(values, np.floor(values)).any()
        ):
            raise ValueError(f"{column} must contain non-negative integers")
        result[column] = values.astype(int)
    if result["universe_size"].eq(0).any():
        raise ValueError("universe_size must be positive")

    for window in MA_WINDOWS:
        above_column = f"above_count_ma{window}"
        valid_column = f"valid_count_ma{window}"
        breadth_column = f"breadth_ma{window}"
        if result[valid_column].eq(0).any():
            raise ValueError(f"{valid_column} must be positive")
        if result[valid_column].gt(result["universe_size"]).any():
            raise ValueError(f"{valid_column} must not exceed universe_size")
        if result[above_column].gt(result[valid_column]).any():
            raise ValueError(f"{above_column} must not exceed {valid_column}")
        breadth = pd.to_numeric(result[breadth_column], errors="coerce")
        if (
            breadth.isna().any()
            or not np.isfinite(breadth.to_numpy(dtype=float)).all()
            or breadth.lt(0).any()
            or breadth.gt(1).any()
        ):
            raise ValueError(
                f"{breadth_column} must contain finite values between zero and one"
            )
        expected = result[above_column].divide(result[valid_column])
        if not np.allclose(breadth, expected, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"{breadth_column} does not match {above_column}/{valid_column}"
            )
        result[breadth_column] = breadth.astype(float)
    return result.reset_index(drop=True)
