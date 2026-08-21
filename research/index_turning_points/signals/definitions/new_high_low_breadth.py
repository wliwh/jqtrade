"""Causal all-A new-high versus new-low breadth reversal signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "new_high_low_breadth_v1_20120104_20260814"
REQUESTED_START_DATE = pd.Timestamp("2012-01-01")
HIGH_LOW_WINDOWS = (60, 120, 250)
CHANGE_LOOKBACK = 5
EXTREME_THRESHOLD = 0.05
CHANGE_THRESHOLD = 0.03
SIGNAL_SPECS = (
    ("new_high_low_breadth_top", "top"),
    ("new_high_low_breadth_bottom", "bottom"),
)


def build_new_high_low_breadth_signals(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build symmetric breadth-reversal signals from 60/120/250-day extremes."""

    market = _validate_daily_features(daily_features, start_date=start_date)
    net_columns = [
        f"new_high_low_net_ratio_{window}" for window in HIGH_LOW_WINDOWS
    ]
    valid_columns = [
        f"valid_count_high_low_{window}" for window in HIGH_LOW_WINDOWS
    ]
    market["new_high_low_net_composite"] = market[net_columns].mean(axis=1)
    market["new_high_low_net_change_5d"] = market[
        "new_high_low_net_composite"
    ].diff(CHANGE_LOOKBACK)
    market["change_available"] = market[
        "new_high_low_net_change_5d"
    ].notna()
    market["minimum_high_low_valid_count"] = (
        market[valid_columns].min(axis=1).astype(int)
    )

    top_triggered = (
        market["new_high_low_net_composite"].ge(EXTREME_THRESHOLD)
        & market["new_high_low_net_change_5d"].le(-CHANGE_THRESHOLD)
    )
    bottom_triggered = (
        market["new_high_low_net_composite"].le(-EXTREME_THRESHOLD)
        & market["new_high_low_net_change_5d"].ge(CHANGE_THRESHOLD)
    )
    triggers = {
        "new_high_low_breadth_top": top_triggered,
        "new_high_low_breadth_bottom": bottom_triggered,
    }

    shared: dict[str, object] = {
        "date": market["date"],
        "raw_value": market["new_high_low_net_composite"],
        "universe_size": market["universe_size"],
        "valid_count": market["minimum_high_low_valid_count"],
        "version": version,
        "new_high_low_net_composite": market[
            "new_high_low_net_composite"
        ],
        "new_high_low_net_change_5d": market[
            "new_high_low_net_change_5d"
        ],
        "change_available": market["change_available"],
    }
    for window in HIGH_LOW_WINDOWS:
        for prefix in (
            "new_high_count",
            "new_low_count",
            "new_high_low_net_count",
            "new_high_ratio",
            "new_low_ratio",
            "new_high_low_net_ratio",
            "valid_count_high_low",
        ):
            column = f"{prefix}_{window}"
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
    for window in HIGH_LOW_WINDOWS:
        required.update(
            {
                f"new_high_count_{window}",
                f"new_low_count_{window}",
                f"new_high_low_net_count_{window}",
                f"new_high_ratio_{window}",
                f"new_low_ratio_{window}",
                f"new_high_low_net_ratio_{window}",
                f"valid_count_high_low_{window}",
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
    for window in HIGH_LOW_WINDOWS:
        integer_columns.extend(
            [
                f"new_high_count_{window}",
                f"new_low_count_{window}",
                f"valid_count_high_low_{window}",
            ]
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

    for window in HIGH_LOW_WINDOWS:
        high_count = f"new_high_count_{window}"
        low_count = f"new_low_count_{window}"
        net_count = f"new_high_low_net_count_{window}"
        high_ratio = f"new_high_ratio_{window}"
        low_ratio = f"new_low_ratio_{window}"
        net_ratio = f"new_high_low_net_ratio_{window}"
        valid_count = f"valid_count_high_low_{window}"

        if result[valid_count].eq(0).any():
            raise ValueError(f"{valid_count} must be positive")
        if result[valid_count].gt(result["universe_size"]).any():
            raise ValueError(f"{valid_count} must not exceed universe_size")
        for count_column in (high_count, low_count):
            if result[count_column].gt(result[valid_count]).any():
                raise ValueError(
                    f"{count_column} must not exceed {valid_count}"
                )

        expected_net_count = result[high_count] - result[low_count]
        observed_net_count = pd.to_numeric(result[net_count], errors="coerce")
        if (
            observed_net_count.isna().any()
            or np.not_equal(
                observed_net_count, np.floor(observed_net_count)
            ).any()
            or not observed_net_count.astype(int).equals(expected_net_count)
        ):
            raise ValueError(
                f"{net_count} does not match {high_count} - {low_count}"
            )
        result[net_count] = observed_net_count.astype(int)

        expected_ratios = {
            high_ratio: result[high_count].divide(result[valid_count]),
            low_ratio: result[low_count].divide(result[valid_count]),
            net_ratio: expected_net_count.divide(result[valid_count]),
        }
        for ratio_column, expected in expected_ratios.items():
            values = pd.to_numeric(result[ratio_column], errors="coerce")
            lower_bound = -1.0 if ratio_column == net_ratio else 0.0
            if (
                values.isna().any()
                or not np.isfinite(values.to_numpy(dtype=float)).all()
                or values.lt(lower_bound).any()
                or values.gt(1.0).any()
            ):
                raise ValueError(
                    f"{ratio_column} must contain finite values in range"
                )
            if not np.allclose(values, expected, rtol=1e-9, atol=1e-12):
                raise ValueError(f"{ratio_column} does not match its counts")
            result[ratio_column] = values.astype(float)

    return result.reset_index(drop=True)
