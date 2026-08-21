"""Causal single-period MA breadth signals for horizon comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "ma_period_breadth_decomposition_v1_20120104_20260814"
REQUESTED_START_DATE = pd.Timestamp("2012-01-01")
MA_WINDOWS = (20, 60, 120)
CHANGE_LOOKBACK = 5
EXTREME_THRESHOLD = 0.70
CHANGE_THRESHOLD = 0.05


def build_ma_period_breadth_signals(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build comparable top/bottom signals for each MA breadth horizon."""

    market = _validate_daily_features(daily_features, start_date=start_date)
    signal_frames = []
    for window in MA_WINDOWS:
        breadth_column = f"breadth_ma{window}"
        above_column = f"above_count_ma{window}"
        valid_column = f"valid_count_ma{window}"
        change = market[breadth_column].diff(CHANGE_LOOKBACK)
        triggers = {
            "top": market[breadth_column].ge(EXTREME_THRESHOLD)
            & change.le(-CHANGE_THRESHOLD),
            "bottom": market[breadth_column].le(1.0 - EXTREME_THRESHOLD)
            & change.ge(CHANGE_THRESHOLD),
        }
        for direction in ("top", "bottom"):
            signal_id = f"ma{window}_breadth_reversal_{direction}"
            signal_frames.append(
                pd.DataFrame(
                    {
                        "date": market["date"],
                        "signal_id": signal_id,
                        "direction": direction,
                        "raw_value": market[breadth_column],
                        "triggered": triggers[direction],
                        "universe_size": market["universe_size"],
                        "valid_count": market[valid_column],
                        "version": version,
                        "ma_window": window,
                        "ma_breadth": market[breadth_column],
                        "breadth_change_5d": change,
                        "change_available": change.notna(),
                        "above_count": market[above_column],
                        "ma_valid_count": market[valid_column],
                    }
                )
            )

    source = pd.concat(signal_frames, ignore_index=True)
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    series_counts = []
    for window in MA_WINDOWS:
        for direction in ("top", "bottom"):
            signal_id = f"ma{window}_breadth_reversal_{direction}"
            signal_daily = daily[daily["signal_id"].eq(signal_id)]
            signal_episodes = episodes[episodes["signal_id"].eq(signal_id)]
            series_counts.append(
                {
                    "signal_id": signal_id,
                    "ma_window": window,
                    "direction": direction,
                    "triggered_days": int(signal_daily["triggered"].sum()),
                    "episodes": len(signal_episodes),
                }
            )

    metadata = {
        "signal_version": version,
        "requested_start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "comparison_start_date": market["date"].min().strftime("%Y-%m-%d"),
        "comparison_end_date": market["date"].max().strftime("%Y-%m-%d"),
        "change_available_from": market["date"].iloc[CHANGE_LOOKBACK].strftime(
            "%Y-%m-%d"
        ),
        "trade_dates": len(market),
        "daily_rows": len(daily),
        "signal_series": len(series_counts),
        "series_counts": series_counts,
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
    if len(result) <= CHANGE_LOOKBACK:
        raise ValueError("daily_features does not cover the five-day change lookback")

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
