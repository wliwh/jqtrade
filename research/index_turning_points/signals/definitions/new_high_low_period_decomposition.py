"""Causal single-period new-high versus new-low breadth signals."""

from __future__ import annotations

import pandas as pd

from ..events import build_signal_events
from .new_high_low_breadth import (
    CHANGE_LOOKBACK,
    CHANGE_THRESHOLD,
    EXTREME_THRESHOLD,
    HIGH_LOW_WINDOWS,
    REQUESTED_START_DATE,
    _validate_daily_features,
)


SIGNAL_VERSION = "new_high_low_period_decomposition_v1_20120104_20260814"


def build_new_high_low_period_signals(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build comparable top/bottom signals for each high-low horizon."""

    market = _validate_daily_features(daily_features, start_date=start_date)
    if len(market) <= CHANGE_LOOKBACK:
        raise ValueError("daily_features does not cover the five-day change lookback")

    signal_frames = []
    for window in HIGH_LOW_WINDOWS:
        high_count = f"new_high_count_{window}"
        low_count = f"new_low_count_{window}"
        net_count = f"new_high_low_net_count_{window}"
        high_ratio = f"new_high_ratio_{window}"
        low_ratio = f"new_low_ratio_{window}"
        net_ratio = f"new_high_low_net_ratio_{window}"
        valid_count = f"valid_count_high_low_{window}"
        change = market[net_ratio].diff(CHANGE_LOOKBACK)
        triggers = {
            "top": market[net_ratio].ge(EXTREME_THRESHOLD)
            & change.le(-CHANGE_THRESHOLD),
            "bottom": market[net_ratio].le(-EXTREME_THRESHOLD)
            & change.ge(CHANGE_THRESHOLD),
        }
        for direction in ("top", "bottom"):
            signal_id = f"new_high_low_{window}_breadth_reversal_{direction}"
            signal_frames.append(
                pd.DataFrame(
                    {
                        "date": market["date"],
                        "signal_id": signal_id,
                        "direction": direction,
                        "raw_value": market[net_ratio],
                        "triggered": triggers[direction],
                        "universe_size": market["universe_size"],
                        "valid_count": market[valid_count],
                        "version": version,
                        "high_low_window": window,
                        "new_high_low_net_ratio": market[net_ratio],
                        "new_high_low_net_change_5d": change,
                        "change_available": change.notna(),
                        "new_high_count": market[high_count],
                        "new_low_count": market[low_count],
                        "new_high_low_net_count": market[net_count],
                        "new_high_ratio": market[high_ratio],
                        "new_low_ratio": market[low_ratio],
                        "high_low_valid_count": market[valid_count],
                    }
                )
            )

    source = pd.concat(signal_frames, ignore_index=True)
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    series_counts = []
    for window in HIGH_LOW_WINDOWS:
        for direction in ("top", "bottom"):
            signal_id = f"new_high_low_{window}_breadth_reversal_{direction}"
            signal_daily = daily[daily["signal_id"].eq(signal_id)]
            signal_episodes = episodes[episodes["signal_id"].eq(signal_id)]
            series_counts.append(
                {
                    "signal_id": signal_id,
                    "high_low_window": window,
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
