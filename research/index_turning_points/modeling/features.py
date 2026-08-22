"""Causal all-A index features used by the first ML experiment."""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd


RETURN_WINDOWS = (1, 5, 10, 20, 60)
MA_WINDOWS = (20, 60, 120)
RANGE_WINDOWS = (20, 60, 120, 250)
VOLATILITY_WINDOWS = (10, 20, 60)


def point_in_time_directional_state(
    daily: pd.DataFrame,
    *,
    threshold: float,
) -> pd.DataFrame:
    """Return an online pending/up/down state from maximum-tolerance reversals.

    The transition order deliberately matches ``directional_change_labels``:
    a bar that extends the running extreme is not also used to confirm the
    opposite direction because daily OHLC do not reveal intraday ordering.
    Unlike the Plotly background, emitted historical states are never repainted
    after a later confirmation.
    """

    prices = _validate_daily(daily)
    threshold = float(threshold)
    if not isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")

    highs = prices["high"].to_numpy(dtype=float)
    lows = prices["low"].to_numpy(dtype=float)
    states = np.full(len(prices), "pending", dtype=object)
    running_high = highs[0]
    running_low = lows[0]
    high_position = low_position = 0
    mode: str | None = None
    extreme = np.nan
    up_factor = 1.0 + threshold
    down_factor = 1.0 - threshold

    for position in range(1, len(prices)):
        bar_high = highs[position]
        bar_low = lows[position]
        if mode is None:
            bottom_trigger = (
                bar_low > running_low and bar_high / running_low >= up_factor
            )
            top_trigger = (
                bar_high < running_high and bar_low / running_high <= down_factor
            )
            initial_event: str | None = None
            if bottom_trigger and top_trigger:
                if low_position > high_position:
                    initial_event = "bottom"
                elif high_position > low_position:
                    initial_event = "top"
            elif bottom_trigger:
                initial_event = "bottom"
            elif top_trigger:
                initial_event = "top"

            if initial_event == "bottom":
                mode = "up"
                extreme = bar_high
                states[position] = "up"
                continue
            if initial_event == "top":
                mode = "down"
                extreme = bar_low
                states[position] = "down"
                continue
            if bar_high >= running_high:
                running_high = bar_high
                high_position = position
            if bar_low <= running_low:
                running_low = bar_low
                low_position = position
            continue

        if mode == "up":
            if bar_high >= extreme:
                extreme = bar_high
                states[position] = "up"
            elif bar_low / extreme <= down_factor:
                mode = "down"
                extreme = bar_low
                states[position] = "down"
            else:
                states[position] = "pending"
        else:
            if bar_low <= extreme:
                extreme = bar_low
                states[position] = "down"
            elif bar_high / extreme >= up_factor:
                mode = "up"
                extreme = bar_high
                states[position] = "up"
            else:
                states[position] = "pending"

    result = pd.DataFrame(
        {"date": prices.index, "index_phase_pti": states}
    )
    for state in ("pending", "up", "down"):
        result[f"index_phase_{state}"] = result["index_phase_pti"].eq(state)
    return result


def build_index_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Build normalized causal OHLC features; raw index levels are excluded."""

    prices = _validate_daily(daily, require_ohlc=True)
    close = prices["close"]
    result = pd.DataFrame(index=prices.index)
    result.index.name = "date"
    result["index_price_available"] = True
    for window in RETURN_WINDOWS:
        result[f"index_return_{window}d"] = close.pct_change(window, fill_method=None)
    result["index_range_1d"] = (prices["high"] - prices["low"]) / close
    for window in MA_WINDOWS:
        moving_average = close.rolling(window, min_periods=window).mean()
        result[f"index_close_to_ma{window}"] = close / moving_average - 1.0
    for window in RANGE_WINDOWS:
        rolling_high = prices["high"].rolling(window, min_periods=window).max()
        rolling_low = prices["low"].rolling(window, min_periods=window).min()
        result[f"index_drawdown_{window}d"] = close / rolling_high - 1.0
        result[f"index_rebound_{window}d"] = close / rolling_low - 1.0
    return_1d = close.pct_change(fill_method=None)
    for window in VOLATILITY_WINDOWS:
        result[f"index_volatility_{window}d"] = return_1d.rolling(
            window, min_periods=window
        ).std(ddof=0)
    return result.reset_index()


def _validate_daily(
    daily: pd.DataFrame,
    *,
    require_ohlc: bool = False,
) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    required = {"high", "low"}
    if require_ohlc:
        required.update({"open", "close"})
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(f"daily is missing columns: {sorted(missing)}")
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise TypeError("daily must use a DatetimeIndex")
    if daily.empty or not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be non-empty, unique and increasing")
    columns = [column for column in ("open", "high", "low", "close") if column in required]
    result = daily[columns].copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all() or (result <= 0.0).any().any():
        raise ValueError("daily OHLC values must be positive and finite")
    if (result["high"] < result["low"]).any():
        raise ValueError("daily high must be greater than or equal to low")
    if require_ohlc and (
        (result["high"] < result[["open", "close"]].max(axis=1)).any()
        or (result["low"] > result[["open", "close"]].min(axis=1)).any()
    ):
        raise ValueError("daily contains invalid OHLC ordering")
    return result
