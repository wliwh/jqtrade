"""High/low directional-change labels for index research."""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd


LABEL_COLUMNS = [
    "event_type",
    "status",
    "eligible",
    "anchor_date",
    "anchor_position",
    "anchor_price",
    "confirmation_date",
    "confirmation_position",
    "confirmation_price",
    "confirmation_lag",
    "reversal_return",
    "threshold",
]


def directional_change_labels(
    high: pd.Series,
    low: pd.Series,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Label tops from daily highs and bottoms from daily lows.

    A top is anchored at a running high and confirmed when a later bar's low
    falls by ``threshold``. A bottom is anchored at a running low and confirmed
    when a later bar's high rises by ``threshold``. An anchor is never confirmed
    on the same daily bar because OHLC data do not reveal the intraday order of
    the high and low.

    The first confirmed pivot only establishes the initial direction and is
    returned with ``eligible=False``. Later confirmed pivots are eligible for
    the main research sample. The final running extreme is returned as one
    ``unconfirmed`` row.

    ``confirmation_lag`` is measured in observations (trading bars), not
    calendar days. Equal running extremes use their latest occurrence.

    Args:
        high: Positive, finite daily highs with a unique, increasing index.
        low: Positive, finite daily lows aligned exactly with ``high``.
        threshold: Relative reversal threshold in the open interval ``(0, 1)``.

    Returns:
        Confirmed events followed by the final unconfirmed candidate. An empty
        frame means no initial direction was established within the sample.
    """

    high_values, low_values = _validate_inputs(high, low, threshold)
    if len(high_values) < 2:
        return _empty_labels()

    dates = high.index
    records: list[dict[str, object]] = []
    up_factor = 1.0 + threshold
    down_factor = 1.0 - threshold

    running_high = high_values[0]
    running_low = low_values[0]
    high_position = low_position = 0
    mode: str | None = None
    next_position = len(high_values)

    for position in range(1, len(high_values)):
        bar_high = high_values[position]
        bar_low = low_values[position]

        bottom_trigger = bar_low > running_low and bar_high / running_low >= up_factor
        top_trigger = bar_high < running_high and bar_low / running_high <= down_factor
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
            records.append(
                _confirmed_record(
                    event_type="bottom",
                    anchor_position=low_position,
                    anchor_price=running_low,
                    confirmation_position=position,
                    confirmation_price=bar_high,
                    dates=dates,
                    threshold=threshold,
                    eligible=False,
                )
            )
            mode = "up"
            extreme_price = bar_high
            extreme_position = position
            next_position = position + 1
            break

        if initial_event == "top":
            records.append(
                _confirmed_record(
                    event_type="top",
                    anchor_position=high_position,
                    anchor_price=running_high,
                    confirmation_position=position,
                    confirmation_price=bar_low,
                    dates=dates,
                    threshold=threshold,
                    eligible=False,
                )
            )
            mode = "down"
            extreme_price = bar_low
            extreme_position = position
            next_position = position + 1
            break

        if bar_high >= running_high:
            running_high = bar_high
            high_position = position
        if bar_low <= running_low:
            running_low = bar_low
            low_position = position

    if mode is None:
        return _empty_labels()

    for position in range(next_position, len(high_values)):
        bar_high = high_values[position]
        bar_low = low_values[position]

        if mode == "up":
            if bar_high >= extreme_price:
                extreme_price = bar_high
                extreme_position = position
            elif bar_low / extreme_price <= down_factor:
                records.append(
                    _confirmed_record(
                        event_type="top",
                        anchor_position=extreme_position,
                        anchor_price=extreme_price,
                        confirmation_position=position,
                        confirmation_price=bar_low,
                        dates=dates,
                        threshold=threshold,
                        eligible=True,
                    )
                )
                mode = "down"
                extreme_price = bar_low
                extreme_position = position
        else:
            if bar_low <= extreme_price:
                extreme_price = bar_low
                extreme_position = position
            elif bar_high / extreme_price >= up_factor:
                records.append(
                    _confirmed_record(
                        event_type="bottom",
                        anchor_position=extreme_position,
                        anchor_price=extreme_price,
                        confirmation_position=position,
                        confirmation_price=bar_high,
                        dates=dates,
                        threshold=threshold,
                        eligible=True,
                    )
                )
                mode = "up"
                extreme_price = bar_high
                extreme_position = position

    records.append(
        {
            "event_type": "top" if mode == "up" else "bottom",
            "status": "unconfirmed",
            "eligible": False,
            "anchor_date": dates[extreme_position],
            "anchor_position": extreme_position,
            "anchor_price": extreme_price,
            "confirmation_date": pd.NaT,
            "confirmation_position": pd.NA,
            "confirmation_price": pd.NA,
            "confirmation_lag": pd.NA,
            "reversal_return": pd.NA,
            "threshold": threshold,
        }
    )

    return _labels_frame(records)


def _validate_inputs(
    high: pd.Series,
    low: pd.Series,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(high, pd.Series) or not isinstance(low, pd.Series):
        raise TypeError("high and low must be pandas Series")
    if not isfinite(float(threshold)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    if not high.index.equals(low.index):
        raise ValueError("high and low indexes must match")
    if not high.index.is_unique:
        raise ValueError("high and low index must be unique")
    if not high.index.is_monotonic_increasing:
        raise ValueError("high and low index must be increasing")

    try:
        high_values = high.to_numpy(dtype=float, na_value=np.nan)
        low_values = low.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("high and low prices must be numeric") from exc

    if not np.isfinite(high_values).all() or not np.isfinite(low_values).all():
        raise ValueError("high and low prices must be finite and non-missing")
    if (high_values <= 0.0).any() or (low_values <= 0.0).any():
        raise ValueError("high and low prices must be strictly positive")
    if (high_values < low_values).any():
        raise ValueError("daily high must be greater than or equal to daily low")
    return high_values, low_values


def _confirmed_record(
    *,
    event_type: str,
    anchor_position: int,
    anchor_price: float,
    confirmation_position: int,
    confirmation_price: float,
    dates: pd.Index,
    threshold: float,
    eligible: bool,
) -> dict[str, object]:
    return {
        "event_type": event_type,
        "status": "confirmed",
        "eligible": eligible,
        "anchor_date": dates[anchor_position],
        "anchor_position": anchor_position,
        "anchor_price": anchor_price,
        "confirmation_date": dates[confirmation_position],
        "confirmation_position": confirmation_position,
        "confirmation_price": confirmation_price,
        "confirmation_lag": confirmation_position - anchor_position,
        "reversal_return": confirmation_price / anchor_price - 1.0,
        "threshold": threshold,
    }


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=LABEL_COLUMNS)


def _labels_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    labels = pd.DataFrame.from_records(records, columns=LABEL_COLUMNS)
    labels["eligible"] = labels["eligible"].astype(bool)
    labels["anchor_position"] = labels["anchor_position"].astype("int64")
    labels["confirmation_position"] = labels["confirmation_position"].astype("Int64")
    labels["confirmation_lag"] = labels["confirmation_lag"].astype("Int64")
    labels["anchor_price"] = labels["anchor_price"].astype("float64")
    labels["confirmation_price"] = labels["confirmation_price"].astype("Float64")
    labels["reversal_return"] = labels["reversal_return"].astype("Float64")
    labels["threshold"] = labels["threshold"].astype("float64")
    return labels
