"""Close-price directional-change labels for index research."""

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
    close: pd.Series,
    threshold: float = 0.10,
) -> pd.DataFrame:
    """Label close-price tops and bottoms confirmed by a relative reversal.

    The first confirmed pivot only establishes the initial direction and is
    returned with ``eligible=False``. Later confirmed pivots are eligible for
    the main research sample. If direction has been established, the final
    running extreme is returned as one ``unconfirmed`` row.

    ``confirmation_lag`` is measured in observations (trading bars), not
    calendar days. Equal running extremes use their latest occurrence.

    Args:
        close: Positive, finite close prices with a unique, increasing index.
        threshold: Relative reversal threshold in the open interval ``(0, 1)``.

    Returns:
        A DataFrame with confirmed events followed by the final unconfirmed
        candidate. An empty frame means that no initial direction was
        established within the sample.
    """

    values = _validate_inputs(close, threshold)
    if len(values) < 2:
        return _empty_labels()

    dates = close.index
    records: list[dict[str, object]] = []
    up_factor = 1.0 + threshold
    down_factor = 1.0 - threshold

    high_price = low_price = values[0]
    high_position = low_position = 0
    mode: str | None = None
    next_position = len(values)

    for position in range(1, len(values)):
        price = values[position]

        if price >= high_price:
            high_price = price
            high_position = position
        if price <= low_price:
            low_price = price
            low_position = position

        if price / low_price >= up_factor:
            records.append(
                _confirmed_record(
                    event_type="bottom",
                    anchor_position=low_position,
                    anchor_price=low_price,
                    confirmation_position=position,
                    confirmation_price=price,
                    dates=dates,
                    threshold=threshold,
                    eligible=False,
                )
            )
            mode = "up"
        elif price / high_price <= down_factor:
            records.append(
                _confirmed_record(
                    event_type="top",
                    anchor_position=high_position,
                    anchor_price=high_price,
                    confirmation_position=position,
                    confirmation_price=price,
                    dates=dates,
                    threshold=threshold,
                    eligible=False,
                )
            )
            mode = "down"

        if mode is not None:
            extreme_price = price
            extreme_position = position
            next_position = position + 1
            break

    if mode is None:
        return _empty_labels()

    for position in range(next_position, len(values)):
        price = values[position]

        if mode == "up":
            if price >= extreme_price:
                extreme_price = price
                extreme_position = position
            elif price / extreme_price <= down_factor:
                records.append(
                    _confirmed_record(
                        event_type="top",
                        anchor_position=extreme_position,
                        anchor_price=extreme_price,
                        confirmation_position=position,
                        confirmation_price=price,
                        dates=dates,
                        threshold=threshold,
                        eligible=True,
                    )
                )
                mode = "down"
                extreme_price = price
                extreme_position = position
        else:
            if price <= extreme_price:
                extreme_price = price
                extreme_position = position
            elif price / extreme_price >= up_factor:
                records.append(
                    _confirmed_record(
                        event_type="bottom",
                        anchor_position=extreme_position,
                        anchor_price=extreme_price,
                        confirmation_position=position,
                        confirmation_price=price,
                        dates=dates,
                        threshold=threshold,
                        eligible=True,
                    )
                )
                mode = "up"
                extreme_price = price
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


def _validate_inputs(close: pd.Series, threshold: float) -> np.ndarray:
    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if not isfinite(float(threshold)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")
    if not close.index.is_unique:
        raise ValueError("close index must be unique")
    if not close.index.is_monotonic_increasing:
        raise ValueError("close index must be increasing")

    try:
        values = close.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("close prices must be numeric") from exc

    if not np.isfinite(values).all():
        raise ValueError("close prices must be finite and non-missing")
    if (values <= 0.0).any():
        raise ValueError("close prices must be strictly positive")
    return values


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
