import numpy as np
import pandas as pd
import pytest

from research.index_turning_points import directional_change_labels


def prices(values, index=None):
    return pd.Series(
        values,
        index=(
            index
            if index is not None
            else pd.date_range("2020-01-01", periods=len(values), freq="D")
        ),
        dtype=float,
    )


def labels_from_single_price(values, threshold=0.10):
    price = prices(values)
    return directional_change_labels(price, price, threshold=threshold)


def test_labels_alternating_confirmed_events_and_final_candidate():
    high = prices([101, 96, 106, 112, 100, 91, 100, 107])
    low = prices([99, 94, 103, 109, 98, 89, 98, 104])

    labels = directional_change_labels(high, low, threshold=0.10)

    assert labels["event_type"].tolist() == ["bottom", "top", "bottom", "top"]
    assert labels["status"].tolist() == [
        "confirmed",
        "confirmed",
        "confirmed",
        "unconfirmed",
    ]
    assert labels["eligible"].tolist() == [False, True, True, False]
    assert labels["anchor_position"].tolist() == [1, 3, 5, 7]
    assert labels["anchor_price"].tolist() == [94, 112, 89, 107]
    assert labels["confirmation_position"].tolist()[:3] == [2, 4, 6]
    assert labels["confirmation_price"].tolist()[:3] == [106, 98, 100]
    assert labels["confirmation_lag"].tolist()[:3] == [1, 1, 1]
    assert pd.isna(labels.iloc[-1]["confirmation_date"])


def test_threshold_is_inclusive_and_reversal_return_is_recorded():
    labels = labels_from_single_price([100, 110, 99])

    assert labels.iloc[0]["event_type"] == "bottom"
    assert labels.iloc[0]["confirmation_position"] == 1
    assert labels.iloc[1]["event_type"] == "top"
    assert labels.iloc[1]["status"] == "confirmed"
    assert labels.iloc[1]["anchor_price"] == 110
    assert labels.iloc[1]["confirmation_price"] == 99
    assert labels.iloc[1]["reversal_return"] == pytest.approx(-0.10)


def test_initial_decline_establishes_top_then_eligible_bottom():
    labels = labels_from_single_price([100, 90, 99])

    assert labels["event_type"].tolist() == ["top", "bottom", "top"]
    assert labels["status"].tolist() == ["confirmed", "confirmed", "unconfirmed"]
    assert labels["eligible"].tolist() == [False, True, False]
    assert labels.iloc[1]["reversal_return"] == pytest.approx(0.10)


def test_equal_extreme_uses_latest_occurrence():
    labels = labels_from_single_price([100, 95, 95, 104.5, 110, 105, 110, 99])

    assert labels.iloc[0]["event_type"] == "bottom"
    assert labels.iloc[0]["anchor_position"] == 2
    assert labels.iloc[1]["event_type"] == "top"
    assert labels.iloc[1]["anchor_position"] == 6


def test_same_bar_high_low_span_cannot_anchor_and_confirm():
    high = prices([100, 110, 109])
    low = prices([100, 90, 100])

    labels = directional_change_labels(high, low, threshold=0.10)

    assert labels.iloc[0]["event_type"] == "bottom"
    assert labels.iloc[0]["anchor_position"] == 1
    assert labels.iloc[0]["confirmation_position"] == 2
    assert labels.iloc[0]["anchor_price"] == 90
    assert labels.iloc[0]["confirmation_price"] == 109


def test_new_extreme_bar_waits_for_a_later_confirmation_bar():
    high = prices([100, 110, 120, 119])
    low = prices([100, 110, 99, 108])

    labels = directional_change_labels(high, low, threshold=0.10)

    top = labels[labels["event_type"] == "top"].iloc[0]
    assert top["anchor_position"] == 2
    assert top["confirmation_position"] == 3
    assert top["anchor_price"] == 120
    assert top["confirmation_price"] == 108


def test_no_initial_direction_returns_empty_labels():
    labels = labels_from_single_price([100, 103, 97, 101])

    assert labels.empty
    assert list(labels.columns) == [
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


@pytest.mark.parametrize("threshold", [0, 1, -0.1, np.nan, np.inf])
def test_rejects_invalid_threshold(threshold):
    price = prices([100, 110])
    with pytest.raises(ValueError, match="threshold"):
        directional_change_labels(price, price, threshold=threshold)


@pytest.mark.parametrize(
    "high, low, message",
    [
        (prices([100, np.nan]), prices([99, 100]), "finite"),
        (prices([100, 110]), prices([99, 0]), "positive"),
        (prices([100, np.inf]), prices([99, 100]), "finite"),
        (
            pd.Series([100, 110], index=[2, 1]),
            pd.Series([99, 109], index=[2, 1]),
            "increasing",
        ),
        (
            pd.Series([100, 110], index=[1, 1]),
            pd.Series([99, 109], index=[1, 1]),
            "unique",
        ),
        (prices([100, 110]), prices([101, 109]), "greater than or equal"),
    ],
)
def test_rejects_invalid_high_low_series(high, low, message):
    with pytest.raises(ValueError, match=message):
        directional_change_labels(high, low, threshold=0.10)


def test_rejects_mismatched_indexes():
    high = prices([100, 110])
    low = prices([99, 109], index=pd.date_range("2020-02-01", periods=2))

    with pytest.raises(ValueError, match="indexes must match"):
        directional_change_labels(high, low, threshold=0.10)


def test_rejects_non_series_input():
    price = prices([100, 110])
    with pytest.raises(TypeError, match="pandas Series"):
        directional_change_labels([100, 110], price, threshold=0.10)
