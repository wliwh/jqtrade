import numpy as np
import pandas as pd
import pytest

from research.index_turning_points import directional_change_labels


def prices(values):
    return pd.Series(
        values,
        index=pd.date_range("2020-01-01", periods=len(values), freq="D"),
        dtype=float,
    )


def test_labels_alternating_confirmed_events_and_final_candidate():
    close = prices([100, 95, 104.5, 110, 99, 90, 99, 105])

    labels = directional_change_labels(close, threshold=0.10)

    assert labels["event_type"].tolist() == ["bottom", "top", "bottom", "top"]
    assert labels["status"].tolist() == [
        "confirmed",
        "confirmed",
        "confirmed",
        "unconfirmed",
    ]
    assert labels["eligible"].tolist() == [False, True, True, False]
    assert labels["anchor_position"].tolist() == [1, 3, 5, 7]
    assert labels["confirmation_position"].tolist()[:3] == [2, 4, 6]
    assert labels["confirmation_lag"].tolist()[:3] == [1, 1, 1]
    assert pd.isna(labels.iloc[-1]["confirmation_date"])


def test_threshold_is_inclusive_and_reversal_return_is_recorded():
    close = prices([100, 110, 99])

    labels = directional_change_labels(close, threshold=0.10)

    assert labels.iloc[0]["event_type"] == "bottom"
    assert labels.iloc[0]["confirmation_position"] == 1
    assert labels.iloc[1]["event_type"] == "top"
    assert labels.iloc[1]["status"] == "confirmed"
    assert labels.iloc[1]["anchor_price"] == 110
    assert labels.iloc[1]["confirmation_price"] == 99
    assert labels.iloc[1]["reversal_return"] == pytest.approx(-0.10)


def test_initial_decline_establishes_top_then_eligible_bottom():
    close = prices([100, 90, 99])

    labels = directional_change_labels(close, threshold=0.10)

    assert labels["event_type"].tolist() == ["top", "bottom", "top"]
    assert labels["status"].tolist() == ["confirmed", "confirmed", "unconfirmed"]
    assert labels["eligible"].tolist() == [False, True, False]
    assert labels.iloc[1]["reversal_return"] == pytest.approx(0.10)


def test_equal_extreme_uses_latest_occurrence():
    close = prices([100, 95, 95, 104.5, 110, 105, 110, 99])

    labels = directional_change_labels(close, threshold=0.10)

    assert labels.iloc[0]["event_type"] == "bottom"
    assert labels.iloc[0]["anchor_position"] == 2
    assert labels.iloc[1]["event_type"] == "top"
    assert labels.iloc[1]["anchor_position"] == 6


def test_no_initial_direction_returns_empty_labels():
    labels = directional_change_labels(prices([100, 103, 97, 101]), threshold=0.10)

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
    with pytest.raises(ValueError, match="threshold"):
        directional_change_labels(prices([100, 110]), threshold=threshold)


@pytest.mark.parametrize(
    "close, message",
    [
        (prices([100, np.nan]), "finite"),
        (prices([100, 0]), "positive"),
        (prices([100, np.inf]), "finite"),
        (pd.Series([100, 110], index=[2, 1]), "increasing"),
        (pd.Series([100, 110], index=[1, 1]), "unique"),
    ],
)
def test_rejects_invalid_close_series(close, message):
    with pytest.raises(ValueError, match=message):
        directional_change_labels(close, threshold=0.10)


def test_rejects_non_series_input():
    with pytest.raises(TypeError, match="pandas Series"):
        directional_change_labels([100, 110], threshold=0.10)
