import pandas as pd

from research.index_turning_points.modeling.features import (
    build_index_features,
    point_in_time_directional_state,
)
from research.index_turning_points.ground_truth.labels import directional_change_labels


def _single_price(values):
    dates = pd.bdate_range("2020-01-01", periods=len(values))
    price = pd.Series(values, index=dates, dtype=float)
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price}
    )


def test_maximum_tolerance_state_is_online_and_has_pending_reversals():
    daily = _single_price([100, 95, 105, 110, 106, 112, 101, 100, 102, 88, 96, 97])

    result = point_in_time_directional_state(daily, threshold=0.10)

    assert result["index_phase_pti"].tolist() == [
        "pending",
        "pending",
        "up",
        "up",
        "pending",
        "up",
        "pending",
        "down",
        "pending",
        "down",
        "pending",
        "up",
    ]
    assert result[["index_phase_pending", "index_phase_up", "index_phase_down"]].sum(axis=1).eq(1).all()


def test_maximum_tolerance_state_is_invariant_when_future_rows_are_removed():
    daily = _single_price([100, 95, 105, 110, 106, 112, 101, 100, 102, 88, 96, 97])
    cutoff = daily.index[8]

    full = point_in_time_directional_state(daily, threshold=0.10)
    truncated = point_in_time_directional_state(
        daily.loc[:cutoff], threshold=0.10
    )

    pd.testing.assert_frame_equal(full.iloc[: len(truncated)], truncated)


def test_state_switches_match_directional_change_confirmation_dates():
    daily = _single_price([100, 95, 105, 110, 106, 112, 101, 100, 102, 88, 96, 97])
    states = point_in_time_directional_state(daily, threshold=0.10).set_index("date")
    labels = directional_change_labels(daily["high"], daily["low"], threshold=0.10)

    confirmed = labels[labels["status"].eq("confirmed")]
    for row in confirmed.itertuples(index=False):
        expected = "up" if row.event_type == "bottom" else "down"
        assert states.loc[row.confirmation_date, "index_phase_pti"] == expected


def test_index_features_are_normalized_and_do_not_export_raw_levels():
    daily = _single_price(range(100, 370))

    result = build_index_features(daily)

    assert {"open", "high", "low", "close"}.isdisjoint(result.columns)
    assert result["index_return_1d"].iloc[-1] == 369 / 368 - 1
    assert result["index_close_to_ma120"].iloc[-1] > 0
    assert result["index_drawdown_250d"].iloc[-1] == 0
