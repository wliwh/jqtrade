import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.new_high_low_period_decomposition import (
    build_new_high_low_period_signals,
)


def _daily_rows():
    dates = pd.bdate_range("2020-01-01", periods=16)
    values = {
        60: [0.10, 0.10, 0.10, 0.10, 0.10, 0.06, 0.07, 0.10] + [0.0] * 8,
        120: [-0.10, -0.10, -0.10, -0.10, -0.10, -0.06, -0.07, -0.10]
        + [0.0] * 8,
        250: [0.0] * 16,
    }
    records = []
    for position, date in enumerate(dates):
        record = {"date": date, "universe_size": 120}
        for window in (60, 120, 250):
            valid_count = 100
            net_value = values[window][position]
            high_count = round(max(net_value, 0) * valid_count)
            low_count = round(max(-net_value, 0) * valid_count)
            net_count = high_count - low_count
            record[f"new_high_count_{window}"] = high_count
            record[f"new_low_count_{window}"] = low_count
            record[f"new_high_low_net_count_{window}"] = net_count
            record[f"new_high_ratio_{window}"] = high_count / valid_count
            record[f"new_low_ratio_{window}"] = low_count / valid_count
            record[f"new_high_low_net_ratio_{window}"] = net_count / valid_count
            record[f"valid_count_high_low_{window}"] = valid_count
        records.append(record)
    return pd.DataFrame(records)


def test_builds_six_independent_period_direction_series():
    source = _daily_rows()

    daily, episodes, metadata = build_new_high_low_period_signals(
        source, start_date="2020-01-01"
    )

    assert daily["signal_id"].nunique() == 6
    high_low_60_top = daily[
        daily["signal_id"].eq("new_high_low_60_breadth_reversal_top")
    ]
    high_low_120_bottom = daily[
        daily["signal_id"].eq("new_high_low_120_breadth_reversal_bottom")
    ]
    expected_dates = [source["date"].iloc[5], source["date"].iloc[6]]
    assert high_low_60_top.loc[
        high_low_60_top["triggered"], "date"
    ].tolist() == expected_dates
    assert high_low_120_bottom.loc[
        high_low_120_bottom["triggered"], "date"
    ].tolist() == expected_dates
    assert high_low_60_top["raw_value"].equals(
        high_low_60_top["new_high_low_net_ratio"]
    )
    assert len(episodes) == 2
    assert set(episodes["signal_id"]) == {
        "new_high_low_60_breadth_reversal_top",
        "new_high_low_120_breadth_reversal_bottom",
    }
    assert metadata["signal_series"] == 6


def test_events_are_invariant_when_input_is_truncated():
    source = _daily_rows()
    full, _, _ = build_new_high_low_period_signals(
        source, start_date="2020-01-01"
    )
    cutoff = source["date"].iloc[10]
    truncated, _, _ = build_new_high_low_period_signals(
        source[source["date"].le(cutoff)], start_date="2020-01-01"
    )
    event_columns = [
        "date",
        "signal_id",
        "direction",
        "raw_value",
        "new_high_low_net_change_5d",
        "triggered",
        "episode_id",
        "episode_stage",
        "event_onset",
        "event_continuation",
        "event_exit",
        "event_capped_confirmation",
    ]

    expected = full[full["date"].le(cutoff)][event_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected,
        truncated[event_columns].reset_index(drop=True),
    )


def test_rejects_net_ratio_that_does_not_match_counts():
    source = _daily_rows()
    source.loc[0, "new_high_low_net_ratio_250"] = 0.123

    with pytest.raises(ValueError, match="does not match"):
        build_new_high_low_period_signals(source, start_date="2020-01-01")


def test_rejects_sample_shorter_than_change_lookback():
    source = _daily_rows().iloc[:5]

    with pytest.raises(ValueError, match="five-day change lookback"):
        build_new_high_low_period_signals(source, start_date="2020-01-01")
