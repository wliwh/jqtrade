import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.ma_period_breadth_decomposition import (
    build_ma_period_breadth_signals,
)


def _daily_rows():
    dates = pd.bdate_range("2020-01-01", periods=16)
    values = {
        20: [
            0.80,
            0.81,
            0.82,
            0.83,
            0.84,
            0.74,
            0.75,
            0.60,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
        ],
        60: [
            0.20,
            0.19,
            0.18,
            0.17,
            0.16,
            0.26,
            0.25,
            0.40,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
            0.50,
        ],
        120: [0.50] * 16,
    }
    records = []
    for position, date in enumerate(dates):
        record = {"date": date, "universe_size": 100}
        for window in (20, 60, 120):
            above_count = round(values[window][position] * 100)
            record[f"above_count_ma{window}"] = above_count
            record[f"valid_count_ma{window}"] = 100
            record[f"breadth_ma{window}"] = above_count / 100
        records.append(record)
    return pd.DataFrame(records)


def test_builds_six_independent_period_direction_series():
    daily, episodes, metadata = build_ma_period_breadth_signals(
        _daily_rows(), start_date="2020-01-01"
    )

    assert daily["signal_id"].nunique() == 6
    ma20_top = daily[daily["signal_id"].eq("ma20_breadth_reversal_top")]
    ma60_bottom = daily[daily["signal_id"].eq("ma60_breadth_reversal_bottom")]
    assert ma20_top.loc[ma20_top["triggered"], "date"].tolist() == [
        pd.Timestamp("2020-01-08"),
        pd.Timestamp("2020-01-09"),
    ]
    assert ma60_bottom.loc[ma60_bottom["triggered"], "date"].tolist() == [
        pd.Timestamp("2020-01-08"),
        pd.Timestamp("2020-01-09"),
    ]
    assert ma20_top["raw_value"].equals(ma20_top["ma_breadth"])
    assert len(episodes) == 2
    assert set(episodes["signal_id"]) == {
        "ma20_breadth_reversal_top",
        "ma60_breadth_reversal_bottom",
    }
    assert metadata["signal_series"] == 6


def test_events_are_invariant_when_input_is_truncated():
    source = _daily_rows()
    full, _, _ = build_ma_period_breadth_signals(
        source, start_date="2020-01-01"
    )
    cutoff = source["date"].iloc[10]
    truncated, _, _ = build_ma_period_breadth_signals(
        source[source["date"].le(cutoff)], start_date="2020-01-01"
    )
    event_columns = [
        "date",
        "signal_id",
        "direction",
        "raw_value",
        "breadth_change_5d",
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


def test_rejects_breadth_that_does_not_match_counts():
    source = _daily_rows()
    source.loc[0, "breadth_ma120"] = 0.123

    with pytest.raises(ValueError, match="does not match"):
        build_ma_period_breadth_signals(source, start_date="2020-01-01")


def test_rejects_sample_shorter_than_change_lookback():
    source = _daily_rows().iloc[:5]

    with pytest.raises(ValueError, match="five-day change lookback"):
        build_ma_period_breadth_signals(source, start_date="2020-01-01")
