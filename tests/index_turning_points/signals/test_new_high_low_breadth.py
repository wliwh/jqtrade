import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.new_high_low_breadth import (
    build_new_high_low_breadth_signals,
)


def _daily_rows():
    dates = pd.bdate_range("2020-01-01", periods=18)
    net_values = [
        0.10,
        0.10,
        0.10,
        0.10,
        0.10,
        0.06,
        0.07,
        0.10,
        -0.15,
        -0.14,
        -0.13,
        -0.12,
        -0.11,
        -0.08,
        -0.07,
        -0.11,
        -0.04,
        0.00,
    ]
    records = []
    for date, net_value in zip(dates, net_values):
        record = {"date": date, "universe_size": 120}
        for window in (60, 120, 250):
            valid_count = 100
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


def test_builds_symmetric_top_and_bottom_reversal_series():
    source = _daily_rows()

    daily, episodes, metadata = build_new_high_low_breadth_signals(
        source, start_date="2020-01-01"
    )

    top = daily[daily["direction"].eq("top")].reset_index(drop=True)
    bottom = daily[daily["direction"].eq("bottom")].reset_index(drop=True)
    assert top.loc[top["triggered"], "date"].tolist() == [
        source["date"].iloc[5],
        source["date"].iloc[6],
    ]
    assert bottom.loc[bottom["triggered"], "date"].tolist() == [
        source["date"].iloc[13],
        source["date"].iloc[14],
    ]
    assert top["new_high_low_net_change_5d"].iloc[:5].isna().all()
    assert not top["triggered"].iloc[:5].any()
    assert top["raw_value"].equals(top["new_high_low_net_composite"])
    assert top["valid_count"].eq(100).all()
    assert len(episodes) == 2
    assert set(episodes["direction"]) == {"top", "bottom"}
    assert metadata["episodes_by_direction"] == {"bottom": 1, "top": 1}


def test_events_are_invariant_when_input_is_truncated():
    source = _daily_rows()
    full, _, _ = build_new_high_low_breadth_signals(
        source, start_date="2020-01-01"
    )
    cutoff = source["date"].iloc[14]
    truncated, _, _ = build_new_high_low_breadth_signals(
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


def test_rejects_net_count_that_does_not_match_high_minus_low():
    source = _daily_rows()
    source.loc[0, "new_high_low_net_count_120"] += 1

    with pytest.raises(ValueError, match="does not match"):
        build_new_high_low_breadth_signals(source, start_date="2020-01-01")


def test_rejects_ratio_that_does_not_match_counts():
    source = _daily_rows()
    source.loc[0, "new_high_ratio_250"] = 0.123

    with pytest.raises(ValueError, match="does not match"):
        build_new_high_low_breadth_signals(source, start_date="2020-01-01")
