import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.multi_period_ma_breadth import (
    build_multi_period_ma_breadth_signals,
)


def _daily_rows():
    dates = pd.bdate_range("2020-01-01", periods=16)
    composite = [
        0.80,
        0.81,
        0.82,
        0.83,
        0.84,
        0.74,
        0.75,
        0.60,
        0.25,
        0.20,
        0.24,
        0.26,
        0.28,
        0.31,
        0.27,
        0.30,
    ]
    records = []
    for date, value in zip(dates, composite):
        record = {"date": date, "universe_size": 100}
        counts = {20: 100, 60: 80, 120: 60}
        for window, valid_count in counts.items():
            above_count = round(value * valid_count)
            record[f"above_count_ma{window}"] = above_count
            record[f"valid_count_ma{window}"] = valid_count
            record[f"breadth_ma{window}"] = above_count / valid_count
        records.append(record)
    return pd.DataFrame(records)


def test_builds_symmetric_top_and_bottom_reversal_series():
    daily, episodes, metadata = build_multi_period_ma_breadth_signals(
        _daily_rows(), start_date="2020-01-01"
    )

    top = daily[daily["direction"].eq("top")].reset_index(drop=True)
    bottom = daily[daily["direction"].eq("bottom")].reset_index(drop=True)
    assert top.loc[top["triggered"], "date"].tolist() == [
        pd.Timestamp("2020-01-08"),
        pd.Timestamp("2020-01-09"),
    ]
    assert bottom.loc[bottom["triggered"], "date"].tolist() == [
        pd.Timestamp("2020-01-21"),
        pd.Timestamp("2020-01-22"),
    ]
    assert top["breadth_change_5d"].iloc[:5].isna().all()
    assert not top["triggered"].iloc[:5].any()
    assert top["raw_value"].equals(top["breadth_composite"])
    assert top["valid_count"].eq(60).all()
    assert len(episodes) == 2
    assert set(episodes["direction"]) == {"top", "bottom"}
    assert metadata["episodes_by_direction"] == {"bottom": 1, "top": 1}


def test_events_are_invariant_when_input_is_truncated():
    source = _daily_rows()
    full, _, _ = build_multi_period_ma_breadth_signals(
        source, start_date="2020-01-01"
    )
    cutoff = source["date"].iloc[12]
    truncated, _, _ = build_multi_period_ma_breadth_signals(
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
    source.loc[0, "breadth_ma60"] = 0.123

    with pytest.raises(ValueError, match="does not match"):
        build_multi_period_ma_breadth_signals(source, start_date="2020-01-01")


def test_rejects_noncausal_date_order():
    source = _daily_rows()
    source.loc[[0, 1], "date"] = source.loc[[1, 0], "date"].to_numpy()

    with pytest.raises(ValueError, match="strictly increasing"):
        build_multi_period_ma_breadth_signals(source, start_date="2020-01-01")
