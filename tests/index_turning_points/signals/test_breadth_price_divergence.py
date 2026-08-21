import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.breadth_price_divergence import (
    build_breadth_price_divergence_signal,
)


def _inputs():
    dates = pd.bdate_range("2020-01-01", periods=70)
    breadth = np.ones(len(dates))
    breadth[59] = 0.75
    breadth[60] = 0.74
    breadth[61:] = 0.95
    market = pd.DataFrame(
        {
            "date": dates,
            "universe_size": 100,
            "valid_count_ma20": 100,
            "above_count_ma20": np.rint(breadth * 100).astype(int),
            "breadth_ma20": breadth,
        }
    )
    prices = pd.DataFrame(
        {
            "date": dates,
            "close": np.arange(100.0, 100.0 + len(dates)),
        }
    )
    return market, prices


def test_triggers_when_price_is_near_high_and_breadth_is_far_below_high():
    market, prices = _inputs()

    daily, episodes, metadata = build_breadth_price_divergence_signal(
        market, prices, start_date="2020-01-01"
    )

    assert daily.loc[daily["triggered"], "date"].tolist() == [
        market["date"].iloc[59],
        market["date"].iloc[60],
    ]
    assert daily["comparison_available"].iloc[:59].eq(False).all()
    assert daily.loc[59, "breadth_price_divergence"] == pytest.approx(0.25)
    assert len(episodes) == 1
    assert episodes.iloc[0]["active_days"] == 2
    assert metadata["first_available_date"] == market["date"].iloc[59].strftime(
        "%Y-%m-%d"
    )


def test_missing_current_price_is_inactive_without_interpolation():
    market, prices = _inputs()
    missing_date = market["date"].iloc[59]
    prices = prices[prices["date"].ne(missing_date)]

    daily, _, metadata = build_breadth_price_divergence_signal(
        market, prices, start_date="2020-01-01"
    )

    row = daily[daily["date"].eq(missing_date)].iloc[0]
    assert pd.isna(row["index_close"])
    assert not row["comparison_available"]
    assert not row["triggered"]
    assert metadata["missing_index_price_dates"] == [
        missing_date.strftime("%Y-%m-%d")
    ]


def test_events_are_invariant_when_both_inputs_are_truncated():
    market, prices = _inputs()
    full, _, _ = build_breadth_price_divergence_signal(
        market, prices, start_date="2020-01-01"
    )
    cutoff = market["date"].iloc[65]
    truncated, _, _ = build_breadth_price_divergence_signal(
        market[market["date"].le(cutoff)],
        prices[prices["date"].le(cutoff)],
        start_date="2020-01-01",
    )
    event_columns = [
        "date",
        "raw_value",
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


def test_rejects_more_than_two_missing_index_dates():
    market, prices = _inputs()
    prices = prices[~prices["date"].isin(market["date"].iloc[10:13])]

    with pytest.raises(ValueError, match="too many missing dates"):
        build_breadth_price_divergence_signal(
            market, prices, start_date="2020-01-01"
        )
