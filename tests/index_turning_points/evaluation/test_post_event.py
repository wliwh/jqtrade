import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.evaluation.post_event import (
    build_forward_event_outcomes,
    summarize_forward_event_outcomes,
)


def _ohlc(periods=80):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    close = 100.0 + np.arange(periods)
    return pd.DataFrame(
        {
            "index_id": "test_index",
            "index_name": "测试指数",
            "date": dates,
            "high": close + 3.0,
            "low": close - 2.0,
            "close": close,
        }
    )


def _signal(dates, event_positions):
    frame = pd.DataFrame(
        {
            "date": dates,
            "signal_id": "test_signal",
            "direction": "bottom",
            "version": "test_v1",
            "episode_id": pd.NA,
            "event_onset": False,
            "event_capped_confirmation": False,
        }
    )
    for number, position in enumerate(event_positions, start=1):
        frame.loc[position, "episode_id"] = f"episode_{number}"
        frame.loc[position, "event_onset"] = True
        frame.loc[position, "event_capped_confirmation"] = True
    return frame


def test_future_outcomes_use_high_and_low_not_close_path():
    prices = _ohlc(10)
    prices.loc[2, "high"] = 150.0
    prices.loc[3, "low"] = 50.0
    signal = _signal(prices["date"], [1])

    outcomes = build_forward_event_outcomes(
        signal,
        prices,
        event_kinds=("onset",),
        horizons=(2,),
    )

    row = outcomes.iloc[0]
    assert bool(row["complete_window"])
    assert row["window_end_date"] == prices.loc[3, "date"]
    assert row["terminal_return"] == pytest.approx(103.0 / 101.0 - 1.0)
    assert row["max_up"] == pytest.approx(150.0 / 101.0 - 1.0)
    assert row["max_down"] == pytest.approx(50.0 / 101.0 - 1.0)


def test_missing_index_date_and_tail_window_remain_explicit():
    prices = _ohlc(8)
    signal = _signal(prices["date"], [3, 7])
    signal.loc[3, "date"] = pd.Timestamp("2020-01-04")

    outcomes = build_forward_event_outcomes(
        signal,
        prices,
        event_kinds=("onset",),
        horizons=(2,),
    ).set_index("episode_id")

    missing = outcomes.loc["episode_1"]
    assert not bool(missing["event_date_available"])
    assert not bool(missing["complete_window"])
    assert missing["available_future_bars"] == 0
    tail = outcomes.loc["episode_2"]
    assert bool(tail["event_date_available"])
    assert not bool(tail["complete_window"])
    assert tail["available_future_bars"] == 0


def test_summary_reports_description_hac_and_two_fdr_families():
    prices = _ohlc(140)
    event_positions = list(range(5, 105, 4))
    signal = _signal(prices["date"], event_positions)
    outcomes = build_forward_event_outcomes(
        signal,
        prices,
        event_kinds=("onset",),
        horizons=(5, 10),
    )

    metrics = summarize_forward_event_outcomes(
        outcomes,
        signal,
        prices,
        min_event_count=20,
        min_baseline_count=30,
    )

    assert len(metrics) == 6
    assert metrics["inference_eligible"].all()
    assert metrics["event_count"].eq(len(event_positions)).all()
    assert metrics["baseline_count"].gt(30).all()
    assert metrics["hac_lag"].tolist() == [5, 5, 5, 10, 10, 10]
    assert metrics["hac_p_value"].between(0.0, 1.0).all()
    assert metrics["local_fdr_q_value"].between(0.0, 1.0).all()
    assert metrics["global_fdr_q_value"].between(0.0, 1.0).all()
    terminal = metrics[
        metrics["outcome_name"].eq("terminal_return")
        & metrics["horizon"].eq(5)
    ].iloc[0]
    assert terminal["mean_difference"] == pytest.approx(
        terminal["event_mean"] - terminal["baseline_mean"]
    )


def test_sample_minimum_suppresses_inference_but_keeps_description():
    prices = _ohlc(40)
    signal = _signal(prices["date"], [5, 10])
    outcomes = build_forward_event_outcomes(
        signal,
        prices,
        event_kinds=("onset",),
        horizons=(5,),
    )

    metrics = summarize_forward_event_outcomes(outcomes, signal, prices)

    assert not metrics["inference_eligible"].any()
    assert metrics["event_count"].eq(2).all()
    assert metrics["event_mean"].notna().all()
    assert metrics["hac_p_value"].isna().all()
    assert metrics["local_fdr_q_value"].isna().all()


def test_rejects_duplicate_signal_series_dates():
    prices = _ohlc(20)
    signal = _signal(prices["date"], [5])
    signal = pd.concat([signal, signal.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate series dates"):
        build_forward_event_outcomes(signal, prices)
