import numpy as np
import pandas as pd

from research.index_turning_points.archive.four_industry_width_v1.code.analyze_forward_returns import (
    analyze_frames,
    benjamini_hochberg,
    build_phase_frame,
    compute_future_returns,
    evaluate_duration_buckets,
    newey_west_ols,
)


def _signal_and_outcomes(periods=90):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    signal = pd.DataFrame(
        {
            "date": dates,
            "breadth_ma20": np.linspace(0.2, 0.8, periods),
            "target_bank": False,
            "target_coal": False,
            "target_nonferrous": False,
            "target_steel": False,
        }
    )
    signal.loc[10:14, "target_bank"] = True
    signal.loc[30:31, "target_coal"] = True
    outcomes = pd.DataFrame(
        {
            "index_id": "test",
            "index_name": "测试指数",
            "date": dates,
            "close": 100.0 + np.arange(periods, dtype=float),
        }
    )
    return signal, outcomes


def test_duration_analysis_builds_one_row_per_bucket_and_horizon():
    signal, outcomes = _signal_and_outcomes()
    result = evaluate_duration_buckets(signal, outcomes, horizons=(1,))

    assert len(result) == 5
    assert set(result["bucket_id"]) == {
        "day_1",
        "day_2_3",
        "day_4_5",
        "day_6_10",
        "day_11_plus",
    }


def test_report_renders_literal_fdr_percent_text():
    signal, outcomes = _signal_and_outcomes()
    result = analyze_frames(signal, outcomes, horizons=(1,))

    assert "5% FDR" in result["report"]
    assert not result["primary"]["inference_eligible"].any()
    assert result["primary"]["raw_q_value"].isna().all()
    assert result["primary"]["global_raw_q_value"].isna().all()


def test_phase_frame_collapses_contiguous_signal_runs():
    phase = build_phase_frame(pd.Series([False, True, True, False, True, False]))

    assert phase["onset"].tolist() == [False, True, False, False, True, False]
    assert phase["continuation"].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert phase["exit"].tolist() == [False, False, False, True, False, True]
    assert phase["episode_id"].fillna(0).tolist() == [0, 1, 1, 0, 2, 0]
    assert phase["episode_day"].fillna(0).tolist() == [0, 1, 2, 0, 1, 0]


def test_future_returns_use_close_to_exact_future_close():
    result = compute_future_returns(pd.Series([100.0, 110.0, 99.0]), (1, 2))

    assert np.isclose(result.loc[0, "future_return_1d"], 0.1)
    assert np.isclose(result.loc[0, "future_return_2d"], -0.01)
    assert pd.isna(result.loc[1, "future_return_2d"])


def test_newey_west_active_coefficient_equals_group_mean_difference():
    outcome = pd.Series([0.01, 0.02, -0.01, 0.04, 0.00, 0.03])
    active = pd.Series([False, True, False, True, False, True])
    design = pd.DataFrame(
        {"intercept": 1.0, "active": active.astype(float)}
    )

    result = newey_west_ols(outcome, design, max_lag=1)
    expected = outcome[active].mean() - outcome[~active].mean()
    assert np.isclose(result.loc["active", "coefficient"], expected)
    assert result.loc["active", "nobs"] == 6


def test_benjamini_hochberg_preserves_order_and_missing_values():
    pvalues = pd.Series([0.01, np.nan, 0.04, 0.03])
    adjusted = benjamini_hochberg(pvalues)

    assert np.isclose(adjusted.iloc[0], 0.03)
    assert pd.isna(adjusted.iloc[1])
    assert np.isclose(adjusted.iloc[2], 0.04)
    assert np.isclose(adjusted.iloc[3], 0.04)
