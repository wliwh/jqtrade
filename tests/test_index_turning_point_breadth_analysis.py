import pandas as pd
import pytest

from research.index_turning_points.analyze_breadth import (
    build_episodes,
    compute_event_metrics,
    compute_forward_comparisons,
    prepare_valid_signal,
)


def _daily_signal():
    dates = pd.date_range("2020-01-01", periods=9, freq="D")
    frame = pd.DataFrame(
        {
            "date": dates,
            "four_industry_present_count": [3, 4, 4, 4, 4, 4, 4, 4, 4],
            "four_industry_top1_triggered": [
                False, True, True, False, True, False, False, False, False
            ],
            "four_industry_top1_ids": [
                "", "bank", "bank", "", "coal", "", "", "", ""
            ],
            "breadth_ma20": [0.4, 0.4, 0.6, 0.6, 0.4, 0.4, 0.6, 0.6, 0.4],
            "top1_tie_count_ma20": [1, 1, 2, 1, 1, 1, 1, 1, 1],
        }
    )
    for target_id in ("bank", "coal", "nonferrous", "steel"):
        frame["target_%s_is_top1_ma20" % target_id] = False
    frame.loc[[1, 2], "target_bank_is_top1_ma20"] = True
    frame.loc[4, "target_coal_is_top1_ma20"] = True
    return frame


def test_signal_starts_at_continuous_full_coverage_and_collapses_episodes():
    signal = prepare_valid_signal(_daily_signal())
    assert signal["date"].iloc[0] == pd.Timestamp("2020-01-02")
    assert signal["triggered"].tolist() == [
        True, True, False, True, False, False, False, False
    ]
    assert signal["onset"].tolist() == [
        True, False, False, True, False, False, False, False
    ]
    assert signal["continuation"].tolist() == [
        False, True, False, False, False, False, False, False
    ]
    assert signal["exit"].tolist() == [
        False, False, True, False, True, False, False, False
    ]
    assert signal["phase"].tolist() == [
        "onset",
        "continuation",
        "exit",
        "onset",
        "exit",
        "inactive",
        "inactive",
        "inactive",
    ]
    assert signal["episode_id"].fillna(0).tolist() == [1, 1, 0, 2, 0, 0, 0, 0]
    assert signal["episode_day"].fillna(0).tolist() == [1, 2, 0, 1, 0, 0, 0, 0]
    assert signal["breadth_le_50_triggered"].tolist() == [
        True, False, False, True, False, False, False, False
    ]
    assert signal["breadth_le_50_onset"].tolist() == [
        True, False, False, True, False, False, False, False
    ]
    assert signal["breadth_le_50_exit"].tolist() == [
        False, True, False, False, True, False, False, False
    ]
    assert signal["breadth_gt_50_triggered"].tolist() == [
        False, True, False, False, False, False, False, False
    ]
    assert signal["breadth_gt_50_onset"].tolist() == [
        False, True, False, False, False, False, False, False
    ]
    assert signal["breadth_gt_50_exit"].tolist() == [
        False, False, True, False, False, False, False, False
    ]
    assert (
        signal["breadth_le_50_onset"]
        == (
            signal["breadth_le_50_onset_industry_only"]
            | signal["breadth_le_50_onset_breadth_only"]
            | signal["breadth_le_50_onset_both"]
        )
    ).all()
    assert (
        signal["breadth_gt_50_exit"]
        == (
            signal["breadth_gt_50_exit_industry_only"]
            | signal["breadth_gt_50_exit_breadth_only"]
            | signal["breadth_gt_50_exit_both"]
        )
    ).all()
    assert signal["breadth_le_50_onset_both"].sum() == 2
    assert signal["breadth_gt_50_onset_breadth_only"].sum() == 1
    assert signal["breadth_gt_50_exit_industry_only"].sum() == 1

    episodes = build_episodes(signal)
    assert episodes["episode_id"].tolist() == [1, 2]
    assert episodes["trading_days"].tolist() == [2, 1]
    assert episodes["target_ids_seen"].tolist() == ["bank", "coal"]
    assert episodes["exit_date"].tolist() == [
        pd.Timestamp("2020-01-04"),
        pd.Timestamp("2020-01-06"),
    ]


def test_signal_rejects_incomplete_coverage_after_analysis_start():
    frame = _daily_signal()
    frame.loc[4, "four_industry_present_count"] = 3
    with pytest.raises(ValueError, match="not continuous"):
        prepare_valid_signal(frame)


def test_event_metrics_use_trading_day_lookback_and_forward_windows():
    signal = prepare_valid_signal(_daily_signal())
    calendar = signal["date"]
    outcomes = pd.DataFrame(
        {
            "index_id": "test",
            "index_name": "测试指数",
            "date": calendar,
        }
    )
    labels = pd.DataFrame(
        [
            {
                "index_id": "test",
                "index_name": "测试指数",
                "threshold_level": "medium",
                "event_type": "top",
                "status": "confirmed",
                "eligible": True,
                "anchor_date": calendar.iloc[6],
                "confirmation_date": calendar.iloc[7],
                "threshold": 0.10,
            }
        ]
    )

    metrics, matches = compute_event_metrics(signal, labels, outcomes)
    selected = metrics[
        metrics["signal_id"].eq("four_industry_top1")
        & metrics["threshold_level"].eq("medium")
        & metrics["event_type"].eq("top")
        & metrics["lead_window_days"].eq(5)
    ].iloc[0]
    assert selected["event_count"] == 1
    assert selected["matched_event_count"] == 1
    assert selected["event_recall"] == 1.0
    assert selected["conditional_precision_lift"] == pytest.approx(0.75)
    assert selected["evaluable_trigger_days"] == 2
    assert selected["hit_signal_days"] == 1
    assert selected["signal_precision"] == pytest.approx(1 / 2)
    assert selected["unconditional_window_event_rate"] == pytest.approx(2 / 3)

    selected_match = matches[
        matches["signal_id"].eq("four_industry_top1")
        & matches["threshold_level"].eq("medium")
        & matches["event_type"].eq("top")
        & matches["lead_window_days"].eq(5)
    ].iloc[0]
    assert bool(selected_match["matched"])
    assert selected_match["nearest_lead_days"] == 3

    onset = metrics[
        metrics["signal_id"].eq("four_industry_top1_onset")
        & metrics["threshold_level"].eq("medium")
        & metrics["event_type"].eq("top")
        & metrics["lead_window_days"].eq(5)
    ].iloc[0]
    assert onset["evaluable_trigger_days"] == 1
    assert onset["hit_signal_days"] == 0


def test_same_day_filtered_onset_counts_as_event_hit():
    signal = prepare_valid_signal(_daily_signal())
    calendar = signal["date"]
    outcomes = pd.DataFrame(
        {"index_id": "test", "index_name": "测试指数", "date": calendar}
    )
    labels = pd.DataFrame(
        [
            {
                "index_id": "test",
                "index_name": "测试指数",
                "threshold_level": "medium",
                "event_type": "top",
                "status": "confirmed",
                "eligible": True,
                "anchor_date": calendar.iloc[3],
                "confirmation_date": calendar.iloc[4],
                "threshold": 0.10,
            }
        ]
    )

    metrics, matches = compute_event_metrics(signal, labels, outcomes)
    signal_id = "four_industry_top1_breadth_le_50_onset"
    selected = metrics[
        metrics["signal_id"].eq(signal_id)
        & metrics["threshold_level"].eq("medium")
        & metrics["event_type"].eq("top")
        & metrics["lead_window_days"].eq(0)
    ].iloc[0]
    selected_match = matches[
        matches["signal_id"].eq(signal_id)
        & matches["threshold_level"].eq("medium")
        & matches["event_type"].eq("top")
        & matches["lead_window_days"].eq(0)
    ].iloc[0]

    assert selected["matched_event_count"] == 1
    assert selected["event_recall"] == 1.0
    assert selected["conditional_precision_lift"] == 2.0
    assert bool(selected_match["matched"])
    assert selected_match["nearest_lead_days"] == 0


def test_forward_phase_comparisons_use_explicit_inactive_controls():
    signal = prepare_valid_signal(_daily_signal())
    outcomes = pd.DataFrame(
        {
            "index_id": "test",
            "index_name": "测试指数",
            "date": signal["date"],
        }
    )
    for horizon in (5, 10, 20, 60):
        outcomes["future_max_down_%dd" % horizon] = -0.01
        outcomes["future_max_up_%dd" % horizon] = 0.01
        outcomes["future_return_%dd" % horizon] = (
            pd.Series(range(len(signal)), dtype=float) / 100.0
        )

    comparisons = compute_forward_comparisons(signal, outcomes)
    selected = comparisons[comparisons["horizon_days"].eq(20)].set_index(
        "signal_id"
    )

    assert selected.loc["four_industry_top1", "trigger_n"] == 3
    assert selected.loc["four_industry_top1", "control_n"] == 5
    assert selected.loc["four_industry_top1_onset", "trigger_n"] == 2
    assert selected.loc["four_industry_top1_onset", "control_n"] == 5
    assert selected.loc["four_industry_top1_continuation", "trigger_n"] == 1
    assert selected.loc["four_industry_top1_continuation", "control_n"] == 5
    assert selected.loc["four_industry_top1_exit", "trigger_n"] == 2
    assert selected.loc["four_industry_top1_exit", "control_n"] == 3
    assert selected.loc[
        "four_industry_top1_breadth_le_50_onset", "trigger_n"
    ] == 2
    assert selected.loc[
        "four_industry_top1_breadth_le_50_onset", "control_n"
    ] == 6
    assert selected.loc[
        "four_industry_top1_breadth_le_50_exit", "trigger_n"
    ] == 2
    assert selected.loc[
        "four_industry_top1_breadth_le_50_exit", "control_n"
    ] == 4
