import numpy as np
import pandas as pd

from research.index_turning_points.evaluation.reports import (
    FORWARD_REPORT_SECTION_ORDER,
    REGION_REPORT_SECTION_ORDER,
    collect_forward_findings,
    collect_region_findings,
    render_forward_report,
    render_region_report,
)


IDENTITY = {
    "signal_id": "test_signal",
    "direction": "top",
    "version": "test_v1",
    "event_kind": "onset",
}


def _region_metrics():
    records = []
    for scope, matched in [("strict", 1), ("loose", 1), ("window", 2)]:
        records.append(
            {
                **IDENTITY,
                "aggregation": "all_indices",
                "index_id": "__all__",
                "index_name": "全部指数（指数-事件对）",
                "match_scope": scope,
                "timing_slice": "all",
                "region_form_slice": "all",
                "region_count": 3,
                "matched_region_count": matched,
                "region_recall": matched / 3,
                "episode_count": 4,
                "matched_episode_count": matched,
                "episode_precision": matched / 4,
                "false_alarm_count": 4 - matched,
                "duplicate_alarm_count": 1,
                "median_lead_lag_days": -2,
            }
        )
    records.append(
        {
            **records[-1],
            "aggregation": "index",
            "index_id": "test_index",
            "index_name": "测试指数",
        }
    )
    return pd.DataFrame(records)


def _region_matches():
    return pd.DataFrame(
        [
            {
                **IDENTITY,
                "match_status": "matched",
                "prediction_window_complete": True,
                "confirmation_window_complete": True,
                "diagnostic_timing": "prediction",
            },
            {
                **IDENTITY,
                "match_status": "missed_region",
                "prediction_window_complete": False,
                "confirmation_window_complete": True,
                "diagnostic_timing": "",
            },
            {
                **IDENTITY,
                "match_status": "duplicate_alarm",
                "prediction_window_complete": np.nan,
                "confirmation_window_complete": np.nan,
                "diagnostic_timing": "prediction",
            },
        ]
    )


def _forward_metrics():
    base = {
        **IDENTITY,
        "index_name": "测试指数",
        "horizon": 5,
        "event_count": 2,
        "event_mean": 0.01,
        "baseline_count": 50,
        "baseline_mean": 0.0,
        "mean_difference": 0.01,
        "ci95_lower": -0.01,
        "ci95_upper": 0.03,
        "local_fdr_q_value": 0.1,
    }
    return pd.DataFrame(
        [
            {
                **base,
                "outcome_name": "terminal_return",
                "hac_p_value": 0.01,
                "global_fdr_q_value": 0.2,
                "inference_eligible": True,
            },
            {
                **base,
                "outcome_name": "max_up",
                "hac_p_value": np.nan,
                "global_fdr_q_value": np.nan,
                "inference_eligible": False,
            },
        ]
    )


def _forward_outcomes():
    return pd.DataFrame(
        [
            {
                **IDENTITY,
                "index_name": "测试指数",
                "horizon": 5,
                "episode_id": "episode_1",
                "event_date_available": False,
                "complete_window": False,
            },
            {
                **IDENTITY,
                "index_name": "测试指数",
                "horizon": 5,
                "episode_id": "episode_2",
                "event_date_available": True,
                "complete_window": True,
            },
        ]
    )


def test_region_report_has_fixed_sections_and_appends_group_findings_last():
    metrics = _region_metrics()
    matches = _region_matches()

    report = render_region_report(
        metrics,
        matches,
        evaluation_version="evaluation_v1",
        label_version="labels_v1",
    )

    positions = [report.index(f"## {section}") for section in REGION_REPORT_SECTION_ORDER]
    assert positions == sorted(positions)
    assert report.rfind("## 分组发现与注意事项") == positions[-1]
    findings = collect_region_findings(metrics, matches)
    assert len(findings) == 1
    assert "test_signal/top/test_v1/onset" in findings[0]
    assert "区域窗口不完整" in findings[0]
    assert "重复报警" in findings[0]
    assert "口径敏感" in findings[0]


def test_forward_report_has_fixed_sections_and_group_specific_notes_last():
    metrics = _forward_metrics()
    outcomes = _forward_outcomes()

    report = render_forward_report(
        metrics,
        outcomes,
        evaluation_version="evaluation_v1",
        min_event_count=20,
        min_baseline_count=30,
    )

    positions = [report.index(f"## {section}") for section in FORWARD_REPORT_SECTION_ORDER]
    assert positions == sorted(positions)
    assert report.rfind("## 分组发现与注意事项") == positions[-1]
    findings = collect_forward_findings(metrics, outcomes)
    assert len(findings) == 1
    assert "事件日缺失 1、窗口不完整 1" in findings[0]
    assert "1/2 项检验未达到样本门槛" in findings[0]
    assert "未通过全局 FDR" in findings[0]


def test_findings_flag_high_recall_low_precision_without_changing_body():
    metrics = _region_metrics()
    selected = metrics["match_scope"].eq("window") & metrics["aggregation"].eq(
        "all_indices"
    )
    metrics.loc[selected, "matched_region_count"] = 3
    metrics.loc[selected, "region_recall"] = 1.0
    metrics.loc[selected, "matched_episode_count"] = 0
    metrics.loc[selected, "episode_precision"] = 0.1

    findings = collect_region_findings(metrics, _region_matches())

    assert "高区域召回但低 episode 精确率" in findings[0]
    assert "100.0%/10.0%" in findings[0]


def test_forward_findings_record_no_nominal_tests_and_consistent_longest_direction():
    records = []
    for index_name in ["指数甲", "指数乙"]:
        records.append(
            {
                **IDENTITY,
                "index_name": index_name,
                "horizon": 20,
                "outcome_name": "terminal_return",
                "hac_p_value": 0.2,
                "global_fdr_q_value": 0.4,
                "inference_eligible": True,
                "mean_difference": -0.01,
            }
        )
    metrics = pd.DataFrame(records)
    outcomes = _forward_outcomes()

    findings = collect_forward_findings(metrics, outcomes)

    assert "2 项合格检验均未达到名义 p<0.05" in findings[0]
    assert "2/2 个指数均为负" in findings[0]
