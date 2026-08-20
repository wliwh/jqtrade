import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.four_industry_top1 import (
    TARGET_IDS,
    build_four_industry_top1_signal,
)


def _daily(periods=12):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    starts = {"bank": 1, "coal": 4, "nonferrous": 0, "steel": 2}
    frame = pd.DataFrame(
        {
            "date": dates,
            "breadth_ma20": 0.4,
            "ranked_industry_count_ma20": 28,
            "top1_tie_count_ma20": 1,
            "top1_industry_codes_ma20": "other",
            "top1_industry_names_ma20": "其他",
            "four_industry_top1_triggered": False,
            "four_industry_top1_ids": "",
        }
    )
    for number, target_id in enumerate(TARGET_IDS, start=1):
        comparable = np.arange(periods) >= starts[target_id]
        frame[f"target_{target_id}_mapping_count"] = comparable.astype(int)
        frame[f"target_{target_id}_industry_code"] = np.where(
            comparable, f"code_{target_id}", ""
        )
        frame[f"target_{target_id}_industry_name"] = np.where(
            comparable, target_id, ""
        )
        frame[f"target_{target_id}_valid_count_ma20"] = np.where(
            comparable, 10 + number, 0
        )
        frame[f"target_{target_id}_breadth_ma20"] = np.where(
            comparable, 0.3 + number / 10, np.nan
        )
        frame[f"target_{target_id}_rank_ma20"] = np.where(
            comparable, number + 1, np.nan
        )
        frame[f"target_{target_id}_is_top1_ma20"] = False
    _set_top1(frame, 3, ["bank"])
    _set_top1(frame, 4, ["coal"])
    _set_top1(frame, 5, ["coal"])
    _set_top1(frame, 8, ["bank", "coal"])
    return frame


def _set_top1(frame, position, target_ids):
    for target_id in target_ids:
        frame.loc[position, f"target_{target_id}_rank_ma20"] = 1.0
        frame.loc[position, f"target_{target_id}_is_top1_ma20"] = True
    frame.loc[position, "four_industry_top1_triggered"] = True
    frame.loc[position, "four_industry_top1_ids"] = "|".join(sorted(target_ids))
    frame.loc[position, "top1_tie_count_ma20"] = len(target_ids)


def test_joint_comparison_starts_at_latest_industry_start_without_carry_in():
    source = _daily()

    daily, episodes, metadata = build_four_industry_top1_signal(source)

    assert metadata["target_start_dates"] == {
        "bank": "2020-01-02",
        "coal": "2020-01-07",
        "nonferrous": "2020-01-01",
        "steel": "2020-01-03",
    }
    assert metadata["comparison_start_date"] == "2020-01-07"
    assert daily["date"].iloc[0] == source.loc[4, "date"]
    assert daily["event_onset"].tolist()[:3] == [True, False, False]
    assert daily["event_capped_confirmation"].tolist()[:3] == [False, True, False]
    assert len(episodes) == 2
    assert daily["valid_count"].eq(4).all()


def test_ties_are_included_and_raw_value_is_max_target_breadth():
    daily, _, _ = build_four_industry_top1_signal(_daily())

    tied = daily[daily["four_industry_top1_ids"].eq("bank|coal")].iloc[0]
    assert bool(tied["triggered"])
    assert tied["top1_tie_count_ma20"] == 2
    assert tied["raw_value"] == pytest.approx(0.7)


def test_rejects_target_gap_after_joint_start():
    source = _daily()
    source.loc[7, "target_coal_mapping_count"] = 0
    source.loc[7, "target_coal_rank_ma20"] = np.nan

    with pytest.raises(ValueError, match="not continuous"):
        build_four_industry_top1_signal(source)


def test_signal_events_are_invariant_after_joint_start_when_input_is_truncated():
    source = _daily()
    full, _, _ = build_four_industry_top1_signal(source)
    truncated, _, _ = build_four_industry_top1_signal(source.iloc[:10])
    event_columns = [
        "date",
        "triggered",
        "episode_id",
        "episode_stage",
        "event_onset",
        "event_continuation",
        "event_exit",
        "event_capped_confirmation",
    ]

    pd.testing.assert_frame_equal(
        full.loc[full["date"].le(truncated["date"].max()), event_columns].reset_index(
            drop=True
        ),
        truncated[event_columns].reset_index(drop=True),
    )
