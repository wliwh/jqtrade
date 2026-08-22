import pandas as pd
import pytest

from research.index_turning_points.modeling.targets import (
    add_future_entry_targets,
    build_lobe_intensity_targets,
)


def _target_inputs():
    dates = pd.bdate_range("2020-01-01", periods=12)
    daily = pd.DataFrame(
        {
            "high": [80, 90, 98, 100, 90, 91, 98.01, 99, 95, 94, 93, 96],
            "low": [78, 88, 96, 98, 88, 89, 96, 97, 92, 90, 89, 94],
        },
        index=dates,
    )
    regions = pd.DataFrame(
        [
            {
                "region_id": "all_a_medium_top_1",
                "index_id": "all_a",
                "event_type": "top",
                "price_band_pct": 0.02,
            },
            {
                "region_id": "all_a_medium_bottom_1",
                "index_id": "all_a",
                "event_type": "bottom",
                "price_band_pct": 0.02,
            },
        ]
    )
    lobes = pd.DataFrame(
        [
            {
                "region_id": "all_a_medium_top_1",
                "lobe_id": "top_lobe_1",
                "index_id": "all_a",
                "event_type": "top",
                "lobe_start": dates[2],
                "lobe_end": dates[3],
                "representative_price": 100.0,
            },
            {
                "region_id": "all_a_medium_top_1",
                "lobe_id": "top_lobe_2",
                "index_id": "all_a",
                "event_type": "top",
                "lobe_start": dates[6],
                "lobe_end": dates[7],
                "representative_price": 99.0,
            },
            {
                "region_id": "all_a_medium_bottom_1",
                "lobe_id": "bottom_lobe_1",
                "index_id": "all_a",
                "event_type": "bottom",
                "lobe_start": dates[9],
                "lobe_end": dates[10],
                "representative_price": 89.0,
            },
        ]
    )
    return dates, daily, regions, lobes


def test_scores_each_strict_lobe_extreme_at_100_without_filling_bridges():
    dates, daily, regions, lobes = _target_inputs()

    result = build_lobe_intensity_targets(daily, regions, lobes).set_index("date")

    assert result.loc[dates[3], "truth_top_intensity"] == 100.0
    assert result.loc[dates[7], "truth_top_intensity"] == 100.0
    assert result.loc[dates[10], "truth_bottom_intensity"] == 100.0
    assert result.loc[dates[6], "truth_top_intensity"] == pytest.approx(50.0)
    assert bool(result.loc[dates[2], "truth_top_in_strict_lobe"])
    assert result.loc[dates[4:6], "truth_top_intensity"].eq(0.0).all()
    assert not result.loc[dates[4:6], "truth_top_in_strict_lobe"].any()
    assert result.loc[dates[0], ["truth_top_intensity", "truth_bottom_intensity"]].eq(0.0).all()


def test_future_entry_targets_are_nested_and_leave_incomplete_tail_unknown():
    _, daily, regions, lobes = _target_inputs()
    intensity = build_lobe_intensity_targets(daily, regions, lobes)

    result = add_future_entry_targets(intensity, horizons=(1, 2))

    complete = result["target_complete_2d"]
    assert (
        result.loc[complete, "target_top_within_1d"].astype(int)
        <= result.loc[complete, "target_top_within_2d"].astype(int)
    ).all()
    assert bool(result.loc[1, "target_top_within_2d"])
    assert bool(result.loc[2, "target_top_within_1d"])
    assert pd.isna(result.iloc[-1]["target_top_within_1d"])
    assert pd.isna(result.iloc[-2]["target_top_within_2d"])
