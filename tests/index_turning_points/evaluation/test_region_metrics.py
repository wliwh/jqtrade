import pandas as pd
import pytest

from research.index_turning_points.evaluation.region_matching import (
    match_signal_regions,
)
from research.index_turning_points.evaluation.region_metrics import (
    add_diagnostic_region_slices,
    summarize_region_slices,
)
from research.index_turning_points.ground_truth.regions import DEFAULT_REGION_PROTOCOL


def _inputs(index_id="test_index", index_name="测试指数", id_prefix=""):
    dates = pd.bdate_range("2020-01-01", periods=70)
    calendar = pd.DataFrame(
        {"index_id": index_id, "index_name": index_name, "date": dates}
    )
    signal = pd.DataFrame(
        {
            "date": dates,
            "signal_id": "test_signal",
            "direction": "top",
            "version": "test_v1",
            "episode_id": pd.NA,
            "event_onset": False,
            "event_capped_confirmation": False,
            "event_exit": False,
        }
    )
    for number, position in enumerate([10, 26, 69], start=1):
        signal.loc[position, "episode_id"] = f"episode_{number}"
        signal.loc[position, "event_onset"] = True

    def region(region_id, anchor, start, end, lobes):
        return {
            "region_id": f"{id_prefix}{region_id}",
            "index_id": index_id,
            "index_name": index_name,
            "event_type": "top",
            "status": "confirmed",
            "eligible": True,
            "region_start": dates[start],
            "region_end": dates[end],
            "anchor_date": dates[anchor],
            "lobe_count": lobes,
            "label_version": DEFAULT_REGION_PROTOCOL.label_version,
        }

    regions = pd.DataFrame(
        [
            region("r1", 10, 10, 10, 1),
            region("r2", 30, 29, 31, 2),
            region("r3", 45, 45, 45, 1),
        ]
    )
    lobes = pd.DataFrame(
        [
            {
                "region_id": f"{id_prefix}r1",
                "lobe_id": f"{id_prefix}r1_lobe_1",
                "lobe_start": dates[10],
                "lobe_end": dates[10],
            },
            {
                "region_id": f"{id_prefix}r2",
                "lobe_id": f"{id_prefix}r2_lobe_1",
                "lobe_start": dates[29],
                "lobe_end": dates[29],
            },
            {
                "region_id": f"{id_prefix}r2",
                "lobe_id": f"{id_prefix}r2_lobe_2",
                "lobe_start": dates[31],
                "lobe_end": dates[31],
            },
            {
                "region_id": f"{id_prefix}r3",
                "lobe_id": f"{id_prefix}r3_lobe_1",
                "lobe_start": dates[45],
                "lobe_end": dates[45],
            },
        ]
    )
    return signal, regions, lobes, calendar


def test_complete_slices_keep_false_alarm_assignment_diagnostic_only():
    signal, regions, lobes, calendar = _inputs()
    matches = match_signal_regions(signal, regions, lobes, calendar)

    enriched = add_diagnostic_region_slices(matches, regions)
    false_alarm = enriched[enriched["match_status"].eq("false_alarm")].iloc[0]
    assert false_alarm["diagnostic_region_id"] == "r3"
    assert false_alarm["diagnostic_timing"] == "confirmation"
    assert false_alarm["diagnostic_region_form"] == "single_lobe"
    assert false_alarm["diagnostic_assignment"] == "nearest_region_for_slice_only"
    assert not bool(false_alarm["primary_match"])

    metrics = summarize_region_slices(enriched)
    strict_all = metrics[
        metrics["aggregation"].eq("index")
        & metrics["match_scope"].eq("strict")
        & metrics["timing_slice"].eq("all")
        & metrics["region_form_slice"].eq("all")
    ].iloc[0]
    assert strict_all["region_count"] == 3
    assert strict_all["matched_region_count"] == 1
    assert strict_all["region_recall"] == pytest.approx(1 / 3)
    assert strict_all["episode_count"] == 3
    assert strict_all["matched_episode_count"] == 1
    assert strict_all["episode_precision"] == pytest.approx(1 / 3)
    assert strict_all["false_alarm_count"] == 2
    assert strict_all["isolated_false_alarm_count"] == 1


def test_prediction_multi_lobe_and_confirmation_slices_have_clear_denominators():
    signal, regions, lobes, calendar = _inputs()
    matches = add_diagnostic_region_slices(
        match_signal_regions(signal, regions, lobes, calendar), regions
    )
    metrics = summarize_region_slices(matches)

    prediction_multi = metrics[
        metrics["aggregation"].eq("index")
        & metrics["match_scope"].eq("window")
        & metrics["timing_slice"].eq("prediction")
        & metrics["region_form_slice"].eq("multi_lobe")
    ].iloc[0]
    assert prediction_multi["region_count"] == 1
    assert prediction_multi["matched_region_count"] == 1
    assert prediction_multi["episode_count"] == 1
    assert prediction_multi["episode_precision"] == 1.0

    confirmation_single = metrics[
        metrics["aggregation"].eq("index")
        & metrics["match_scope"].eq("window")
        & metrics["timing_slice"].eq("confirmation")
        & metrics["region_form_slice"].eq("single_lobe")
    ].iloc[0]
    assert confirmation_single["region_count"] == 2
    assert confirmation_single["matched_region_count"] == 0
    assert confirmation_single["episode_count"] == 1
    assert confirmation_single["isolated_false_alarm_count"] == 1


def test_all_indices_uses_explicit_index_event_pair_units():
    signal, regions_1, lobes_1, calendar_1 = _inputs()
    _, regions_2, lobes_2, calendar_2 = _inputs(
        index_id="other_index", index_name="另一指数", id_prefix="other_"
    )
    regions = pd.concat([regions_1, regions_2], ignore_index=True)
    lobes = pd.concat([lobes_1, lobes_2], ignore_index=True)
    calendars = pd.concat([calendar_1, calendar_2], ignore_index=True)
    matches = match_signal_regions(signal, regions, lobes, calendars)
    metrics = summarize_region_slices(
        add_diagnostic_region_slices(matches, regions)
    )

    aggregate = metrics[
        metrics["aggregation"].eq("all_indices")
        & metrics["match_scope"].eq("window")
        & metrics["timing_slice"].eq("all")
        & metrics["region_form_slice"].eq("all")
    ].iloc[0]
    assert aggregate["observation_unit"] == "index_region_or_index_episode_pair"
    assert aggregate["region_count"] == 6
    assert aggregate["episode_count"] == 6
    assert aggregate["matched_region_count"] == 4
