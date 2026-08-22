import pandas as pd

from research.index_turning_points.modeling.episode_targets import (
    OPERATIONAL_LABEL_VERSION,
    build_operational_episode_labels,
)


def _regions_and_lobes(calendar):
    anchors = [calendar[10], calendar[30], calendar[50], calendar[70]]
    regions = pd.DataFrame(
        {
            "region_id": ["r1", "r2", "r3", "r4"],
            "index_id": ["all_a"] * 4,
            "event_type": ["top"] * 4,
            "eligible": [True] * 4,
            "anchor_date": anchors,
        }
    )
    lobes = pd.DataFrame(
        {
            "lobe_id": ["l1", "l2", "l3", "l4"],
            "region_id": ["r1", "r2", "r3", "r4"],
            "lobe_start": [calendar[10], calendar[30], calendar[50], calendar[77]],
            "lobe_end": [calendar[10], calendar[30], calendar[50], calendar[77]],
        }
    )
    return regions, lobes


def test_operational_window_is_symmetric_five_days_and_preserves_strict_core():
    calendar = pd.bdate_range("2020-01-01", periods=90)
    regions, lobes = _regions_and_lobes(calendar)
    candidates = pd.DataFrame(
        {
            "candidate_episode_id": ["minus5", "plus5", "minus6", "strict7"],
            "direction": ["top"] * 4,
            "onset_date": [calendar[5], calendar[35], calendar[44], calendar[77]],
        }
    )

    result = build_operational_episode_labels(
        candidates, regions, lobes, calendar
    ).set_index("candidate_episode_id")

    assert bool(result.loc["minus5", "target_operational_match"])
    assert bool(result.loc["plus5", "target_operational_match"])
    assert not bool(result.loc["minus6", "target_operational_match"])
    assert bool(result.loc["strict7", "target_operational_match"])
    assert bool(result.loc["strict7", "target_operational_strict_match"])
    assert result.loc["strict7", "operational_match_level"] == "core_lobe"
    assert result["operational_label_version"].eq(
        OPERATIONAL_LABEL_VERSION
    ).all()
    assert result["operational_window_trade_days"].eq(5).all()


def test_operational_matching_is_one_to_one_and_marks_duplicate_candidates():
    calendar = pd.bdate_range("2020-01-01", periods=30)
    regions = pd.DataFrame(
        {
            "region_id": ["r1"],
            "index_id": ["all_a"],
            "event_type": ["bottom"],
            "eligible": [True],
            "anchor_date": [calendar[15]],
        }
    )
    lobes = pd.DataFrame(
        {
            "lobe_id": ["l1"],
            "region_id": ["r1"],
            "lobe_start": [calendar[15]],
            "lobe_end": [calendar[15]],
        }
    )
    candidates = pd.DataFrame(
        {
            "candidate_episode_id": ["far", "near"],
            "direction": ["bottom", "bottom"],
            "onset_date": [calendar[11], calendar[14]],
        }
    )

    result = build_operational_episode_labels(
        candidates, regions, lobes, calendar
    ).set_index("candidate_episode_id")

    assert result.loc["near", "operational_match_status"] == "matched"
    assert bool(result.loc["near", "target_operational_match"])
    assert result.loc["far", "operational_match_status"] == "duplicate_candidate"
    assert not bool(result.loc["far", "target_operational_match"])
