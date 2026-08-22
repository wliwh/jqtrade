import numpy as np

from research.index_turning_points.modeling.episode_filter import (
    select_episode_filter_threshold,
)


def test_episode_filter_selects_precision_improving_threshold_with_recall_floor():
    probability = np.array([0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.30, 0.20])
    target = np.array([1, 1, 1, 0, 1, 0, 0, 0])

    selected = select_episode_filter_threshold(
        probability,
        target,
        calibration_year_count=2,
        annual_candidate_budget=6,
        min_selected_candidates=3,
        min_match_recall=0.75,
    )

    assert selected.status == "selected"
    assert selected.threshold == 0.85
    assert selected.selected_candidates == 3
    assert selected.selected_matches == 3
    assert selected.match_recall == 0.75


def test_episode_filter_passes_through_when_support_constraint_is_impossible():
    selected = select_episode_filter_threshold(
        np.array([0.8, 0.4, 0.2]),
        np.array([1, 0, 0]),
        calibration_year_count=1,
        min_selected_candidates=6,
    )

    assert selected.status == "passthrough_no_feasible_threshold"
    assert selected.threshold == 0.0
    assert selected.selected_candidates == 3
