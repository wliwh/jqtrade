import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.modeling.dataset import today_feature_columns
from research.index_turning_points.modeling.today_training import (
    TODAY_TRAINING_VERSION,
    run_today_walk_forward_training,
)


def _synthetic_today_training_daily() -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    position = np.arange(len(dates))
    frame = pd.DataFrame({"date": dates})
    for number, column in enumerate(today_feature_columns()):
        frame[column] = (
            np.sin(position / (7.0 + number % 5)) + number / 1000.0
        )
    for direction, offset in (("top", 0), ("bottom", 17)):
        membership = ((position + offset) % 53) < 4
        frame[f"truth_{direction}_intensity"] = np.where(
            membership, 100.0, 0.0
        )
        frame[f"truth_{direction}_in_strict_lobe"] = membership
    return frame


def test_today_walk_forward_outputs_direct_calibrated_probabilities():
    result = run_today_walk_forward_training(
        _synthetic_today_training_daily(),
        model_ids=("elastic_net",),
        first_validation_year=2018,
        boundary_gap=5,
        annual_episode_budget=3,
        version="test_today_walk_forward",
    )

    assert set(result.signal_daily["test_year"]) == {2019, 2020}
    assert len(result.probability_metrics) == 2 * 1 * 2
    assert len(result.fit_audit) == 2 * 1 * 2
    assert result.thresholds["validation_episode_count"].le(3).all()
    assert result.folds["boundary_gap_trade_days"].eq(5).all()
    probability = result.signal_daily["pred_probability_today"]
    assert probability.between(0.0, 1.0).all()
    np.testing.assert_allclose(
        result.signal_daily["pred_score"], 100.0 * probability
    )
    np.testing.assert_allclose(
        result.signal_daily["raw_value"], result.signal_daily["pred_score"]
    )
    assert "actual_in_strict_lobe_today" in result.signal_daily
    assert not any(
        column.startswith("pred_probability_5d")
        for column in result.signal_daily
    )


def test_frozen_today_version_rejects_a_model_subset():
    with pytest.raises(ValueError, match="model_ids are frozen"):
        run_today_walk_forward_training(
            _synthetic_today_training_daily(),
            model_ids=("elastic_net",),
            version=TODAY_TRAINING_VERSION,
        )


def test_auxiliary_intensity_does_not_change_membership_probabilities():
    source = _synthetic_today_training_daily()
    changed = source.copy()
    changed["truth_top_intensity"] = np.where(
        changed["truth_top_in_strict_lobe"], 37.0, 0.0
    )
    changed["truth_bottom_intensity"] = np.where(
        changed["truth_bottom_in_strict_lobe"], 63.0, 0.0
    )

    original = run_today_walk_forward_training(
        source,
        model_ids=("elastic_net",),
        first_validation_year=2018,
        boundary_gap=5,
        version="test_today_walk_forward",
    )
    modified = run_today_walk_forward_training(
        changed,
        model_ids=("elastic_net",),
        first_validation_year=2018,
        boundary_gap=5,
        version="test_today_walk_forward",
    )

    np.testing.assert_allclose(
        original.signal_daily["pred_probability_today"],
        modified.signal_daily["pred_probability_today"],
    )
