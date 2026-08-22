import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.modeling.dataset import feature_columns
from research.index_turning_points.modeling.training import (
    TRAINING_VERSION_V2,
    TRAINING_VERSION_V3,
    run_walk_forward_training,
)


def _synthetic_training_daily() -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", "2020-12-31")
    position = np.arange(len(dates))
    frame = pd.DataFrame({"date": dates})
    for number, column in enumerate(feature_columns()):
        if column == "index_price_available":
            frame[column] = True
        else:
            frame[column] = (
                np.sin(position / (7.0 + number % 11)) + number / 1000.0
            )
    for direction, offset in (("top", 0), ("bottom", 19)):
        phase = (position + offset) % 53
        frame[f"truth_{direction}_intensity"] = np.where(phase < 4, 100.0, 0.0)
        frame[f"target_{direction}_within_5d"] = phase < 4
        frame[f"target_{direction}_within_10d"] = phase < 7
        frame[f"target_{direction}_within_20d"] = phase < 12
    return frame


def test_walk_forward_exports_only_test_year_predictions_and_audits():
    result = run_walk_forward_training(
        _synthetic_training_daily(),
        model_ids=("elastic_net",),
        first_validation_year=2018,
        boundary_gap=5,
        annual_episode_budget=3,
        version="test_walk_forward",
    )

    assert set(result.signal_daily["test_year"]) == {2019, 2020}
    assert set(result.signal_daily["date"].dt.year) == {2019, 2020}
    assert len(result.probability_metrics) == 2 * 1 * 2 * 3
    assert len(result.fit_audit) == 2 * 1 * 2 * 3
    assert result.thresholds["validation_episode_count"].le(3).all()
    assert result.folds["boundary_gap_trade_days"].eq(5).all()
    probabilities = result.signal_daily[
        [
            "pred_probability_5d",
            "pred_probability_10d",
            "pred_probability_20d",
        ]
    ].to_numpy()
    assert (probabilities[:, 0] <= probabilities[:, 1]).all()
    assert (probabilities[:, 1] <= probabilities[:, 2]).all()
    assert result.signal_daily["event_onset"].sum() == len(
        result.signal_episodes
    )


def test_v1_rejects_unfrozen_horizons():
    with pytest.raises(ValueError, match="frozen"):
        run_walk_forward_training(
            _synthetic_training_daily(),
            model_ids=("elastic_net",),
            horizons=(1, 2, 3),
            version="test_walk_forward",
        )


def test_frozen_v2_rejects_a_model_subset():
    with pytest.raises(ValueError, match="model_ids are frozen"):
        run_walk_forward_training(
            _synthetic_training_daily(),
            model_ids=("elastic_net",),
            version=TRAINING_VERSION_V2,
        )


def test_v3_uses_short_horizon_score_and_caps_each_alert_episode():
    result = run_walk_forward_training(
        _synthetic_training_daily(),
        first_validation_year=2018,
        boundary_gap=5,
        annual_episode_budget=3,
        version=TRAINING_VERSION_V3,
    )

    assert result.thresholds["score_weight_5d"].eq(0.7).all()
    assert result.thresholds["score_weight_10d"].eq(0.3).all()
    assert result.thresholds["score_weight_20d"].eq(0.0).all()
    assert result.thresholds["max_alert_active_days"].eq(2).all()
    assert result.thresholds["raw_test_active_days"].ge(
        result.thresholds["test_active_days"]
    ).all()
    assert result.signal_episodes["active_days"].le(2).all()
    expected_score = 100 * (
        0.7 * result.signal_daily["pred_probability_5d"]
        + 0.3 * result.signal_daily["pred_probability_10d"]
    )
    np.testing.assert_allclose(result.signal_daily["pred_score"], expected_score)
