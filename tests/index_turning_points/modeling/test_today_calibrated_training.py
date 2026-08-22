import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.modeling.dataset import today_feature_columns
from research.index_turning_points.modeling.today_calibrated_training import (
    TODAY_CALIBRATED_MODEL_IDS,
    TODAY_CALIBRATED_TRAINING_VERSION,
    run_calibrated_today_walk_forward_training,
)


def _synthetic_today_training_daily() -> pd.DataFrame:
    dates = pd.bdate_range("2012-01-02", "2020-12-31")
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


def test_v2_outputs_calibrated_probabilities_reliability_and_alert_policy():
    result = run_calibrated_today_walk_forward_training(
        _synthetic_today_training_daily(),
        model_ids=("elastic_net",),
        first_test_year=2019,
        calibration_year_count=3,
        boundary_gap=5,
        annual_episode_budget=3,
        version="test_today_calibrated_walk_forward",
    )

    assert set(result.signal_daily["test_year"]) == {2019, 2020}
    assert result.signal_daily["probability_status"].eq("calibrated").all()
    probability = result.signal_daily["pred_probability_today"]
    np.testing.assert_allclose(result.signal_daily["pred_score"], 100 * probability)
    assert result.probability_metrics["expected_calibration_error"].notna().all()
    assert result.calibration_reliability["rows"].sum() == len(result.signal_daily)
    assert result.thresholds["entry_probability"].ge(0.50).all()
    assert result.thresholds["exit_probability"].eq(0.30).all()
    assert result.thresholds["cooldown_days"].eq(10).all()
    assert result.thresholds["calibration_episode_budget"].eq(9).all()


def test_insufficient_calibration_events_disable_formal_bottom_alerts():
    source = _synthetic_today_training_daily()
    calibration_window = source["date"].dt.year.isin([2016, 2017, 2018])
    source.loc[calibration_window, "truth_bottom_in_strict_lobe"] = False
    source.loc[calibration_window, "truth_bottom_intensity"] = 0.0

    result = run_calibrated_today_walk_forward_training(
        source,
        model_ids=("elastic_net",),
        first_test_year=2019,
        calibration_year_count=3,
        boundary_gap=5,
        version="test_today_calibrated_walk_forward",
    )

    bottom_2019 = result.signal_daily[
        result.signal_daily["direction"].eq("bottom")
        & result.signal_daily["test_year"].eq(2019)
    ]
    assert bottom_2019["probability_status"].eq(
        "insufficient_calibration_events"
    ).all()
    assert not bottom_2019["triggered"].any()
    assert bottom_2019["pred_probability_today"].nunique() > 1


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"model_ids": ("elastic_net",)}, "model_ids"),
        ({"calibration_year_count": 2}, "calibration_year_count"),
        ({"annual_episode_budget": 5}, "annual_episode_budget"),
        ({"min_entry_probability": 0.6}, "min_entry_probability"),
        ({"cooldown_days": 5}, "cooldown_days"),
    ],
)
def test_frozen_v2_rejects_parameter_changes(override, expected):
    with pytest.raises(ValueError, match=rf"{expected}.*frozen"):
        run_calibrated_today_walk_forward_training(
            _synthetic_today_training_daily(),
            version=TODAY_CALIBRATED_TRAINING_VERSION,
            **override,
        )

    assert TODAY_CALIBRATED_MODEL_IDS == ("elastic_net", "shallow_gbdt")
