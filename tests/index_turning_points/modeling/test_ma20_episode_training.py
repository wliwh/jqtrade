import numpy as np
import pandas as pd

from research.index_turning_points.modeling.ma20_episode_dataset import (
    MA20_EPISODE_FEATURE_COLUMNS,
)
from research.index_turning_points.modeling.ma20_episode_training import (
    run_ma20_episode_walk_forward_training,
)


def _training_fixture():
    calendar = pd.bdate_range("2012-01-02", "2020-12-31")
    daily_calendar = pd.DataFrame(
        {"date": calendar, "universe_size": 1, "valid_count": 1}
    )
    records = []
    for year in range(2012, 2021):
        year_dates = calendar[calendar.year == year]
        positions = np.linspace(10, len(year_dates) - 11, 8, dtype=int)
        for direction_number, direction in enumerate(("top", "bottom")):
            for candidate_number, position in enumerate(positions):
                onset = pd.Timestamp(year_dates[position + direction_number])
                matched = (candidate_number + year + direction_number) % 3 != 0
                record = {
                    "candidate_episode_id": (
                        f"candidate::{direction}::{year}::{candidate_number}"
                    ),
                    "direction": direction,
                    "onset_date": onset,
                    "candidate_year": year,
                    "target_operational_match": matched,
                    "target_operational_strict_match": matched and candidate_number % 2 == 0,
                    "target_legacy_window_20d_match": matched,
                    "target_legacy_strict_match": matched and candidate_number % 2 == 0,
                }
                for feature_number, column in enumerate(
                    MA20_EPISODE_FEATURE_COLUMNS, start=1
                ):
                    record[column] = (
                        feature_number
                        + 0.01 * year
                        + 0.1 * candidate_number
                        + 0.05 * direction_number
                    )
                records.append(record)
    return pd.DataFrame(records), daily_calendar


def test_walk_forward_scores_candidates_and_emits_one_day_filtered_alerts():
    candidates, calendar = _training_fixture()

    result = run_ma20_episode_walk_forward_training(
        candidates,
        calendar,
        first_oof_year=2015,
        first_test_year=2018,
        boundary_gap=5,
        min_calibration_positives=2,
        min_calibration_negatives=2,
        annual_candidate_budget=6,
        min_selected_candidates=2,
        min_match_recall=0.5,
        version="ma20_episode_test_v1",
    )

    predictions = result.candidate_predictions
    assert len(predictions) == 48
    assert predictions["pred_probability_episode_match"].between(0, 1).all()
    assert set(predictions["test_year"]) == {2018, 2019, 2020}
    assert int(result.signal_daily["raw_triggered"].sum()) == 48
    assert int(result.signal_daily["triggered"].sum()) <= 48
    assert result.signal_daily["probability_status"].notna().all()
    assert result.signal_daily["filter_status"].notna().all()
    assert result.signal_episodes["active_days"].eq(1).all()
    assert len(result.thresholds) == 6
