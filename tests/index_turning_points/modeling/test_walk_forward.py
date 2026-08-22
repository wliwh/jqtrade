import pandas as pd
import pytest

from research.index_turning_points.modeling.walk_forward import (
    build_yearly_calibration_folds,
    build_yearly_expanding_folds,
)


def test_yearly_folds_are_expanding_and_leave_each_boundary_gap():
    dates = pd.bdate_range("2016-01-01", "2020-12-31")

    folds = build_yearly_expanding_folds(
        dates, first_validation_year=2018, boundary_gap=20
    )

    assert [fold.test_year for fold in folds] == [2019, 2020]
    first = folds[0]
    assert len(first.raw_train_positions) - len(first.train_positions) == 20
    assert (
        len(first.raw_validation_positions)
        - len(first.validation_positions)
        == 20
    )
    assert dates[first.train_positions[-1]] < dates[first.raw_train_positions[-1]]
    assert dates[first.validation_positions[-1]] < dates[first.test_positions[0]]
    assert len(folds[1].train_positions) > len(first.train_positions)


def test_fold_builder_rejects_a_gap_larger_than_available_history():
    dates = pd.bdate_range("2017-12-01", "2019-01-31")

    with pytest.raises(ValueError, match="too few rows"):
        build_yearly_expanding_folds(
            dates, first_validation_year=2018, boundary_gap=30
        )


def test_builds_three_year_calibration_folds_without_training_overlap():
    dates = pd.bdate_range("2012-01-02", "2020-12-31")

    folds = build_yearly_calibration_folds(
        dates,
        first_test_year=2019,
        calibration_year_count=3,
        boundary_gap=5,
    )

    assert [fold.test_year for fold in folds] == [2019, 2020]
    first = folds[0]
    train_dates = dates[list(first.train_positions)]
    calibration_dates = dates[list(first.calibration_positions)]
    test_dates = dates[list(first.test_positions)]
    assert train_dates.max().year == 2015
    assert set(calibration_dates.year) == {2016, 2017, 2018}
    assert test_dates.min().year == 2019
    assert train_dates.max() < calibration_dates.min() < test_dates.min()
    assert len(first.raw_train_positions) - len(first.train_positions) == 5
    assert (
        len(first.raw_calibration_positions)
        - len(first.calibration_positions)
        == 5
    )
