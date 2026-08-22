import pandas as pd
import pytest

from research.index_turning_points.modeling.walk_forward import (
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
