import numpy as np

from research.index_turning_points.modeling.calibration import (
    PriorShiftCalibrator,
    calibration_reliability,
)


def test_prior_shift_preserves_ranking_and_matches_requested_mean_probability():
    raw = np.array([0.1, 0.2, 0.4, 0.8])

    calibrator = PriorShiftCalibrator().fit(raw, target_prevalence=0.15)
    calibrated = calibrator.predict(raw)

    assert np.all(np.diff(calibrated) > 0.0)
    np.testing.assert_allclose(calibrated.mean(), 0.15, rtol=0.0, atol=1e-8)


def test_fixed_probability_bins_report_weighted_ece():
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.05, 0.15, 0.75, 0.95])

    ece, reliability = calibration_reliability(target, probability, bin_count=10)

    assert reliability["rows"].sum() == 4
    expected = np.mean([0.05, 0.15, 0.25, 0.05])
    np.testing.assert_allclose(ece, expected)
    assert set(reliability["bin_number"]) == {0, 1, 7, 9}
