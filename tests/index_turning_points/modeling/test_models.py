import numpy as np
import pandas as pd

from research.index_turning_points.modeling.models import (
    SIMPLE_RULE_FEATURES,
    SHORT_HORIZON_SCORE_WEIGHTS,
    SigmoidCalibrator,
    fit_probability_model,
    project_nested_probabilities,
    score_nested_probabilities,
)


def test_probability_projection_enforces_nested_horizons():
    raw = np.array([[0.6, 0.2, 0.8], [0.8, 0.2, 0.4], [0.1, 0.2, 0.3]])

    result = project_nested_probabilities(raw)

    np.testing.assert_allclose(result[0], [0.4, 0.4, 0.8])
    np.testing.assert_allclose(result[1], [1.4 / 3.0] * 3)
    np.testing.assert_allclose(result[2], raw[2])
    assert (result[:, 0] <= result[:, 1]).all()
    assert (result[:, 1] <= result[:, 2]).all()
    np.testing.assert_allclose(
        score_nested_probabilities(result),
        100 * (result @ np.array([0.5, 0.3, 0.2])),
    )
    np.testing.assert_allclose(
        score_nested_probabilities(
            result,
            weights=SHORT_HORIZON_SCORE_WEIGHTS,
        ),
        100 * (result @ np.array([0.7, 0.3, 0.0])),
    )


def test_calibrator_has_deterministic_one_class_fallback():
    calibrator = SigmoidCalibrator().fit(
        np.array([0.1, 0.2, 0.3]), np.array([0, 0, 0])
    )

    assert calibrator.status == "constant_validation_class"
    np.testing.assert_allclose(calibrator.predict(np.array([0.2, 0.8])), 0.2)


def test_both_frozen_classifiers_fit_and_emit_probabilities():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 4))
    target = (features[:, 0] + features[:, 1] > 0).astype(int)

    for model_id in ("elastic_net", "shallow_gbdt", "shallow_xgboost"):
        model = fit_probability_model(model_id, features, target)
        probability = model.predict_proba(features[:5])[:, 1]
        assert probability.shape == (5,)
        assert ((probability >= 0.0) & (probability <= 1.0)).all()


def test_simple_rule_weights_do_not_depend_on_training_labels():
    columns = [name for name, _sign in SIMPLE_RULE_FEATURES["top"]]
    features = pd.DataFrame(
        np.arange(60, dtype=float).reshape(10, 6), columns=columns
    )
    first = fit_probability_model(
        "simple_rule", features, np.zeros(10), direction="top"
    )
    second = fit_probability_model(
        "simple_rule", features, np.ones(10), direction="top"
    )

    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features)
    )
