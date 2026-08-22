"""Frozen probability models, validation calibration and horizon projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


MODEL_IDS = ("simple_rule", "elastic_net", "shallow_xgboost")
SUPPORTED_MODEL_IDS = (*MODEL_IDS, "shallow_gbdt")
RANDOM_STATE = 20260821
DEFAULT_SCORE_WEIGHTS = (0.5, 0.3, 0.2)
SHORT_HORIZON_SCORE_WEIGHTS = (0.7, 0.3, 0.0)
SIMPLE_RULE_FEATURES = {
    "top": (
        ("breadth_ma20_change_10d", -1.0),
        ("breadth_ma60_change_10d", -1.0),
        ("new_high_low_net_ratio_60_change_10d", -1.0),
        ("limit_hit_net_ratio_change_5d", -1.0),
        ("turnover_ratio_pct_p50_change_10d", -1.0),
        ("index_close_to_ma60", 1.0),
    ),
    "bottom": (
        ("breadth_ma20_change_10d", 1.0),
        ("breadth_ma60_change_10d", 1.0),
        ("new_high_low_net_ratio_60_change_10d", 1.0),
        ("limit_hit_net_ratio_change_5d", 1.0),
        ("turnover_ratio_pct_p50_change_10d", 1.0),
        ("index_close_to_ma60", -1.0),
    ),
}


@dataclass
class ConstantProbabilityModel:
    """Small deterministic fallback when a training slice has one class."""

    probability: float

    def predict_proba(self, features: object) -> np.ndarray:
        row_count = len(features)  # type: ignore[arg-type]
        positive = np.full(row_count, self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class SimpleRuleModel:
    """Target-free equal-weight directional rule fitted only for robust scaling."""

    direction: str
    centers: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, features: object) -> "SimpleRuleModel":
        values = self._values(features)
        centers = np.nanmedian(values, axis=0)
        lower = np.nanquantile(values, 0.25, axis=0)
        upper = np.nanquantile(values, 0.75, axis=0)
        scales = upper - lower
        centers = np.where(np.isfinite(centers), centers, 0.0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-12), scales, 1.0)
        self.centers = centers
        self.scales = scales
        return self

    def predict_proba(self, features: object) -> np.ndarray:
        if self.centers is None or self.scales is None:
            raise RuntimeError("simple rule must be fitted before predict")
        values = self._values(features)
        values = np.where(np.isnan(values), self.centers, values)
        signs = np.array(
            [sign for _column, sign in SIMPLE_RULE_FEATURES[self.direction]]
        )
        standardized = np.clip((values - self.centers) / self.scales, -3.0, 3.0)
        raw_score = (standardized * signs).mean(axis=1)
        positive = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -30.0, 30.0)))
        return np.column_stack([1.0 - positive, positive])

    def _values(self, features: object) -> np.ndarray:
        if self.direction not in SIMPLE_RULE_FEATURES:
            raise ValueError("direction must be top or bottom")
        if not hasattr(features, "loc"):
            raise TypeError("simple rule features must be a pandas DataFrame")
        columns = [name for name, _sign in SIMPLE_RULE_FEATURES[self.direction]]
        return features.loc[:, columns].to_numpy(dtype=float)  # type: ignore[union-attr]


@dataclass
class SigmoidCalibrator:
    """Platt-style validation-only sigmoid calibration."""

    estimator: LogisticRegression | None = None
    constant_probability: float | None = None
    status: str = "unfitted"

    def fit(self, raw_probability: np.ndarray, target: np.ndarray) -> "SigmoidCalibrator":
        probability = _finite_probability(raw_probability)
        labels = _binary_target(target)
        if len(probability) != len(labels) or len(labels) == 0:
            raise ValueError("calibration probability and target must align")
        if np.unique(labels).size < 2:
            self.constant_probability = float((labels.sum() + 1.0) / (len(labels) + 2.0))
            self.estimator = None
            self.status = "constant_validation_class"
            return self
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        estimator.fit(_logit(probability).reshape(-1, 1), labels)
        self.estimator = estimator
        self.constant_probability = None
        self.status = "sigmoid"
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        probability = _finite_probability(raw_probability)
        if self.estimator is not None:
            return self.estimator.predict_proba(
                _logit(probability).reshape(-1, 1)
            )[:, 1]
        if self.constant_probability is not None:
            return np.full(len(probability), self.constant_probability, dtype=float)
        raise RuntimeError("calibrator must be fitted before predict")


def make_classifier(model_id: str) -> Pipeline:
    """Create one frozen trainable classifier without fitting it."""

    if model_id == "elastic_net":
        classifier = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=0.05,
            class_weight="balanced",
            max_iter=3000,
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("classifier", classifier),
            ]
        )
    if model_id == "shallow_gbdt":
        classifier = GradientBoostingClassifier(
            n_estimators=60,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=RANDOM_STATE,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("classifier", classifier),
            ]
        )
    if model_id == "shallow_xgboost":
        classifier = XGBClassifier(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            min_child_weight=20.0,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=RANDOM_STATE,
            verbosity=0,
        )
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("classifier", classifier),
            ]
        )
    raise ValueError(f"unknown model_id: {model_id}")


def fit_probability_model(
    model_id: str,
    features: object,
    target: np.ndarray,
    *,
    direction: str | None = None,
) -> Pipeline | ConstantProbabilityModel | SimpleRuleModel:
    """Fit a classifier, with an auditable one-class fallback."""

    if model_id not in SUPPORTED_MODEL_IDS:
        raise ValueError(f"unknown model_id: {model_id}")
    if model_id == "simple_rule":
        if direction not in SIMPLE_RULE_FEATURES:
            raise ValueError("simple_rule requires direction=top or bottom")
        return SimpleRuleModel(direction).fit(features)
    labels = _binary_target(target)
    if len(labels) == 0:
        raise ValueError("training target must not be empty")
    if np.unique(labels).size < 2:
        return ConstantProbabilityModel(float((labels.sum() + 1.0) / (len(labels) + 2.0)))
    model = make_classifier(model_id)
    if model_id in ("shallow_gbdt", "shallow_xgboost"):
        sample_weight = compute_sample_weight(class_weight="balanced", y=labels)
        model.fit(features, labels, classifier__sample_weight=sample_weight)
    else:
        model.fit(features, labels)
    return model


def project_nested_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """L2-project rows onto p5 <= p10 <= p20 using pooled adjacent blocks."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    projected = np.empty_like(values)
    for row_number, row in enumerate(values):
        blocks: list[tuple[float, int]] = []
        for value in np.clip(row, 0.0, 1.0):
            blocks.append((float(value), 1))
            while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
                right_value, right_weight = blocks.pop()
                left_value, left_weight = blocks.pop()
                weight = left_weight + right_weight
                mean = (
                    left_value * left_weight + right_value * right_weight
                ) / weight
                blocks.append((mean, weight))
        projected[row_number] = np.concatenate(
            [np.repeat(value, weight) for value, weight in blocks]
        )
    return projected


def score_nested_probabilities(
    probabilities: np.ndarray,
    *,
    weights: tuple[float, float, float] = DEFAULT_SCORE_WEIGHTS,
) -> np.ndarray:
    """Apply one explicit, normalized weighting to nested probabilities."""

    projected = project_nested_probabilities(probabilities)
    weight_array = np.asarray(weights, dtype=float)
    if (
        weight_array.shape != (3,)
        or not np.isfinite(weight_array).all()
        or (weight_array < 0.0).any()
        or not np.isclose(weight_array.sum(), 1.0)
    ):
        raise ValueError("score weights must be three non-negative values summing to one")
    return 100.0 * (projected @ weight_array)


def _finite_probability(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("probability must be finite")
    return np.clip(result, 1e-6, 1.0 - 1e-6)


def _binary_target(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=int).reshape(-1)
    if not np.isin(result, [0, 1]).all():
        raise ValueError("target must contain only zero and one")
    return result


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))
