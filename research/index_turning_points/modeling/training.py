"""Expanding walk-forward training and out-of-sample signal generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from ..signals.events import build_signal_events
from .alerts import (
    DEFAULT_ANNUAL_EPISODE_BUDGET,
    DEFAULT_MAX_ALERT_ACTIVE_DAYS,
    count_contiguous_episodes,
    limit_alert_duration,
    select_episode_budget_threshold,
)
from .dataset import feature_columns
from .models import (
    MODEL_IDS,
    SUPPORTED_MODEL_IDS,
    DEFAULT_SCORE_WEIGHTS,
    SHORT_HORIZON_SCORE_WEIGHTS,
    ConstantProbabilityModel,
    SimpleRuleModel,
    SigmoidCalibrator,
    fit_probability_model,
    project_nested_probabilities,
    score_nested_probabilities,
)
from .targets import DEFAULT_HORIZONS
from .walk_forward import (
    DEFAULT_BOUNDARY_GAP,
    DEFAULT_FIRST_VALIDATION_YEAR,
    build_yearly_expanding_folds,
)


TRAINING_VERSION_V1 = "all_a_ml_walk_forward_v1"
TRAINING_VERSION_V2 = "all_a_ml_walk_forward_v2"
TRAINING_VERSION_V3 = "all_a_ml_walk_forward_v3"
TRAINING_VERSION = TRAINING_VERSION_V3
FROZEN_MODEL_IDS_BY_VERSION = {
    TRAINING_VERSION_V1: ("elastic_net", "shallow_gbdt"),
    TRAINING_VERSION_V2: MODEL_IDS,
    TRAINING_VERSION_V3: MODEL_IDS,
}
SCORE_WEIGHTS_BY_VERSION = {
    TRAINING_VERSION_V1: DEFAULT_SCORE_WEIGHTS,
    TRAINING_VERSION_V2: DEFAULT_SCORE_WEIGHTS,
    TRAINING_VERSION_V3: SHORT_HORIZON_SCORE_WEIGHTS,
}
MAX_ALERT_ACTIVE_DAYS_BY_VERSION = {
    TRAINING_VERSION_V3: DEFAULT_MAX_ALERT_ACTIVE_DAYS,
}
DIRECTIONS = ("top", "bottom")


@dataclass(frozen=True)
class WalkForwardResult:
    signal_daily: pd.DataFrame
    signal_episodes: pd.DataFrame
    probability_metrics: pd.DataFrame
    folds: pd.DataFrame
    thresholds: pd.DataFrame
    fit_audit: pd.DataFrame


def run_walk_forward_training(
    training_daily: pd.DataFrame,
    *,
    model_ids: tuple[str, ...] = MODEL_IDS,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    first_validation_year: int = DEFAULT_FIRST_VALIDATION_YEAR,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
    annual_episode_budget: int = DEFAULT_ANNUAL_EPISODE_BUDGET,
    version: str = TRAINING_VERSION,
) -> WalkForwardResult:
    """Fit yearly models and emit only test-year out-of-sample predictions."""

    if horizons != DEFAULT_HORIZONS:
        raise ValueError(f"ML horizons are frozen at {DEFAULT_HORIZONS}")
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be non-empty and unique")
    unknown_models = set(model_ids).difference(SUPPORTED_MODEL_IDS)
    if unknown_models:
        raise ValueError(f"unknown model_ids: {sorted(unknown_models)}")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be non-empty")
    frozen_model_ids = FROZEN_MODEL_IDS_BY_VERSION.get(version)
    if frozen_model_ids is not None and model_ids != frozen_model_ids:
        raise ValueError(
            f"{version} model_ids are frozen at {frozen_model_ids}"
        )
    score_weights = SCORE_WEIGHTS_BY_VERSION.get(version, DEFAULT_SCORE_WEIGHTS)
    max_alert_active_days = MAX_ALERT_ACTIVE_DAYS_BY_VERSION.get(version)
    source = _validate_training_daily(training_daily, horizons=horizons)
    columns = feature_columns()
    features = _feature_matrix(source, columns)
    folds = build_yearly_expanding_folds(
        source["date"],
        first_validation_year=first_validation_year,
        boundary_gap=boundary_gap,
    )
    date_index = pd.DatetimeIndex(source["date"])
    fold_records = []
    prediction_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    threshold_records: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []

    for fold in folds:
        fold_record = fold.audit_record(date_index)
        fold_record["boundary_gap_trade_days"] = boundary_gap
        fold_records.append(fold_record)
        train_positions = np.asarray(fold.train_positions, dtype=int)
        validation_positions = np.asarray(fold.validation_positions, dtype=int)
        test_positions = np.asarray(fold.test_positions, dtype=int)

        for model_id in model_ids:
            for direction in DIRECTIONS:
                validation_probability = np.empty((len(validation_positions), 3))
                test_probability = np.empty((len(test_positions), 3))
                target_arrays: dict[int, pd.Series] = {}
                for horizon_number, horizon in enumerate(horizons):
                    target_column = f"target_{direction}_within_{horizon}d"
                    target = _nullable_binary(source[target_column], target_column)
                    target_arrays[horizon] = target
                    train_target = target.iloc[train_positions]
                    train_known = train_target.notna().to_numpy()
                    validation_target = target.iloc[validation_positions]
                    validation_known = validation_target.notna().to_numpy()
                    if not train_known.any() or not validation_known.any():
                        raise ValueError(
                            f"{fold.fold_id} has no known {target_column} labels"
                        )

                    model = fit_probability_model(
                        model_id,
                        features.iloc[train_positions[train_known]],
                        train_target.iloc[np.flatnonzero(train_known)]
                        .astype(int)
                        .to_numpy(),
                        direction=direction,
                    )
                    raw_validation = model.predict_proba(
                        features.iloc[validation_positions]
                    )[:, 1]
                    raw_test = model.predict_proba(features.iloc[test_positions])[:, 1]
                    calibrator = SigmoidCalibrator().fit(
                        raw_validation[validation_known],
                        validation_target.iloc[np.flatnonzero(validation_known)]
                        .astype(int)
                        .to_numpy(),
                    )
                    validation_probability[:, horizon_number] = calibrator.predict(
                        raw_validation
                    )
                    test_probability[:, horizon_number] = calibrator.predict(raw_test)
                    train_labels = (
                        train_target.iloc[np.flatnonzero(train_known)]
                        .astype(int)
                        .to_numpy()
                    )
                    validation_labels = (
                        validation_target.iloc[np.flatnonzero(validation_known)]
                        .astype(int)
                        .to_numpy()
                    )
                    fit_records.append(
                        {
                            "fold_id": fold.fold_id,
                            "model_id": model_id,
                            "direction": direction,
                            "horizon_trade_days": horizon,
                            "train_rows": len(train_labels),
                            "train_positives": int(train_labels.sum()),
                            "validation_rows": len(validation_labels),
                            "validation_positives": int(validation_labels.sum()),
                            "model_fit_status": (
                                "fixed_rule"
                                if isinstance(model, SimpleRuleModel)
                                else (
                                    "constant_training_class"
                                    if isinstance(model, ConstantProbabilityModel)
                                    else "fitted"
                                )
                            ),
                            "calibration_status": calibrator.status,
                        }
                    )

                validation_probability = project_nested_probabilities(
                    validation_probability
                )
                test_probability = project_nested_probabilities(test_probability)
                validation_scores = score_nested_probabilities(
                    validation_probability,
                    weights=score_weights,
                )
                test_scores = score_nested_probabilities(
                    test_probability,
                    weights=score_weights,
                )
                selected = select_episode_budget_threshold(
                    validation_scores,
                    max_episodes=annual_episode_budget,
                )
                raw_validation_triggered = (
                    validation_scores >= selected.threshold
                )
                raw_test_triggered = test_scores >= selected.threshold
                validation_triggered = _apply_alert_duration_limit(
                    raw_validation_triggered,
                    max_active_days=max_alert_active_days,
                )
                test_triggered = _apply_alert_duration_limit(
                    raw_test_triggered,
                    max_active_days=max_alert_active_days,
                )
                threshold_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "validation_year": fold.validation_year,
                        "test_year": fold.test_year,
                        "model_id": model_id,
                        "direction": direction,
                        "annual_episode_budget": annual_episode_budget,
                        "threshold": selected.threshold,
                        "score_weight_5d": score_weights[0],
                        "score_weight_10d": score_weights[1],
                        "score_weight_20d": score_weights[2],
                        "max_alert_active_days": max_alert_active_days,
                        "validation_episode_count": count_contiguous_episodes(
                            validation_triggered
                        ),
                        "raw_validation_active_days": selected.active_days,
                        "validation_active_days": int(
                            validation_triggered.sum()
                        ),
                        "test_episode_count": count_contiguous_episodes(
                            test_triggered
                        ),
                        "raw_test_active_days": int(raw_test_triggered.sum()),
                        "test_active_days": int(test_triggered.sum()),
                    }
                )

                prediction = pd.DataFrame(
                    {
                        "date": source.iloc[test_positions]["date"].to_numpy(),
                        "signal_id": f"ml_{model_id}",
                        "model_id": model_id,
                        "direction": direction,
                        "version": version,
                        "fold_id": fold.fold_id,
                        "validation_year": fold.validation_year,
                        "test_year": fold.test_year,
                        "raw_value": test_scores,
                        "pred_score": test_scores,
                        "threshold": selected.threshold,
                        "raw_triggered": raw_test_triggered,
                        "triggered": test_triggered,
                        "universe_size": 1,
                        "valid_count": 1,
                        "truth_intensity": pd.to_numeric(
                            source.iloc[test_positions][
                                f"truth_{direction}_intensity"
                            ],
                            errors="coerce",
                        ).to_numpy(),
                    }
                )
                for horizon_number, horizon in enumerate(horizons):
                    prediction[f"pred_probability_{horizon}d"] = (
                        test_probability[:, horizon_number]
                    )
                    actual = target_arrays[horizon].iloc[test_positions]
                    prediction[f"actual_entry_within_{horizon}d"] = actual.to_numpy()
                    known = actual.notna().to_numpy()
                    labels = actual.iloc[np.flatnonzero(known)].astype(int).to_numpy()
                    metric_records.append(
                        _probability_metric_record(
                            labels,
                            test_probability[known, horizon_number],
                            fold_id=fold.fold_id,
                            test_year=fold.test_year,
                            model_id=model_id,
                            direction=direction,
                            horizon=horizon,
                        )
                    )
                prediction_frames.append(prediction)

    signal_input = pd.concat(prediction_frames, ignore_index=True)
    signal_daily, signal_episodes = build_signal_events(signal_input)
    return WalkForwardResult(
        signal_daily=signal_daily,
        signal_episodes=signal_episodes,
        probability_metrics=pd.DataFrame(metric_records),
        folds=pd.DataFrame(fold_records),
        thresholds=pd.DataFrame(threshold_records),
        fit_audit=pd.DataFrame(fit_records),
    )


def _apply_alert_duration_limit(
    triggered: np.ndarray,
    *,
    max_active_days: int | None,
) -> np.ndarray:
    if max_active_days is None:
        return np.asarray(triggered, dtype=bool).reshape(-1)
    return limit_alert_duration(triggered, max_active_days=max_active_days)


def _validate_training_daily(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("training_daily must be a non-empty DataFrame")
    required = {
        "date",
        *feature_columns(),
        *(f"truth_{direction}_intensity" for direction in DIRECTIONS),
        *(
            f"target_{direction}_within_{horizon}d"
            for direction in DIRECTIONS
            for horizon in horizons
        ),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training_daily is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("training_daily contains an invalid date")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("training_daily dates must be unique and increasing")
    return result.reset_index(drop=True)


def _feature_matrix(
    source: pd.DataFrame,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    result = pd.DataFrame(index=source.index)
    for column in columns:
        values = source[column]
        if column == "index_price_available":
            values = _strict_boolean(values, column).astype(int)
        numeric = pd.to_numeric(values, errors="coerce")
        finite = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise ValueError(f"feature contains infinity: {column}")
        result[column] = numeric.astype(float)
    return result


def _strict_boolean(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain booleans")
    return normalized.map(mapping).astype(bool)


def _nullable_binary(values: pd.Series, name: str) -> pd.Series:
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    known = values.notna() & values.astype(str).str.strip().ne("")
    normalized = values.loc[known].astype(str).str.strip().str.lower()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    if not normalized.isin(mapping).all():
        invalid = sorted(normalized[~normalized.isin(mapping)].unique())
        raise ValueError(f"{name} contains invalid binary values: {invalid}")
    result.loc[known] = normalized.map(mapping).astype(int)
    return result


def _probability_metric_record(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    fold_id: str,
    test_year: int,
    model_id: str,
    direction: str,
    horizon: int,
) -> dict[str, object]:
    labels = np.asarray(target, dtype=int)
    predictions = np.asarray(probability, dtype=float)
    record: dict[str, object] = {
        "fold_id": fold_id,
        "test_year": test_year,
        "model_id": model_id,
        "direction": direction,
        "horizon_trade_days": horizon,
        "rows": len(labels),
        "positives": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else np.nan,
        "brier_score": np.nan,
        "log_loss": np.nan,
        "roc_auc": np.nan,
        "average_precision": np.nan,
    }
    if not len(labels):
        return record
    record["brier_score"] = float(brier_score_loss(labels, predictions))
    record["log_loss"] = float(log_loss(labels, predictions, labels=[0, 1]))
    if np.unique(labels).size == 2:
        record["roc_auc"] = float(roc_auc_score(labels, predictions))
        record["average_precision"] = float(
            average_precision_score(labels, predictions)
        )
    return record
