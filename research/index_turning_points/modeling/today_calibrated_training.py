"""Multi-year calibrated current-day strict-lobe probability training V2."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd

from ..signals.events import build_signal_events
from .alert_policy import (
    DEFAULT_COOLDOWN_DAYS,
    DEFAULT_EXIT_PROBABILITY,
    DEFAULT_MIN_ENTRY_PROBABILITY,
    apply_hysteresis_cooldown,
)
from .alerts import (
    DEFAULT_ANNUAL_EPISODE_BUDGET,
    count_contiguous_episodes,
    select_episode_budget_threshold,
)
from .calibration import PriorShiftCalibrator, calibration_reliability
from .dataset import today_feature_columns
from .models import ConstantProbabilityModel, SigmoidCalibrator, fit_probability_model
from .today_training import _validate_today_training_daily
from .training import _feature_matrix, _nullable_binary, _probability_metric_record
from .walk_forward import (
    DEFAULT_BOUNDARY_GAP,
    build_yearly_calibration_folds,
)


TODAY_CALIBRATED_TRAINING_VERSION = "all_a_ml_today_walk_forward_v2"
TODAY_CALIBRATED_MODEL_IDS = ("elastic_net", "shallow_gbdt")
DEFAULT_FIRST_TEST_YEAR = 2019
DEFAULT_CALIBRATION_YEAR_COUNT = 3
MIN_CALIBRATION_POSITIVES = 5
MIN_CALIBRATION_NEGATIVES = 30
CALIBRATION_BIN_COUNT = 10
DIRECTIONS = ("top", "bottom")
TARGET_SEMANTICS = "today_strict_lobe_membership"


@dataclass(frozen=True)
class CalibratedWalkForwardResult:
    signal_daily: pd.DataFrame
    signal_episodes: pd.DataFrame
    probability_metrics: pd.DataFrame
    calibration_reliability: pd.DataFrame
    folds: pd.DataFrame
    thresholds: pd.DataFrame
    fit_audit: pd.DataFrame


def run_calibrated_today_walk_forward_training(
    training_daily: pd.DataFrame,
    *,
    model_ids: tuple[str, ...] = TODAY_CALIBRATED_MODEL_IDS,
    first_test_year: int = DEFAULT_FIRST_TEST_YEAR,
    calibration_year_count: int = DEFAULT_CALIBRATION_YEAR_COUNT,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
    annual_episode_budget: int = DEFAULT_ANNUAL_EPISODE_BUDGET,
    min_calibration_positives: int = MIN_CALIBRATION_POSITIVES,
    min_calibration_negatives: int = MIN_CALIBRATION_NEGATIVES,
    min_entry_probability: float = DEFAULT_MIN_ENTRY_PROBABILITY,
    exit_probability: float = DEFAULT_EXIT_PROBABILITY,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
    version: str = TODAY_CALIBRATED_TRAINING_VERSION,
) -> CalibratedWalkForwardResult:
    """Fit V2 models with disjoint three-year calibration and alert policy."""

    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be non-empty")
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be non-empty and unique")
    unknown = set(model_ids).difference(TODAY_CALIBRATED_MODEL_IDS)
    if unknown:
        raise ValueError(f"unknown current-day model_ids: {sorted(unknown)}")
    _validate_positive_integer(first_test_year, "first_test_year")
    _validate_positive_integer(calibration_year_count, "calibration_year_count")
    _validate_nonnegative_integer(boundary_gap, "boundary_gap")
    _validate_positive_integer(annual_episode_budget, "annual_episode_budget")
    _validate_positive_integer(min_calibration_positives, "min_calibration_positives")
    _validate_positive_integer(min_calibration_negatives, "min_calibration_negatives")
    _validate_probability_thresholds(
        min_entry_probability=min_entry_probability,
        exit_probability=exit_probability,
    )
    _validate_nonnegative_integer(cooldown_days, "cooldown_days")
    if version == TODAY_CALIBRATED_TRAINING_VERSION:
        _validate_frozen_v2_parameters(
            model_ids=model_ids,
            first_test_year=first_test_year,
            calibration_year_count=calibration_year_count,
            boundary_gap=boundary_gap,
            annual_episode_budget=annual_episode_budget,
            min_calibration_positives=min_calibration_positives,
            min_calibration_negatives=min_calibration_negatives,
            min_entry_probability=min_entry_probability,
            exit_probability=exit_probability,
            cooldown_days=cooldown_days,
        )

    source = _validate_today_training_daily(training_daily)
    features = _feature_matrix(source, today_feature_columns())
    folds = build_yearly_calibration_folds(
        source["date"],
        first_test_year=first_test_year,
        calibration_year_count=calibration_year_count,
        boundary_gap=boundary_gap,
    )
    date_index = pd.DatetimeIndex(source["date"])
    fold_records: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    reliability_frames: list[pd.DataFrame] = []
    threshold_records: list[dict[str, object]] = []
    fit_records: list[dict[str, object]] = []

    for fold in folds:
        fold_record = fold.audit_record(date_index)
        fold_record["boundary_gap_trade_days"] = boundary_gap
        fold_records.append(fold_record)
        train_positions = np.asarray(fold.train_positions, dtype=int)
        calibration_positions = np.asarray(fold.calibration_positions, dtype=int)
        test_positions = np.asarray(fold.test_positions, dtype=int)

        for model_id in model_ids:
            for direction in DIRECTIONS:
                target_column = f"truth_{direction}_in_strict_lobe"
                target = _nullable_binary(source[target_column], target_column)
                train_target = target.iloc[train_positions]
                calibration_target = target.iloc[calibration_positions]
                train_known = train_target.notna().to_numpy()
                calibration_known = calibration_target.notna().to_numpy()
                if not train_known.any() or not calibration_known.any():
                    raise ValueError(
                        f"{fold.fold_id} has no known {target_column} labels"
                    )
                train_labels = (
                    train_target.iloc[np.flatnonzero(train_known)]
                    .astype(int)
                    .to_numpy()
                )
                calibration_labels = (
                    calibration_target.iloc[np.flatnonzero(calibration_known)]
                    .astype(int)
                    .to_numpy()
                )
                calibration_positives = int(calibration_labels.sum())
                calibration_negatives = len(calibration_labels) - calibration_positives

                model = fit_probability_model(
                    model_id,
                    features.iloc[train_positions[train_known]],
                    train_labels,
                    direction=direction,
                )
                raw_calibration = model.predict_proba(
                    features.iloc[calibration_positions]
                )[:, 1]
                raw_test = model.predict_proba(features.iloc[test_positions])[:, 1]
                sufficient = (
                    calibration_positives >= min_calibration_positives
                    and calibration_negatives >= min_calibration_negatives
                )
                if sufficient:
                    calibrator = SigmoidCalibrator().fit(
                        raw_calibration[calibration_known], calibration_labels
                    )
                    calibration_probability = calibrator.predict(raw_calibration)
                    test_probability = calibrator.predict(raw_test)
                    probability_status = "calibrated"
                    calibration_method = calibrator.status
                else:
                    training_prevalence = float(
                        (train_labels.sum() + 1.0) / (len(train_labels) + 2.0)
                    )
                    calibrator = PriorShiftCalibrator().fit(
                        raw_calibration[calibration_known],
                        target_prevalence=training_prevalence,
                    )
                    calibration_probability = calibrator.predict(raw_calibration)
                    test_probability = calibrator.predict(raw_test)
                    probability_status = "insufficient_calibration_events"
                    calibration_method = "prior_shift_training_prevalence"

                calibration_episode_budget = (
                    annual_episode_budget * calibration_year_count
                )
                if probability_status == "calibrated":
                    selected = select_episode_budget_threshold(
                        100.0 * calibration_probability,
                        max_episodes=calibration_episode_budget,
                    )
                    candidate_probability = selected.threshold / 100.0
                    entry_probability = max(
                        float(min_entry_probability), candidate_probability
                    )
                    raw_calibration_triggered = (
                        calibration_probability >= entry_probability
                    )
                    raw_test_triggered = test_probability >= entry_probability
                    calibration_triggered = apply_hysteresis_cooldown(
                        calibration_probability,
                        entry_threshold=entry_probability,
                        exit_threshold=exit_probability,
                        cooldown_days=cooldown_days,
                    )
                    test_triggered = apply_hysteresis_cooldown(
                        test_probability,
                        entry_threshold=entry_probability,
                        exit_threshold=exit_probability,
                        cooldown_days=cooldown_days,
                    )
                    alert_eligible = True
                else:
                    candidate_probability = np.nan
                    entry_probability = np.nan
                    raw_calibration_triggered = np.zeros(
                        len(calibration_probability), dtype=bool
                    )
                    raw_test_triggered = np.zeros(len(test_probability), dtype=bool)
                    calibration_triggered = raw_calibration_triggered.copy()
                    test_triggered = raw_test_triggered.copy()
                    alert_eligible = False

                fit_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "model_id": model_id,
                        "direction": direction,
                        "target_semantics": TARGET_SEMANTICS,
                        "train_rows": len(train_labels),
                        "train_positives": int(train_labels.sum()),
                        "calibration_rows": len(calibration_labels),
                        "calibration_positives": calibration_positives,
                        "calibration_negatives": calibration_negatives,
                        "model_fit_status": (
                            "constant_training_class"
                            if isinstance(model, ConstantProbabilityModel)
                            else "fitted"
                        ),
                        "calibration_method": calibration_method,
                        "probability_status": probability_status,
                        "alert_eligible": alert_eligible,
                    }
                )
                threshold_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "test_year": fold.test_year,
                        "model_id": model_id,
                        "direction": direction,
                        "probability_status": probability_status,
                        "alert_eligible": alert_eligible,
                        "annual_episode_budget": annual_episode_budget,
                        "calibration_episode_budget": calibration_episode_budget,
                        "candidate_entry_probability": candidate_probability,
                        "entry_probability": entry_probability,
                        "exit_probability": exit_probability,
                        "cooldown_days": cooldown_days,
                        "raw_calibration_episode_count": count_contiguous_episodes(
                            raw_calibration_triggered
                        ),
                        "calibration_episode_count": count_contiguous_episodes(
                            calibration_triggered
                        ),
                        "calibration_active_days": int(
                            calibration_triggered.sum()
                        ),
                        "raw_test_episode_count": count_contiguous_episodes(
                            raw_test_triggered
                        ),
                        "test_episode_count": count_contiguous_episodes(
                            test_triggered
                        ),
                        "test_active_days": int(test_triggered.sum()),
                    }
                )

                actual = target.iloc[test_positions]
                score = 100.0 * test_probability
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "date": source.iloc[test_positions]["date"].to_numpy(),
                            "signal_id": f"ml_today_calibrated_{model_id}",
                            "model_id": model_id,
                            "direction": direction,
                            "version": version,
                            "fold_id": fold.fold_id,
                            "calibration_start_year": fold.calibration_years[0],
                            "calibration_end_year": fold.calibration_years[-1],
                            "test_year": fold.test_year,
                            "raw_value": score,
                            "pred_score": score,
                            "pred_probability_today": test_probability,
                            "probability_status": probability_status,
                            "alert_eligible": alert_eligible,
                            "threshold": 100.0 * entry_probability,
                            "entry_probability": entry_probability,
                            "exit_probability": exit_probability,
                            "cooldown_days": cooldown_days,
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
                            "actual_in_strict_lobe_today": actual.to_numpy(),
                        }
                    )
                )

                known = actual.notna().to_numpy()
                labels = actual.iloc[np.flatnonzero(known)].astype(int).to_numpy()
                ece, reliability = calibration_reliability(
                    labels,
                    test_probability[known],
                    bin_count=CALIBRATION_BIN_COUNT,
                )
                metric = _probability_metric_record(
                    labels,
                    test_probability[known],
                    fold_id=fold.fold_id,
                    test_year=fold.test_year,
                    model_id=model_id,
                    direction=direction,
                    horizon=0,
                )
                metric["target_semantics"] = TARGET_SEMANTICS
                metric["probability_status"] = probability_status
                metric["expected_calibration_error"] = ece
                metric_records.append(metric)
                reliability.insert(0, "direction", direction)
                reliability.insert(0, "model_id", model_id)
                reliability.insert(0, "test_year", fold.test_year)
                reliability.insert(0, "fold_id", fold.fold_id)
                reliability["probability_status"] = probability_status
                reliability_frames.append(reliability)

    signal_input = pd.concat(prediction_frames, ignore_index=True)
    signal_daily, signal_episodes = build_signal_events(signal_input)
    return CalibratedWalkForwardResult(
        signal_daily=signal_daily,
        signal_episodes=signal_episodes,
        probability_metrics=pd.DataFrame(metric_records),
        calibration_reliability=pd.concat(reliability_frames, ignore_index=True),
        folds=pd.DataFrame(fold_records),
        thresholds=pd.DataFrame(threshold_records),
        fit_audit=pd.DataFrame(fit_records),
    )


def _validate_positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_nonnegative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_probability_thresholds(
    *,
    min_entry_probability: float,
    exit_probability: float,
) -> None:
    entry = float(min_entry_probability)
    exit_value = float(exit_probability)
    if (
        not isfinite(entry)
        or not isfinite(exit_value)
        or not 0.0 <= exit_value < entry <= 1.0
    ):
        raise ValueError(
            "probability thresholds must satisfy "
            "0 <= exit_probability < min_entry_probability <= 1"
        )


def _validate_frozen_v2_parameters(**actual: object) -> None:
    expected = {
        "model_ids": TODAY_CALIBRATED_MODEL_IDS,
        "first_test_year": DEFAULT_FIRST_TEST_YEAR,
        "calibration_year_count": DEFAULT_CALIBRATION_YEAR_COUNT,
        "boundary_gap": DEFAULT_BOUNDARY_GAP,
        "annual_episode_budget": DEFAULT_ANNUAL_EPISODE_BUDGET,
        "min_calibration_positives": MIN_CALIBRATION_POSITIVES,
        "min_calibration_negatives": MIN_CALIBRATION_NEGATIVES,
        "min_entry_probability": DEFAULT_MIN_ENTRY_PROBABILITY,
        "exit_probability": DEFAULT_EXIT_PROBABILITY,
        "cooldown_days": DEFAULT_COOLDOWN_DAYS,
    }
    for name, expected_value in expected.items():
        if actual[name] != expected_value:
            raise ValueError(
                f"{TODAY_CALIBRATED_TRAINING_VERSION} {name} is frozen at "
                f"{expected_value!r}"
            )
