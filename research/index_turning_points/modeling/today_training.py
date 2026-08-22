"""Current-day strict-lobe probability walk-forward training."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..signals.events import build_signal_events
from .alerts import (
    DEFAULT_ANNUAL_EPISODE_BUDGET,
    count_contiguous_episodes,
    select_episode_budget_threshold,
)
from .dataset import today_feature_columns
from .models import (
    ConstantProbabilityModel,
    SigmoidCalibrator,
    fit_probability_model,
)
from .training import (
    WalkForwardResult,
    _feature_matrix,
    _nullable_binary,
    _probability_metric_record,
)
from .walk_forward import (
    DEFAULT_BOUNDARY_GAP,
    DEFAULT_FIRST_VALIDATION_YEAR,
    build_yearly_expanding_folds,
)


TODAY_TRAINING_VERSION = "all_a_ml_today_walk_forward_v1"
TODAY_MODEL_IDS = ("elastic_net", "shallow_gbdt")
DIRECTIONS = ("top", "bottom")
TARGET_SEMANTICS = "today_strict_lobe_membership"


def run_today_walk_forward_training(
    training_daily: pd.DataFrame,
    *,
    model_ids: tuple[str, ...] = TODAY_MODEL_IDS,
    first_validation_year: int = DEFAULT_FIRST_VALIDATION_YEAR,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
    annual_episode_budget: int = DEFAULT_ANNUAL_EPISODE_BUDGET,
    version: str = TODAY_TRAINING_VERSION,
) -> WalkForwardResult:
    """Fit independent top/bottom nowcasts and emit test-year probabilities."""

    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError("model_ids must be non-empty and unique")
    unknown = set(model_ids).difference(TODAY_MODEL_IDS)
    if unknown:
        raise ValueError(f"unknown current-day model_ids: {sorted(unknown)}")
    if version == TODAY_TRAINING_VERSION and model_ids != TODAY_MODEL_IDS:
        raise ValueError(
            f"{TODAY_TRAINING_VERSION} model_ids are frozen at {TODAY_MODEL_IDS}"
        )
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be non-empty")

    source = _validate_today_training_daily(training_daily)
    columns = today_feature_columns()
    features = _feature_matrix(source, columns)
    folds = build_yearly_expanding_folds(
        source["date"],
        first_validation_year=first_validation_year,
        boundary_gap=boundary_gap,
    )
    date_index = pd.DatetimeIndex(source["date"])
    fold_records: list[dict[str, object]] = []
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
                target_column = f"truth_{direction}_in_strict_lobe"
                target = _nullable_binary(source[target_column], target_column)
                train_target = target.iloc[train_positions]
                validation_target = target.iloc[validation_positions]
                train_known = train_target.notna().to_numpy()
                validation_known = validation_target.notna().to_numpy()
                if not train_known.any() or not validation_known.any():
                    raise ValueError(
                        f"{fold.fold_id} has no known {target_column} labels"
                    )

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
                model = fit_probability_model(
                    model_id,
                    features.iloc[train_positions[train_known]],
                    train_labels,
                    direction=direction,
                )
                raw_validation = model.predict_proba(
                    features.iloc[validation_positions]
                )[:, 1]
                raw_test = model.predict_proba(features.iloc[test_positions])[:, 1]
                calibrator = SigmoidCalibrator().fit(
                    raw_validation[validation_known], validation_labels
                )
                validation_probability = calibrator.predict(raw_validation)
                test_probability = calibrator.predict(raw_test)
                validation_scores = 100.0 * validation_probability
                test_scores = 100.0 * test_probability
                selected = select_episode_budget_threshold(
                    validation_scores,
                    max_episodes=annual_episode_budget,
                )
                validation_triggered = validation_scores >= selected.threshold
                test_triggered = test_scores >= selected.threshold

                fit_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "model_id": model_id,
                        "direction": direction,
                        "target_semantics": TARGET_SEMANTICS,
                        "train_rows": len(train_labels),
                        "train_positives": int(train_labels.sum()),
                        "validation_rows": len(validation_labels),
                        "validation_positives": int(validation_labels.sum()),
                        "model_fit_status": (
                            "constant_training_class"
                            if isinstance(model, ConstantProbabilityModel)
                            else "fitted"
                        ),
                        "calibration_status": calibrator.status,
                    }
                )
                threshold_records.append(
                    {
                        "fold_id": fold.fold_id,
                        "validation_year": fold.validation_year,
                        "test_year": fold.test_year,
                        "model_id": model_id,
                        "direction": direction,
                        "target_semantics": TARGET_SEMANTICS,
                        "annual_episode_budget": annual_episode_budget,
                        "threshold": selected.threshold,
                        "validation_episode_count": count_contiguous_episodes(
                            validation_triggered
                        ),
                        "validation_active_days": int(
                            validation_triggered.sum()
                        ),
                        "test_episode_count": count_contiguous_episodes(
                            test_triggered
                        ),
                        "test_active_days": int(test_triggered.sum()),
                    }
                )

                actual = target.iloc[test_positions]
                prediction = pd.DataFrame(
                    {
                        "date": source.iloc[test_positions]["date"].to_numpy(),
                        "signal_id": f"ml_today_{model_id}",
                        "model_id": model_id,
                        "direction": direction,
                        "version": version,
                        "fold_id": fold.fold_id,
                        "validation_year": fold.validation_year,
                        "test_year": fold.test_year,
                        "raw_value": test_scores,
                        "pred_score": test_scores,
                        "pred_probability_today": test_probability,
                        "threshold": selected.threshold,
                        "raw_triggered": test_triggered,
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
                prediction_frames.append(prediction)

                known = actual.notna().to_numpy()
                labels = actual.iloc[np.flatnonzero(known)].astype(int).to_numpy()
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
                metric_records.append(metric)

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


def _validate_today_training_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("training_daily must be a non-empty DataFrame")
    required = {
        "date",
        *today_feature_columns(),
        *(f"truth_{direction}_intensity" for direction in DIRECTIONS),
        *(f"truth_{direction}_in_strict_lobe" for direction in DIRECTIONS),
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
