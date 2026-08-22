"""Causal walk-forward filtering of MA20 candidate episodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..signals.events import build_signal_events
from .calibration import PriorShiftCalibrator, calibration_reliability
from .episode_filter import (
    DEFAULT_ANNUAL_CANDIDATE_BUDGET,
    DEFAULT_MIN_MATCH_RECALL,
    DEFAULT_MIN_SELECTED_CANDIDATES,
    EpisodeFilterThreshold,
    select_episode_filter_threshold,
)
from .ma20_episode_dataset import ma20_episode_feature_columns
from .models import ConstantProbabilityModel, RANDOM_STATE, SigmoidCalibrator
from .training import _probability_metric_record
from .walk_forward import DEFAULT_BOUNDARY_GAP


MA20_EPISODE_TRAINING_VERSION = "all_a_ma20_episode_match_walk_forward_v1"
MA20_EPISODE_MODEL_ID = "l2_logistic"
FIRST_OOF_YEAR = 2016
FIRST_TEST_YEAR = 2019
MIN_CALIBRATION_POSITIVES = 3
MIN_CALIBRATION_NEGATIVES = 10
CALIBRATION_BIN_COUNT = 10
DIRECTIONS = ("top", "bottom")
TARGET_SEMANTICS = "ma20_candidate_operational_episode_match"


@dataclass(frozen=True)
class Ma20EpisodeWalkForwardResult:
    signal_daily: pd.DataFrame
    signal_episodes: pd.DataFrame
    candidate_predictions: pd.DataFrame
    probability_metrics: pd.DataFrame
    calibration_reliability: pd.DataFrame
    folds: pd.DataFrame
    thresholds: pd.DataFrame
    fit_audit: pd.DataFrame


def run_ma20_episode_walk_forward_training(
    candidate_episodes: pd.DataFrame,
    daily_calendar: pd.DataFrame,
    *,
    first_oof_year: int = FIRST_OOF_YEAR,
    first_test_year: int = FIRST_TEST_YEAR,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
    min_calibration_positives: int = MIN_CALIBRATION_POSITIVES,
    min_calibration_negatives: int = MIN_CALIBRATION_NEGATIVES,
    annual_candidate_budget: int = DEFAULT_ANNUAL_CANDIDATE_BUDGET,
    min_selected_candidates: int = DEFAULT_MIN_SELECTED_CANDIDATES,
    min_match_recall: float = DEFAULT_MIN_MATCH_RECALL,
    version: str = MA20_EPISODE_TRAINING_VERSION,
) -> Ma20EpisodeWalkForwardResult:
    """Generate candidate-level OOS probabilities and filtered daily alerts."""

    _validate_parameters(
        first_oof_year=first_oof_year,
        first_test_year=first_test_year,
        boundary_gap=boundary_gap,
        min_calibration_positives=min_calibration_positives,
        min_calibration_negatives=min_calibration_negatives,
        annual_candidate_budget=annual_candidate_budget,
        min_selected_candidates=min_selected_candidates,
        min_match_recall=min_match_recall,
        version=version,
    )
    source = _validate_candidates(candidate_episodes)
    calendar = _validate_daily_calendar(daily_calendar)
    dates = pd.DatetimeIndex(calendar["date"])
    features = source.loc[:, ma20_episode_feature_columns()].apply(
        pd.to_numeric, errors="coerce"
    )
    max_year = int(dates.year.max())
    raw_frames: list[pd.DataFrame] = []
    fit_records: list[dict[str, object]] = []

    for prediction_year in range(first_oof_year, max_year + 1):
        cutoff = _training_cutoff(dates, prediction_year, boundary_gap)
        for direction in DIRECTIONS:
            train_mask = source["direction"].eq(direction) & source[
                "onset_date"
            ].le(cutoff)
            test_mask = source["direction"].eq(direction) & source[
                "candidate_year"
            ].eq(prediction_year)
            train_positions = np.flatnonzero(train_mask.to_numpy())
            test_positions = np.flatnonzero(test_mask.to_numpy())
            if not len(train_positions):
                raise ValueError(
                    f"{direction} has no training candidates before {prediction_year}"
                )
            train_labels = source.iloc[train_positions][
                "target_operational_match"
            ].astype(int).to_numpy()
            model = _fit_episode_model(features.iloc[train_positions], train_labels)
            raw_probability = (
                model.predict_proba(features.iloc[test_positions])[:, 1]
                if len(test_positions)
                else np.array([], dtype=float)
            )
            fit_records.append(
                {
                    "prediction_year": prediction_year,
                    "model_id": MA20_EPISODE_MODEL_ID,
                    "direction": direction,
                    "training_cutoff_date": cutoff,
                    "boundary_gap_trade_days": boundary_gap,
                    "train_candidates": len(train_positions),
                    "train_matches": int(train_labels.sum()),
                    "test_candidates": len(test_positions),
                    "model_fit_status": (
                        "constant_training_class"
                        if isinstance(model, ConstantProbabilityModel)
                        else "fitted"
                    ),
                }
            )
            if len(test_positions):
                frame = source.iloc[test_positions].copy()
                frame["prediction_year"] = prediction_year
                frame["raw_probability"] = raw_probability
                raw_frames.append(frame)

    raw_oof = pd.concat(raw_frames, ignore_index=True)
    prediction_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, object]] = []
    reliability_frames: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    threshold_records: list[dict[str, object]] = []

    for test_year in range(first_test_year, max_year + 1):
        cutoff = _training_cutoff(dates, test_year, boundary_gap)
        for direction in DIRECTIONS:
            calibration = raw_oof[
                raw_oof["direction"].eq(direction)
                & raw_oof["prediction_year"].ge(first_oof_year)
                & raw_oof["prediction_year"].lt(test_year)
                & raw_oof["onset_date"].le(cutoff)
            ].copy()
            test = raw_oof[
                raw_oof["direction"].eq(direction)
                & raw_oof["prediction_year"].eq(test_year)
            ].copy()
            training = source[
                source["direction"].eq(direction)
                & source["onset_date"].le(cutoff)
            ]
            calibration_labels = calibration[
                "target_operational_match"
            ].astype(int).to_numpy()
            calibration_positives = int(calibration_labels.sum())
            calibration_negatives = len(calibration_labels) - calibration_positives
            training_labels = training["target_operational_match"].astype(int).to_numpy()
            sufficient = (
                calibration_positives >= min_calibration_positives
                and calibration_negatives >= min_calibration_negatives
            )
            raw_calibration = calibration["raw_probability"].to_numpy(dtype=float)
            raw_test = test["raw_probability"].to_numpy(dtype=float)
            if sufficient:
                calibrator = SigmoidCalibrator().fit(
                    raw_calibration, calibration_labels
                )
                calibration_probability = calibrator.predict(raw_calibration)
                test_probability = (
                    calibrator.predict(raw_test)
                    if len(raw_test)
                    else np.array([], dtype=float)
                )
                probability_status = "calibrated"
                calibration_method = calibrator.status
                calibration_year_count = test_year - first_oof_year
                selected = select_episode_filter_threshold(
                    calibration_probability,
                    calibration_labels,
                    calibration_year_count=calibration_year_count,
                    annual_candidate_budget=annual_candidate_budget,
                    min_selected_candidates=min_selected_candidates,
                    min_match_recall=min_match_recall,
                )
                filter_status = selected.status
            else:
                training_prevalence = float(
                    (training_labels.sum() + 1.0) / (len(training_labels) + 2.0)
                )
                if len(raw_calibration):
                    calibrator = PriorShiftCalibrator().fit(
                        raw_calibration,
                        target_prevalence=training_prevalence,
                    )
                    calibration_probability = calibrator.predict(raw_calibration)
                    test_probability = (
                        calibrator.predict(raw_test)
                        if len(raw_test)
                        else np.array([], dtype=float)
                    )
                    calibration_method = "prior_shift_training_prevalence"
                else:
                    calibration_probability = np.array([], dtype=float)
                    test_probability = np.full(
                        len(raw_test), training_prevalence, dtype=float
                    )
                    calibration_method = "constant_training_prevalence"
                probability_status = "insufficient_episode_evidence"
                filter_status = "passthrough_insufficient_episode_evidence"
                selected = _passthrough_threshold(
                    calibration_labels,
                    calibration_year_count=test_year - first_oof_year,
                    annual_candidate_budget=annual_candidate_budget,
                )

            final_alert = (
                test_probability >= selected.threshold
                if filter_status == "selected"
                else np.ones(len(test_probability), dtype=bool)
            )
            test = test.copy()
            test["fold_id"] = f"episode_wf_{test_year}"
            test["test_year"] = test_year
            test["model_id"] = MA20_EPISODE_MODEL_ID
            test["version"] = version
            test["pred_probability_episode_match"] = test_probability
            test["pred_score"] = 100.0 * test_probability
            test["probability_status"] = probability_status
            test["calibration_method"] = calibration_method
            test["filter_status"] = filter_status
            test["filter_threshold_probability"] = selected.threshold
            test["final_alert"] = final_alert
            prediction_frames.append(test)

            labels = test["target_operational_match"].astype(int).to_numpy()
            metric = _probability_metric_record(
                labels,
                test_probability,
                fold_id=f"episode_wf_{test_year}",
                test_year=test_year,
                model_id=MA20_EPISODE_MODEL_ID,
                direction=direction,
                horizon=0,
            )
            metric["target_semantics"] = TARGET_SEMANTICS
            metric["probability_status"] = probability_status
            if len(labels):
                ece, reliability = calibration_reliability(
                    labels,
                    test_probability,
                    bin_count=CALIBRATION_BIN_COUNT,
                )
                reliability.insert(0, "direction", direction)
                reliability.insert(0, "test_year", test_year)
                reliability.insert(0, "fold_id", f"episode_wf_{test_year}")
                reliability["probability_status"] = probability_status
                reliability_frames.append(reliability)
            else:
                ece = np.nan
            metric["expected_calibration_error"] = ece
            metric_records.append(metric)

            retained_matches = int(labels[final_alert].sum()) if len(labels) else 0
            fold_records.append(
                {
                    "fold_id": f"episode_wf_{test_year}",
                    "test_year": test_year,
                    "direction": direction,
                    "training_cutoff_date": cutoff,
                    "boundary_gap_trade_days": boundary_gap,
                    "train_candidates": len(training),
                    "train_matches": int(training_labels.sum()),
                    "calibration_candidates": len(calibration),
                    "calibration_matches": calibration_positives,
                    "test_candidates": len(test),
                    "test_matches": int(labels.sum()),
                }
            )
            threshold_records.append(
                {
                    "fold_id": f"episode_wf_{test_year}",
                    "test_year": test_year,
                    "direction": direction,
                    "probability_status": probability_status,
                    "filter_status": filter_status,
                    "calibration_method": calibration_method,
                    "calibration_candidates": len(calibration),
                    "calibration_matches": calibration_positives,
                    "calibration_nonmatches": calibration_negatives,
                    "threshold_probability": selected.threshold,
                    "annual_candidate_budget": annual_candidate_budget,
                    "candidate_budget": selected.candidate_budget,
                    "min_selected_candidates": min_selected_candidates,
                    "min_match_recall": min_match_recall,
                    "selected_calibration_candidates": selected.selected_candidates,
                    "selected_calibration_matches": selected.selected_matches,
                    "calibration_selected_precision": selected.precision,
                    "calibration_selected_match_recall": selected.match_recall,
                    "calibration_precision_wilson_lower": (
                        selected.precision_wilson_lower
                    ),
                    "test_candidates": len(test),
                    "test_matches": int(labels.sum()),
                    "retained_test_candidates": int(final_alert.sum()),
                    "retained_test_matches": retained_matches,
                    "test_filter_precision": (
                        retained_matches / int(final_alert.sum())
                        if final_alert.any()
                        else np.nan
                    ),
                    "test_match_recall": (
                        retained_matches / int(labels.sum())
                        if labels.sum()
                        else np.nan
                    ),
                }
            )

    candidate_predictions = pd.concat(prediction_frames, ignore_index=True)
    thresholds = pd.DataFrame(threshold_records)
    signal_daily, signal_episodes = _build_filtered_signal_daily(
        candidate_predictions,
        calendar,
        thresholds,
        first_test_year=first_test_year,
        version=version,
    )
    reliability = (
        pd.concat(reliability_frames, ignore_index=True)
        if reliability_frames
        else pd.DataFrame()
    )
    return Ma20EpisodeWalkForwardResult(
        signal_daily=signal_daily,
        signal_episodes=signal_episodes,
        candidate_predictions=candidate_predictions,
        probability_metrics=pd.DataFrame(metric_records),
        calibration_reliability=reliability,
        folds=pd.DataFrame(fold_records),
        thresholds=thresholds,
        fit_audit=pd.DataFrame(fit_records),
    )


def _fit_episode_model(
    features: pd.DataFrame,
    target: np.ndarray,
) -> Pipeline | ConstantProbabilityModel:
    labels = np.asarray(target, dtype=int)
    if np.unique(labels).size < 2:
        return ConstantProbabilityModel(
            float((labels.sum() + 1.0) / (len(labels) + 2.0))
        )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=0.1,
                    solver="lbfgs",
                    max_iter=2000,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(features, labels)
    return model


def _passthrough_threshold(
    labels: np.ndarray,
    *,
    calibration_year_count: int,
    annual_candidate_budget: int,
) -> EpisodeFilterThreshold:
    rows = len(labels)
    matches = int(labels.sum()) if rows else 0
    precision = matches / rows if rows else np.nan
    return EpisodeFilterThreshold(
        threshold=0.0,
        status="passthrough_insufficient_episode_evidence",
        selected_candidates=rows,
        selected_matches=matches,
        precision=precision,
        match_recall=1.0 if matches else np.nan,
        precision_wilson_lower=np.nan,
        candidate_budget=annual_candidate_budget * calibration_year_count,
    )


def _build_filtered_signal_daily(
    predictions: pd.DataFrame,
    calendar: pd.DataFrame,
    thresholds: pd.DataFrame,
    *,
    first_test_year: int,
    version: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = calendar[calendar["date"].dt.year.ge(first_test_year)].copy()
    calendar["test_year"] = calendar["date"].dt.year
    frames: list[pd.DataFrame] = []
    for direction in DIRECTIONS:
        direction_predictions = predictions[predictions["direction"].eq(direction)]
        years = thresholds[thresholds["direction"].eq(direction)][
            [
                "test_year",
                "probability_status",
                "filter_status",
                "threshold_probability",
            ]
        ].rename(
            columns={"threshold_probability": "filter_threshold_probability"}
        )
        frame = calendar.merge(
            years,
            on="test_year",
            how="left",
            validate="many_to_one",
        )
        onset_columns = [
            "onset_date",
            "candidate_episode_id",
            "pred_probability_episode_match",
            "pred_score",
            "target_operational_match",
            "target_operational_strict_match",
            "target_legacy_window_20d_match",
            "target_legacy_strict_match",
            "final_alert",
        ]
        frame = frame.merge(
            direction_predictions[onset_columns],
            left_on="date",
            right_on="onset_date",
            how="left",
            validate="one_to_one",
        ).drop(columns="onset_date")
        candidate_present = frame["candidate_episode_id"].notna()
        final_alert = frame["final_alert"].astype("boolean").fillna(False).astype(bool)
        probability = pd.to_numeric(
            frame["pred_probability_episode_match"], errors="coerce"
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": frame["date"],
                    "signal_id": f"ma20_episode_ml_{MA20_EPISODE_MODEL_ID}",
                    "model_id": MA20_EPISODE_MODEL_ID,
                    "direction": direction,
                    "version": version,
                    "test_year": frame["test_year"],
                    "raw_value": (100.0 * probability).fillna(0.0),
                    "pred_score": frame["pred_score"],
                    "pred_probability_episode_match": probability,
                    "probability_status": frame["probability_status"],
                    "filter_status": frame["filter_status"],
                    "threshold": 100.0
                    * frame["filter_threshold_probability"],
                    "threshold_probability": frame[
                        "filter_threshold_probability"
                    ],
                    "candidate_present": candidate_present,
                    "source_candidate_episode_id": frame[
                        "candidate_episode_id"
                    ].fillna(""),
                    "raw_triggered": candidate_present,
                    "triggered": final_alert,
                    "universe_size": frame["universe_size"],
                    "valid_count": frame["valid_count"],
                    "target_operational_match": frame[
                        "target_operational_match"
                    ].astype("boolean"),
                    "target_operational_strict_match": frame[
                        "target_operational_strict_match"
                    ].astype("boolean"),
                    "target_legacy_window_20d_match": frame[
                        "target_legacy_window_20d_match"
                    ].astype("boolean"),
                    "target_legacy_strict_match": frame[
                        "target_legacy_strict_match"
                    ].astype("boolean"),
                }
            )
        )
    return build_signal_events(pd.concat(frames, ignore_index=True))


def _training_cutoff(
    dates: pd.DatetimeIndex,
    prediction_year: int,
    boundary_gap: int,
) -> pd.Timestamp:
    prior = dates[dates.year < prediction_year]
    if len(prior) <= boundary_gap:
        raise ValueError(
            f"too few calendar rows before {prediction_year} for gap={boundary_gap}"
        )
    used = prior[:-boundary_gap] if boundary_gap else prior
    return pd.Timestamp(used[-1])


def _validate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "candidate_episode_id",
        "direction",
        "onset_date",
        "candidate_year",
        "target_operational_match",
        "target_operational_strict_match",
        "target_legacy_window_20d_match",
        "target_legacy_strict_match",
        *ma20_episode_feature_columns(),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"candidate_episodes is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("candidate_episodes must not be empty")
    result = frame.copy()
    result["candidate_episode_id"] = result["candidate_episode_id"].astype(str)
    result["onset_date"] = pd.to_datetime(result["onset_date"], errors="coerce")
    if result["candidate_episode_id"].duplicated().any():
        raise ValueError("candidate_episode_id must be unique")
    if result["onset_date"].isna().any():
        raise ValueError("candidate onset_date must be valid")
    if not result["direction"].isin(DIRECTIONS).all():
        raise ValueError("candidate direction must be top or bottom")
    expected_year = result["onset_date"].dt.year
    if not pd.to_numeric(result["candidate_year"], errors="coerce").eq(
        expected_year
    ).all():
        raise ValueError("candidate_year must equal onset year")
    for column in (
        "target_operational_match",
        "target_operational_strict_match",
        "target_legacy_window_20d_match",
        "target_legacy_strict_match",
    ):
        result[column] = _strict_bool(result[column], column)
    return result.sort_values(
        ["onset_date", "direction", "candidate_episode_id"]
    ).reset_index(drop=True)


def _validate_daily_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "universe_size", "valid_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily_calendar is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result.empty or result["date"].isna().any():
        raise ValueError("daily_calendar must contain valid dates")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("daily_calendar dates must be unique and increasing")
    return result


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only booleans")
    return normalized.map(mapping).astype(bool)


def _validate_parameters(**values: object) -> None:
    version = values.pop("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("version must be non-empty")
    for name in (
        "first_oof_year",
        "first_test_year",
        "min_calibration_positives",
        "min_calibration_negatives",
        "annual_candidate_budget",
        "min_selected_candidates",
    ):
        value = values[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    boundary_gap = values["boundary_gap"]
    if (
        isinstance(boundary_gap, bool)
        or not isinstance(boundary_gap, int)
        or boundary_gap < 0
    ):
        raise ValueError("boundary_gap must be a non-negative integer")
    recall = float(values["min_match_recall"])
    if not np.isfinite(recall) or not 0.0 < recall <= 1.0:
        raise ValueError("min_match_recall must be between zero and one")
    if int(values["first_oof_year"]) >= int(values["first_test_year"]):
        raise ValueError("first_oof_year must precede first_test_year")
    if version == MA20_EPISODE_TRAINING_VERSION:
        frozen = {
            "first_oof_year": FIRST_OOF_YEAR,
            "first_test_year": FIRST_TEST_YEAR,
            "boundary_gap": DEFAULT_BOUNDARY_GAP,
            "min_calibration_positives": MIN_CALIBRATION_POSITIVES,
            "min_calibration_negatives": MIN_CALIBRATION_NEGATIVES,
            "annual_candidate_budget": DEFAULT_ANNUAL_CANDIDATE_BUDGET,
            "min_selected_candidates": DEFAULT_MIN_SELECTED_CANDIDATES,
            "min_match_recall": DEFAULT_MIN_MATCH_RECALL,
        }
        for name, expected in frozen.items():
            if values[name] != expected:
                raise ValueError(
                    f"{MA20_EPISODE_TRAINING_VERSION} {name} is frozen at "
                    f"{expected!r}"
                )
