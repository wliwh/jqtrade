"""Candidate-level threshold selection aligned to episode match outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np


DEFAULT_ANNUAL_CANDIDATE_BUDGET = 6
DEFAULT_MIN_SELECTED_CANDIDATES = 6
DEFAULT_MIN_MATCH_RECALL = 0.60
WILSON_Z_95 = 1.96


@dataclass(frozen=True)
class EpisodeFilterThreshold:
    threshold: float
    status: str
    selected_candidates: int
    selected_matches: int
    precision: float
    match_recall: float
    precision_wilson_lower: float
    candidate_budget: int


def select_episode_filter_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    calibration_year_count: int,
    annual_candidate_budget: int = DEFAULT_ANNUAL_CANDIDATE_BUDGET,
    min_selected_candidates: int = DEFAULT_MIN_SELECTED_CANDIDATES,
    min_match_recall: float = DEFAULT_MIN_MATCH_RECALL,
) -> EpisodeFilterThreshold:
    """Maximize a conservative precision bound under recall/support constraints."""

    predictions = np.asarray(probability, dtype=float).reshape(-1)
    labels = np.asarray(target, dtype=int).reshape(-1)
    if not len(predictions) or len(predictions) != len(labels):
        raise ValueError("probability and target must be non-empty and aligned")
    if not np.isfinite(predictions).all() or (
        (predictions < 0.0) | (predictions > 1.0)
    ).any():
        raise ValueError("probability must be finite and between zero and one")
    if not np.isin(labels, [0, 1]).all() or labels.sum() == 0:
        raise ValueError("target must contain zero/one and at least one match")
    for value, name in (
        (calibration_year_count, "calibration_year_count"),
        (annual_candidate_budget, "annual_candidate_budget"),
        (min_selected_candidates, "min_selected_candidates"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    recall_floor = float(min_match_recall)
    if not isfinite(recall_floor) or not 0.0 < recall_floor <= 1.0:
        raise ValueError("min_match_recall must be between zero and one")

    budget = annual_candidate_budget * calibration_year_count
    candidates = np.unique(np.concatenate([predictions, np.array([0.0])]))[::-1]
    best: EpisodeFilterThreshold | None = None
    for threshold in candidates:
        selected = predictions >= threshold
        selected_count = int(selected.sum())
        if selected_count < min_selected_candidates or selected_count > budget:
            continue
        selected_matches = int(labels[selected].sum())
        recall = selected_matches / int(labels.sum())
        if recall + 1e-12 < recall_floor:
            continue
        precision = selected_matches / selected_count
        lower = _wilson_lower(selected_matches, selected_count)
        candidate = EpisodeFilterThreshold(
            threshold=float(threshold),
            status="selected",
            selected_candidates=selected_count,
            selected_matches=selected_matches,
            precision=precision,
            match_recall=recall,
            precision_wilson_lower=lower,
            candidate_budget=budget,
        )
        if best is None or _selection_key(candidate) > _selection_key(best):
            best = candidate
    if best is not None:
        return best

    selected_matches = int(labels.sum())
    selected_count = len(labels)
    return EpisodeFilterThreshold(
        threshold=0.0,
        status="passthrough_no_feasible_threshold",
        selected_candidates=selected_count,
        selected_matches=selected_matches,
        precision=selected_matches / selected_count,
        match_recall=1.0,
        precision_wilson_lower=_wilson_lower(selected_matches, selected_count),
        candidate_budget=budget,
    )


def _selection_key(value: EpisodeFilterThreshold) -> tuple[float, ...]:
    return (
        value.precision_wilson_lower,
        value.precision,
        value.match_recall,
        value.threshold,
    )


def _wilson_lower(successes: int, trials: int) -> float:
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    z2 = WILSON_Z_95**2
    denominator = 1.0 + z2 / trials
    center = proportion + z2 / (2.0 * trials)
    spread = WILSON_Z_95 * sqrt(
        proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials**2)
    )
    return (center - spread) / denominator
