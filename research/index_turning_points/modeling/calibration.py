"""Probability calibration fallbacks and fixed-bin reliability metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


@dataclass
class PriorShiftCalibrator:
    """Shift raw logits to one target prevalence while preserving ranking."""

    intercept_shift: float | None = None

    def fit(
        self,
        raw_probability: np.ndarray,
        *,
        target_prevalence: float,
    ) -> "PriorShiftCalibrator":
        raw = _finite_probability(raw_probability)
        prevalence = float(target_prevalence)
        if not isfinite(prevalence) or not 0.0 < prevalence < 1.0:
            raise ValueError("target_prevalence must be finite and between zero and one")
        logits = _logit(raw)
        lower, upper = -50.0, 50.0
        for _ in range(200):
            midpoint = (lower + upper) / 2.0
            mean = float(_sigmoid(logits + midpoint).mean())
            if mean < prevalence:
                lower = midpoint
            else:
                upper = midpoint
        self.intercept_shift = (lower + upper) / 2.0
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        if self.intercept_shift is None:
            raise RuntimeError("prior-shift calibrator must be fitted before predict")
        raw = _finite_probability(raw_probability)
        return _sigmoid(_logit(raw) + self.intercept_shift)


def calibration_reliability(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    bin_count: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Return fixed-width reliability rows and sample-weighted ECE."""

    if isinstance(bin_count, bool) or not isinstance(bin_count, int) or bin_count <= 1:
        raise ValueError("bin_count must be an integer greater than one")
    labels = np.asarray(target, dtype=int).reshape(-1)
    predictions = _finite_probability(probability)
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError("target and probability must be non-empty and aligned")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("target must contain only zero and one")

    bin_numbers = np.minimum(
        np.floor(predictions * bin_count).astype(int), bin_count - 1
    )
    records: list[dict[str, object]] = []
    weighted_gap = 0.0
    for bin_number in range(bin_count):
        selected = bin_numbers == bin_number
        rows = int(selected.sum())
        if not rows:
            continue
        mean_probability = float(predictions[selected].mean())
        observed_rate = float(labels[selected].mean())
        absolute_gap = abs(mean_probability - observed_rate)
        weighted_gap += rows * absolute_gap
        records.append(
            {
                "bin_number": bin_number,
                "bin_lower": bin_number / bin_count,
                "bin_upper": (bin_number + 1) / bin_count,
                "rows": rows,
                "mean_predicted_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": absolute_gap,
            }
        )
    return weighted_gap / len(labels), pd.DataFrame(records)


def _finite_probability(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if not len(result) or not np.isfinite(result).all():
        raise ValueError("probability must be non-empty and finite")
    return np.clip(result, 1e-6, 1.0 - 1e-6)


def _logit(values: np.ndarray) -> np.ndarray:
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))
