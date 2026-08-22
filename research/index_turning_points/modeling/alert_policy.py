"""Causal probability alert policy with hysteresis and cooldown."""

from __future__ import annotations

from math import isfinite

import numpy as np


DEFAULT_MIN_ENTRY_PROBABILITY = 0.50
DEFAULT_EXIT_PROBABILITY = 0.30
DEFAULT_COOLDOWN_DAYS = 10


def apply_hysteresis_cooldown(
    probability: np.ndarray,
    *,
    entry_threshold: float = DEFAULT_MIN_ENTRY_PROBABILITY,
    exit_threshold: float = DEFAULT_EXIT_PROBABILITY,
    cooldown_days: int = DEFAULT_COOLDOWN_DAYS,
) -> np.ndarray:
    """Return causal active days under one entry/exit/cooldown state machine."""

    values = np.asarray(probability, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("probability must be non-empty and finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probability must be between zero and one")
    entry = float(entry_threshold)
    exit_value = float(exit_threshold)
    if (
        not isfinite(entry)
        or not isfinite(exit_value)
        or not 0.0 <= exit_value < entry <= 1.0
    ):
        raise ValueError("thresholds must satisfy 0 <= exit < entry <= 1")
    if (
        isinstance(cooldown_days, bool)
        or not isinstance(cooldown_days, int)
        or cooldown_days < 0
    ):
        raise ValueError("cooldown_days must be a non-negative integer")

    result = np.zeros(len(values), dtype=bool)
    active = False
    cooldown_remaining = 0
    for position, value in enumerate(values):
        if active:
            if value >= exit_value:
                result[position] = True
            else:
                active = False
                cooldown_remaining = cooldown_days
            continue
        if cooldown_remaining:
            cooldown_remaining -= 1
            continue
        if value >= entry:
            active = True
            result[position] = True
    return result
