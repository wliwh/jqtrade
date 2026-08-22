"""Validation-only threshold selection under a fixed episode budget."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_ANNUAL_EPISODE_BUDGET = 6
DEFAULT_MAX_ALERT_ACTIVE_DAYS = 2


@dataclass(frozen=True)
class AlertThreshold:
    threshold: float
    episode_count: int
    active_days: int


def count_contiguous_episodes(triggered: np.ndarray) -> int:
    """Count false-to-true transitions in one ordered boolean series."""

    active = np.asarray(triggered, dtype=bool).reshape(-1)
    if not len(active):
        return 0
    return int(active[0]) + int(np.count_nonzero(active[1:] & ~active[:-1]))


def limit_alert_duration(
    triggered: np.ndarray,
    *,
    max_active_days: int = DEFAULT_MAX_ALERT_ACTIVE_DAYS,
) -> np.ndarray:
    """Keep only the first N days of each causal threshold episode.

    A suppressed episode cannot re-arm until the underlying threshold condition
    becomes false. This avoids turning a persistent high-score regime into a
    months-long active signal while preserving the original onset date.
    """

    if isinstance(max_active_days, bool) or not isinstance(max_active_days, int):
        raise ValueError("max_active_days must be a positive integer")
    if max_active_days <= 0:
        raise ValueError("max_active_days must be a positive integer")
    active = np.asarray(triggered, dtype=bool).reshape(-1)
    result = np.zeros(len(active), dtype=bool)
    run_length = 0
    for position, value in enumerate(active):
        if not value:
            run_length = 0
            continue
        run_length += 1
        result[position] = run_length <= max_active_days
    return result


def select_episode_budget_threshold(
    scores: np.ndarray,
    *,
    max_episodes: int = DEFAULT_ANNUAL_EPISODE_BUDGET,
) -> AlertThreshold:
    """Select the most episode-rich threshold within the validation budget.

    Unique score thresholds are traversed high to low. The selected threshold
    maximizes the number of separate episodes without exceeding the budget; a
    tie keeps the higher, more selective threshold. Labels are never consulted.
    """

    if isinstance(max_episodes, bool) or not isinstance(max_episodes, int):
        raise ValueError("max_episodes must be a positive integer")
    if max_episodes <= 0:
        raise ValueError("max_episodes must be a positive integer")
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be non-empty and finite")

    candidates = np.unique(values)[::-1]
    no_alert_threshold = float(np.nextafter(candidates[0], np.inf))
    best = AlertThreshold(no_alert_threshold, 0, 0)
    for threshold in candidates:
        triggered = values >= threshold
        episode_count = count_contiguous_episodes(triggered)
        if episode_count <= max_episodes and episode_count > best.episode_count:
            best = AlertThreshold(
                threshold=float(threshold),
                episode_count=episode_count,
                active_days=int(triggered.sum()),
            )
            if episode_count == max_episodes:
                break
    return best
