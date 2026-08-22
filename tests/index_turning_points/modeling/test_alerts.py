import numpy as np
import pytest

from research.index_turning_points.modeling.alerts import (
    count_contiguous_episodes,
    limit_alert_duration,
    select_episode_budget_threshold,
)


def test_threshold_uses_highest_level_that_reaches_episode_budget():
    scores = np.array([0, 9, 0, 8, 0, 7, 0, 6, 0], dtype=float)

    selected = select_episode_budget_threshold(scores, max_episodes=3)

    assert selected.threshold == 7
    assert selected.episode_count == 3
    assert selected.active_days == 3


def test_episode_counter_handles_leading_and_trailing_active_runs():
    triggered = np.array([True, True, False, True, False, False, True])

    assert count_contiguous_episodes(triggered) == 3


def test_alert_duration_limit_requires_a_raw_false_before_rearming():
    triggered = np.array(
        [False, True, True, True, False, True, True, True, True, False]
    )

    limited = limit_alert_duration(triggered, max_active_days=2)

    assert limited.tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert count_contiguous_episodes(limited) == 2


def test_alert_budget_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        select_episode_budget_threshold(np.array([1.0]), max_episodes=0)
    with pytest.raises(ValueError, match="positive"):
        limit_alert_duration(np.array([True]), max_active_days=0)
