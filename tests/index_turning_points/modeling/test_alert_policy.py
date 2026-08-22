import numpy as np

from research.index_turning_points.modeling.alert_policy import (
    apply_hysteresis_cooldown,
)


def test_hysteresis_and_cooldown_require_exit_then_delay_reentry():
    probability = np.array([0.49, 0.50, 0.40, 0.30, 0.29, 0.80, 0.80, 0.80])

    triggered = apply_hysteresis_cooldown(
        probability,
        entry_threshold=0.50,
        exit_threshold=0.30,
        cooldown_days=2,
    )

    assert triggered.tolist() == [False, True, True, True, False, False, False, True]
