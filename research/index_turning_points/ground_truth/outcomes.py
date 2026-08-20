"""Post-hoc forward outcomes for signal evaluation."""

from __future__ import annotations

import pandas as pd


HORIZONS = (5, 10, 20, 60)


def forward_outcomes(
    close: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Calculate full-window future downside, upside and terminal return."""

    result = pd.DataFrame({"close": close.astype(float)})
    for horizon in horizons:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizons must contain positive integers")

        future = pd.concat(
            [close.shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        complete = future.notna().all(axis=1)
        returns = future.divide(close, axis=0).sub(1.0)
        result[f"future_max_down_{horizon}d"] = returns.min(axis=1).where(complete)
        result[f"future_max_up_{horizon}d"] = returns.max(axis=1).where(complete)
        result[f"future_return_{horizon}d"] = (
            close.shift(-horizon).divide(close).sub(1.0).where(complete)
        )

    return result
