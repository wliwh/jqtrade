"""Causal all-A limit-up versus limit-down breadth reversal signals."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "limit_up_down_breadth_v1_20120705_20260814"
REQUESTED_START_DATE = pd.Timestamp("2012-01-01")
HISTORY_WINDOW = 250
MIN_HISTORY = 120
CHANGE_LOOKBACK = 5
EXTREME_THRESHOLD = 0.75
CHANGE_THRESHOLD = 0.10
SIGNAL_SPECS = (
    ("limit_up_down_breadth_top", "top"),
    ("limit_up_down_breadth_bottom", "bottom"),
)

_UP_DOWN_COUNT_COLUMNS = (
    "limit_up_hit_count",
    "limit_down_hit_count",
    "limit_up_close_count",
    "limit_down_close_count",
)
_NET_COUNT_COLUMNS = (
    "limit_hit_net_count",
    "limit_close_net_count",
)
_UP_DOWN_RATIO_COLUMNS = (
    "limit_up_hit_ratio",
    "limit_down_hit_ratio",
    "limit_up_close_ratio",
    "limit_down_close_ratio",
)
_NET_RATIO_COLUMNS = (
    "limit_hit_net_ratio",
    "limit_close_net_ratio",
)


def build_limit_up_down_breadth_signals(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build symmetric reversal signals from hit and close net-limit breadth.

    Each component is ranked against valid observations within the prior 250
    trading days. The current observation is strictly excluded, and no rank is
    emitted with fewer than 120 valid observations in that window.
    """

    market = _validate_daily_features(daily_features, start_date=start_date)
    hit_rank, hit_history_count = _causal_historical_midrank(
        market["limit_hit_net_ratio"],
        history_window=HISTORY_WINDOW,
        min_history=MIN_HISTORY,
    )
    close_rank, close_history_count = _causal_historical_midrank(
        market["limit_close_net_ratio"],
        history_window=HISTORY_WINDOW,
        min_history=MIN_HISTORY,
    )
    market["limit_hit_net_rank250"] = hit_rank
    market["limit_close_net_rank250"] = close_rank
    market["limit_hit_rank_history_count"] = hit_history_count
    market["limit_close_rank_history_count"] = close_history_count
    market["limit_rank_history_count"] = np.minimum(
        hit_history_count,
        close_history_count,
    )
    market["limit_score"] = market[
        ["limit_hit_net_rank250", "limit_close_net_rank250"]
    ].mean(axis=1, skipna=False)
    market["score_available"] = market["limit_score"].notna()

    available_positions = np.flatnonzero(
        market["score_available"].to_numpy(dtype=bool)
    )
    if not len(available_positions):
        raise ValueError(
            "daily_features has no limit score with at least "
            f"{MIN_HISTORY} prior valid observations"
        )
    source_start_date = market["date"].iloc[0]
    market = market.iloc[available_positions[0] :].reset_index(drop=True)
    market["limit_score_change_5d"] = market["limit_score"].diff(
        CHANGE_LOOKBACK
    )
    market["change_available"] = market["limit_score_change_5d"].notna()
    market["signal_available"] = (
        market["quality_available"]
        & market["score_available"]
        & market["change_available"]
    )
    market["limit_hit_total_ratio"] = (
        market["limit_up_hit_ratio"] + market["limit_down_hit_ratio"]
    )
    market["limit_close_total_ratio"] = (
        market["limit_up_close_ratio"] + market["limit_down_close_ratio"]
    )

    top_triggered = (
        market["signal_available"]
        & market["limit_score"].ge(EXTREME_THRESHOLD)
        & market["limit_score_change_5d"].le(-CHANGE_THRESHOLD)
    )
    bottom_triggered = (
        market["signal_available"]
        & market["limit_score"].le(1.0 - EXTREME_THRESHOLD)
        & market["limit_score_change_5d"].ge(CHANGE_THRESHOLD)
    )
    triggers = {
        "limit_up_down_breadth_top": top_triggered,
        "limit_up_down_breadth_bottom": bottom_triggered,
    }

    audit_columns = (
        *_UP_DOWN_COUNT_COLUMNS,
        *_NET_COUNT_COLUMNS,
        *_UP_DOWN_RATIO_COLUMNS,
        *_NET_RATIO_COLUMNS,
        "valid_count_limit",
        "limit_price_missing_count",
        "limit_hit_total_ratio",
        "limit_close_total_ratio",
    )
    shared: dict[str, object] = {
        "date": market["date"],
        "raw_value": market["limit_score"],
        "universe_size": market["universe_size"],
        "valid_count": market["valid_count_limit"],
        "version": version,
        "quality_available": market["quality_available"],
        "score_available": market["score_available"],
        "change_available": market["change_available"],
        "signal_available": market["signal_available"],
        "limit_hit_net_rank250": market["limit_hit_net_rank250"],
        "limit_close_net_rank250": market["limit_close_net_rank250"],
        "limit_hit_rank_history_count": market[
            "limit_hit_rank_history_count"
        ],
        "limit_close_rank_history_count": market[
            "limit_close_rank_history_count"
        ],
        "limit_rank_history_count": market["limit_rank_history_count"],
        "limit_score": market["limit_score"],
        "limit_score_change_5d": market["limit_score_change_5d"],
    }
    for column in audit_columns:
        shared[column] = market[column]

    signal_frames = []
    for signal_id, direction in SIGNAL_SPECS:
        frame = pd.DataFrame(shared)
        frame.insert(1, "signal_id", signal_id)
        frame.insert(2, "direction", direction)
        frame.insert(4, "triggered", triggers[signal_id])
        signal_frames.append(frame)

    source = pd.concat(signal_frames, ignore_index=True)
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    episode_counts = episodes.groupby("direction").size().astype(int).to_dict()
    trigger_counts = (
        daily.groupby("direction")["triggered"].sum().astype(int).to_dict()
    )
    directions = {direction for _, direction in SIGNAL_SPECS}
    episodes_by_direction = {
        direction: int(episode_counts.get(direction, 0))
        for direction in sorted(directions)
    }
    triggered_by_direction = {
        direction: int(trigger_counts.get(direction, 0))
        for direction in sorted(directions)
    }
    metadata = {
        "signal_version": version,
        "requested_start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "source_start_date": source_start_date.strftime("%Y-%m-%d"),
        "comparison_start_date": market["date"].min().strftime("%Y-%m-%d"),
        "comparison_end_date": market["date"].max().strftime("%Y-%m-%d"),
        "trade_dates": len(market),
        "daily_rows": len(daily),
        "signal_series": len(SIGNAL_SPECS),
        "quality_available_dates": int(market["quality_available"].sum()),
        "score_available_dates": int(market["score_available"].sum()),
        "change_available_dates": int(market["change_available"].sum()),
        "triggered_days_by_direction": triggered_by_direction,
        "episodes_by_direction": episodes_by_direction,
        "score_available_from": market.loc[
            market["score_available"], "date"
        ].iloc[0].strftime("%Y-%m-%d"),
        "change_available_from": _first_available_date(
            market,
            "change_available",
        ),
    }
    return daily, episodes, metadata


def _causal_historical_midrank(
    values: pd.Series,
    *,
    history_window: int,
    min_history: int,
) -> tuple[pd.Series, pd.Series]:
    """Return midranks against valid values in a prior-row trading-day window."""

    if history_window <= 0:
        raise ValueError("history_window must be positive")
    if min_history <= 0 or min_history > history_window:
        raise ValueError("min_history must be in [1, history_window]")

    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    history: deque[float] = deque(maxlen=history_window)
    ranks = np.full(len(numeric), np.nan, dtype=float)
    history_counts = np.zeros(len(numeric), dtype=int)
    for position, current in enumerate(numeric):
        prior_window = np.fromiter(history, dtype=float, count=len(history))
        prior = prior_window[np.isfinite(prior_window)]
        history_counts[position] = len(prior)
        if np.isfinite(current) and len(prior) >= min_history:
            less = int(np.count_nonzero(prior < current))
            equal = int(np.count_nonzero(prior == current))
            ranks[position] = (less + 0.5 * equal) / len(prior)
        history.append(float(current))
    return (
        pd.Series(ranks, index=values.index, dtype=float),
        pd.Series(history_counts, index=values.index, dtype=int),
    )


def _validate_daily_features(
    frame: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("daily_features must be a pandas DataFrame")
    required = {
        "date",
        "universe_size",
        "valid_count_limit",
        "limit_price_missing_count",
        *_UP_DOWN_COUNT_COLUMNS,
        *_NET_COUNT_COLUMNS,
        *_UP_DOWN_RATIO_COLUMNS,
        *_NET_RATIO_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily_features is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("daily_features must not be empty")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily_features contains an invalid date")
    result = result[result["date"].ge(pd.Timestamp(start_date))].copy()
    if result.empty:
        raise ValueError("daily_features has no rows on or after start_date")
    if result["date"].duplicated().any():
        raise ValueError("daily_features contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("daily_features dates must be strictly increasing")

    nonnegative_integer_columns = (
        "universe_size",
        "valid_count_limit",
        "limit_price_missing_count",
        *_UP_DOWN_COUNT_COLUMNS,
    )
    for column in nonnegative_integer_columns:
        result[column] = _validated_integer_column(
            result[column],
            name=column,
            allow_negative=False,
        )
    for column in _NET_COUNT_COLUMNS:
        result[column] = _validated_integer_column(
            result[column],
            name=column,
            allow_negative=True,
        )
    if result["universe_size"].eq(0).any():
        raise ValueError("universe_size must be positive")
    if result["valid_count_limit"].gt(result["universe_size"]).any():
        raise ValueError("valid_count_limit must not exceed universe_size")
    if (
        result["valid_count_limit"] + result["limit_price_missing_count"]
    ).gt(result["universe_size"]).any():
        raise ValueError(
            "valid_count_limit + limit_price_missing_count must not exceed "
            "universe_size"
        )
    for column in _UP_DOWN_COUNT_COLUMNS:
        if result[column].gt(result["valid_count_limit"]).any():
            raise ValueError(f"{column} must not exceed valid_count_limit")
    if result["limit_up_close_count"].gt(result["limit_up_hit_count"]).any():
        raise ValueError("limit_up_close_count must not exceed limit_up_hit_count")
    if result["limit_down_close_count"].gt(
        result["limit_down_hit_count"]
    ).any():
        raise ValueError(
            "limit_down_close_count must not exceed limit_down_hit_count"
        )

    _validate_limit_side(
        result,
        side="hit",
        up_count="limit_up_hit_count",
        down_count="limit_down_hit_count",
        net_count="limit_hit_net_count",
        up_ratio="limit_up_hit_ratio",
        down_ratio="limit_down_hit_ratio",
        net_ratio="limit_hit_net_ratio",
    )
    _validate_limit_side(
        result,
        side="close",
        up_count="limit_up_close_count",
        down_count="limit_down_close_count",
        net_count="limit_close_net_count",
        up_ratio="limit_up_close_ratio",
        down_ratio="limit_down_close_ratio",
        net_ratio="limit_close_net_ratio",
    )
    result["quality_available"] = result["valid_count_limit"].gt(0)
    return result.reset_index(drop=True)


def _validated_integer_column(
    values: pd.Series,
    *,
    name: str,
    allow_negative: bool,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = (
        numeric.isna()
        | ~np.isfinite(numeric)
        | np.not_equal(numeric, np.floor(numeric))
    )
    if not allow_negative:
        invalid |= numeric.lt(0)
    if invalid.any():
        qualifier = "integers" if allow_negative else "non-negative integers"
        raise ValueError(f"{name} must contain {qualifier}")
    return numeric.astype(int)


def _validate_limit_side(
    result: pd.DataFrame,
    *,
    side: str,
    up_count: str,
    down_count: str,
    net_count: str,
    up_ratio: str,
    down_ratio: str,
    net_ratio: str,
) -> None:
    expected_net_count = result[up_count] - result[down_count]
    if not result[net_count].equals(expected_net_count):
        raise ValueError(f"{net_count} does not match {up_count} - {down_count}")

    valid = result["valid_count_limit"].gt(0)
    missing = ~valid
    ratio_specs = (
        (up_ratio, result[up_count], 0.0),
        (down_ratio, result[down_count], 0.0),
        (net_ratio, expected_net_count, -1.0),
    )
    for ratio_column, numerator, lower_bound in ratio_specs:
        observed = pd.to_numeric(result[ratio_column], errors="coerce")
        coercion_failed = result[ratio_column].notna() & observed.isna()
        if coercion_failed.any():
            raise ValueError(f"{ratio_column} contains a non-numeric value")
        if observed[missing].notna().any():
            raise ValueError(
                f"{ratio_column} must be missing when valid_count_limit is zero"
            )
        valid_values = observed[valid]
        if (
            valid_values.isna().any()
            or not np.isfinite(valid_values.to_numpy(dtype=float)).all()
            or valid_values.lt(lower_bound).any()
            or valid_values.gt(1.0).any()
        ):
            raise ValueError(
                f"{ratio_column} must contain finite values in range when "
                "valid_count_limit is positive"
            )
        expected = numerator[valid].divide(result.loc[valid, "valid_count_limit"])
        if not np.allclose(valid_values, expected, rtol=1e-9, atol=1e-12):
            raise ValueError(f"{ratio_column} does not match its counts")
        result[ratio_column] = observed.astype(float)

    if side not in {"hit", "close"}:
        raise AssertionError(f"unexpected limit side: {side}")


def _first_available_date(frame: pd.DataFrame, column: str) -> str | None:
    dates = frame.loc[frame[column], "date"]
    if dates.empty:
        return None
    return dates.iloc[0].strftime("%Y-%m-%d")
