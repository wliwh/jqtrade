"""Causal all-A turnover-heat top signal."""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "turnover_heat_v1_20120705_20260814"
SIGNAL_ID = "all_a_turnover_heat_top"
REQUESTED_START_DATE = pd.Timestamp("2012-01-01")
HISTORY_WINDOW = 250
MIN_HISTORY = 120
CHANGE_LOOKBACK = 5
SCORE_THRESHOLD = 0.75
CHANGE_THRESHOLD = 0.10
RANK_COMPONENTS = (
    "turnover_ratio_pct_p50",
    "turnover_ratio_pct_cap_weighted_mean",
    "turnover_ge_10pct_ratio",
)
AUDIT_SUMMARY_COLUMNS = (
    "turnover_ratio_pct_mean",
    "turnover_ratio_pct_p25",
    "turnover_ratio_pct_p50",
    "turnover_ratio_pct_p75",
    "turnover_ratio_pct_p90",
    "turnover_ratio_pct_p95",
)
EXTREME_THRESHOLDS = (5, 10, 20)


def build_turnover_heat_signal(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the frozen top signal from three causal turnover ranks."""

    market = _validate_daily_features(daily_features, start_date=start_date)
    history_start_date = market["date"].iloc[0]
    history_input_trade_dates = len(market)
    rank_columns: list[str] = []
    history_count_columns: list[str] = []
    rank_available_from: dict[str, str | None] = {}

    for component in RANK_COMPONENTS:
        ranked = causal_historical_midrank(market[component])
        rank_column = f"{component}_rank250"
        history_count_column = f"{component}_rank250_history_count"
        market[rank_column] = ranked["rank"]
        market[history_count_column] = ranked["history_count"]
        rank_columns.append(rank_column)
        history_count_columns.append(history_count_column)
        available = market.loc[market[rank_column].notna(), "date"]
        rank_available_from[component] = (
            available.iloc[0].strftime("%Y-%m-%d")
            if not available.empty
            else None
        )

    market["turnover_score"] = market[rank_columns].mean(
        axis=1,
        skipna=False,
    )
    market["quality_available"] = market["turnover_score"].notna()
    if not market["quality_available"].any():
        raise ValueError(
            "daily_features has no date with sufficient turnover history"
        )

    market["turnover_score_change_5d"] = market["turnover_score"].diff(
        CHANGE_LOOKBACK
    )
    market["change_available"] = (
        market["quality_available"]
        & market["turnover_score"].shift(CHANGE_LOOKBACK).notna()
    )
    triggered = (
        market["quality_available"]
        & market["change_available"]
        & market["turnover_score"].ge(SCORE_THRESHOLD)
        & market["turnover_score_change_5d"].le(-CHANGE_THRESHOLD)
    )

    first_score_position = int(
        np.flatnonzero(market["quality_available"].to_numpy())[0]
    )
    market = market.iloc[first_score_position:].copy().reset_index(drop=True)
    triggered = triggered.iloc[first_score_position:].reset_index(drop=True)

    output_columns = [
        "date",
        "universe_size",
        "turnover_valid_count",
        "turnover_cap_weight_valid_count",
        *AUDIT_SUMMARY_COLUMNS,
    ]
    for threshold in EXTREME_THRESHOLDS:
        output_columns.extend(
            (
                f"turnover_ge_{threshold}pct_count",
                f"turnover_ge_{threshold}pct_ratio",
            )
        )
    output_columns.extend(rank_columns)
    output_columns.extend(history_count_columns)
    output_columns.extend(
        (
            "turnover_score",
            "turnover_score_change_5d",
            "quality_available",
            "change_available",
        )
    )

    source = market[output_columns].copy()
    source.insert(1, "signal_id", SIGNAL_ID)
    source.insert(2, "direction", "top")
    source.insert(3, "raw_value", market["turnover_score"])
    source.insert(4, "triggered", triggered.astype(bool))
    source.insert(
        6,
        "valid_count",
        market[
            ["turnover_valid_count", "turnover_cap_weight_valid_count"]
        ].min(axis=1),
    )
    source.insert(7, "version", version)

    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    change_dates = market.loc[market["change_available"], "date"]
    metadata = {
        "signal_version": version,
        "requested_start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "history_start_date": _format_date(history_start_date),
        "comparison_start_date": market["date"].iloc[0].strftime("%Y-%m-%d"),
        "comparison_end_date": market["date"].iloc[-1].strftime("%Y-%m-%d"),
        "first_score_available_date": market["date"].iloc[0].strftime(
            "%Y-%m-%d"
        ),
        "first_change_available_date": (
            change_dates.iloc[0].strftime("%Y-%m-%d")
            if not change_dates.empty
            else None
        ),
        "rank_available_from_by_component": rank_available_from,
        "history_input_trade_dates": history_input_trade_dates,
        "trade_dates": len(market),
        "daily_rows": len(daily),
        "quality_available_dates": int(market["quality_available"].sum()),
        "quality_unavailable_dates": int((~market["quality_available"]).sum()),
        "change_available_dates": int(market["change_available"].sum()),
        "triggered_days": int(daily["triggered"].sum()),
        "episodes": len(episodes),
    }
    return daily, episodes, metadata


def causal_historical_midrank(
    values: pd.Series,
    *,
    window: int = HISTORY_WINDOW,
    min_history: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Rank against valid values in the prior trading-day window."""

    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer")
    if (
        isinstance(min_history, bool)
        or not isinstance(min_history, int)
        or min_history <= 0
        or min_history > window
    ):
        raise ValueError("min_history must be a positive integer not above window")

    numeric = pd.to_numeric(pd.Series(values, copy=False), errors="coerce")
    numeric = numeric.where(np.isfinite(numeric), np.nan)
    history: deque[float] = deque(maxlen=window)
    ranks = np.full(len(numeric), np.nan, dtype=float)
    history_counts = np.zeros(len(numeric), dtype=int)

    for position, current in enumerate(numeric.to_numpy(dtype=float)):
        previous = np.fromiter(history, dtype=float, count=len(history))
        previous = previous[np.isfinite(previous)]
        history_counts[position] = len(previous)
        if np.isfinite(current) and len(previous) >= min_history:
            less = int(np.count_nonzero(previous < current))
            equal = int(np.count_nonzero(previous == current))
            ranks[position] = (less + 0.5 * equal) / len(previous)
        history.append(float(current))

    return pd.DataFrame(
        {"rank": ranks, "history_count": history_counts},
        index=numeric.index,
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
        "turnover_valid_count",
        "turnover_cap_weight_valid_count",
        *AUDIT_SUMMARY_COLUMNS,
    }
    for threshold in EXTREME_THRESHOLDS:
        required.update(
            {
                f"turnover_ge_{threshold}pct_count",
                f"turnover_ge_{threshold}pct_ratio",
            }
        )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily_features is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("daily_features must not be empty")

    start = _validate_start_date(start_date)
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily_features contains an invalid date")
    result = result[result["date"].ge(start)].copy()
    if result.empty:
        raise ValueError("daily_features has no rows on or after start_date")
    if result["date"].duplicated().any():
        raise ValueError("daily_features contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("daily_features dates must be strictly increasing")

    count_columns = [
        "universe_size",
        "turnover_valid_count",
        "turnover_cap_weight_valid_count",
        *(f"turnover_ge_{threshold}pct_count" for threshold in EXTREME_THRESHOLDS),
    ]
    for column in count_columns:
        values = pd.to_numeric(result[column], errors="coerce")
        if (
            values.isna().any()
            or values.lt(0).any()
            or np.not_equal(values, np.floor(values)).any()
        ):
            raise ValueError(f"{column} must contain non-negative integers")
        result[column] = values.astype(int)
    if result["universe_size"].eq(0).any():
        raise ValueError("universe_size must be positive")
    if result["turnover_valid_count"].gt(result["universe_size"]).any():
        raise ValueError("turnover_valid_count must not exceed universe_size")
    if result["turnover_cap_weight_valid_count"].gt(
        result["turnover_valid_count"]
    ).any():
        raise ValueError(
            "turnover_cap_weight_valid_count must not exceed turnover_valid_count"
        )

    summary = {}
    for column in AUDIT_SUMMARY_COLUMNS:
        summary[column] = _validated_optional_nonnegative(result[column], column)
        result[column] = summary[column]
    turnover_available = result["turnover_valid_count"].gt(0)
    for column in AUDIT_SUMMARY_COLUMNS:
        if summary[column].notna().ne(turnover_available).any():
            raise ValueError(
                f"{column} availability must match turnover_valid_count"
            )

    quantiles = result[
        [
            "turnover_ratio_pct_p25",
            "turnover_ratio_pct_p50",
            "turnover_ratio_pct_p75",
            "turnover_ratio_pct_p90",
            "turnover_ratio_pct_p95",
        ]
    ]
    if (
        quantiles.diff(axis=1).iloc[:, 1:].lt(0).any(axis=None)
    ):
        raise ValueError("turnover quantiles must be nondecreasing")

    weighted_column = "turnover_ratio_pct_cap_weighted_mean"
    weighted = _validated_optional_nonnegative(
        result[weighted_column], weighted_column
    )
    weighted_available = result["turnover_cap_weight_valid_count"].gt(0)
    if weighted.notna().ne(weighted_available).any():
        raise ValueError(
            "turnover_ratio_pct_cap_weighted_mean availability must match "
            "turnover_cap_weight_valid_count"
        )
    result[weighted_column] = weighted

    counts = []
    ratios = []
    for threshold in EXTREME_THRESHOLDS:
        count_column = f"turnover_ge_{threshold}pct_count"
        ratio_column = f"turnover_ge_{threshold}pct_ratio"
        if result[count_column].gt(result["turnover_valid_count"]).any():
            raise ValueError(
                f"{count_column} must not exceed turnover_valid_count"
            )
        ratio = _validated_optional_nonnegative(result[ratio_column], ratio_column)
        if ratio.gt(1.0).any():
            raise ValueError(f"{ratio_column} must not exceed one")
        expected = result[count_column].divide(
            result["turnover_valid_count"].replace(0, np.nan)
        )
        if not np.allclose(
            ratio.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"{ratio_column} does not match {count_column}/turnover_valid_count"
            )
        result[ratio_column] = ratio
        counts.append(count_column)
        ratios.append(ratio_column)

    if result[counts].diff(axis=1).iloc[:, 1:].gt(0).any(axis=None):
        raise ValueError("turnover extreme counts must be nonincreasing")
    if result[ratios].diff(axis=1).iloc[:, 1:].gt(0).any(axis=None):
        raise ValueError("turnover extreme ratios must be nonincreasing")
    return result.reset_index(drop=True)


def _validated_optional_nonnegative(values: pd.Series, name: str) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce")
    invalid_text = values.notna() & converted.isna()
    finite_values = converted.dropna().to_numpy(dtype=float)
    if (
        invalid_text.any()
        or not np.isfinite(finite_values).all()
        or converted.dropna().lt(0).any()
    ):
        raise ValueError(f"{name} must contain non-negative finite values or missing")
    return converted.astype(float)


def _validate_start_date(value: str | pd.Timestamp) -> pd.Timestamp:
    try:
        result = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("start_date must be a valid date") from error
    if pd.isna(result):
        raise ValueError("start_date must be a valid date")
    return result


def _format_date(value: pd.Timestamp) -> str:
    return value.strftime("%Y-%m-%d")
