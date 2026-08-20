"""Post-event OHLC outcomes and inference for causal signal events."""

from __future__ import annotations

from math import erfc, sqrt

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (5, 10, 20)
EVENT_FLAGS = {
    "onset": "event_onset",
    "capped_confirmation": "event_capped_confirmation",
}
SERIES_COLUMNS = ("signal_id", "direction", "version")
OUTCOME_NAMES = ("terminal_return", "max_up", "max_down")
OUTCOME_COLUMNS = (
    "signal_id",
    "direction",
    "version",
    "event_kind",
    "episode_id",
    "event_date",
    "index_id",
    "index_name",
    "horizon",
    "event_date_available",
    "available_future_bars",
    "complete_window",
    "window_end_date",
    "close",
    "terminal_return",
    "max_up",
    "max_down",
)
METRIC_COLUMNS = (
    "signal_id",
    "direction",
    "version",
    "event_kind",
    "index_id",
    "index_name",
    "horizon",
    "outcome_name",
    "event_count",
    "event_mean",
    "event_median",
    "event_q25",
    "event_q75",
    "baseline_count",
    "baseline_mean",
    "mean_difference",
    "hac_lag",
    "hac_standard_error",
    "ci95_lower",
    "ci95_upper",
    "hac_p_value",
    "local_fdr_q_value",
    "global_fdr_q_value",
    "inference_eligible",
)


def build_forward_event_outcomes(
    signal_daily: pd.DataFrame,
    ohlc: pd.DataFrame,
    *,
    event_kinds: tuple[str, ...] = tuple(EVENT_FLAGS),
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Calculate future OHLC paths for every event/index/horizon combination.

    The close on the event date is the reference price. Future extrema use the
    next ``horizon`` index bars, excluding the event bar. Missing index dates
    and incomplete tail windows remain explicit rows rather than disappearing.
    """

    kinds = _validate_event_kinds(event_kinds)
    valid_horizons = _validate_horizons(horizons)
    signal = _validate_signal_daily(signal_daily, kinds)
    prices = _validate_ohlc(ohlc)
    records: list[dict[str, object]] = []

    price_groups = {
        str(index_id): group.sort_values("date").reset_index(drop=True)
        for index_id, group in prices.groupby("index_id", sort=True)
    }
    for key, series in signal.groupby(list(SERIES_COLUMNS), sort=True):
        identity = dict(zip(SERIES_COLUMNS, (str(value) for value in key)))
        for event_kind in kinds:
            flag = EVENT_FLAGS[event_kind]
            events = series.loc[series[flag], ["episode_id", "date"]]
            for event in events.itertuples(index=False):
                event_date = pd.Timestamp(event.date)
                for index_id, index_prices in price_groups.items():
                    index_name = str(index_prices["index_name"].iloc[0])
                    positions = index_prices.index[
                        index_prices["date"].eq(event_date)
                    ].tolist()
                    position = positions[0] if positions else None
                    for horizon in valid_horizons:
                        records.append(
                            {
                                **identity,
                                "event_kind": event_kind,
                                "episode_id": str(event.episode_id),
                                "event_date": event_date,
                                "index_id": index_id,
                                "index_name": index_name,
                                **_event_outcome(index_prices, position, horizon),
                            }
                        )

    result = pd.DataFrame(records, columns=OUTCOME_COLUMNS)
    if result.empty:
        return result
    result["event_date"] = pd.to_datetime(result["event_date"])
    result["window_end_date"] = pd.to_datetime(result["window_end_date"])
    return result.sort_values(
        [
            *SERIES_COLUMNS,
            "event_kind",
            "event_date",
            "episode_id",
            "index_id",
            "horizon",
        ]
    ).reset_index(drop=True)


def summarize_forward_event_outcomes(
    event_outcomes: pd.DataFrame,
    signal_daily: pd.DataFrame,
    ohlc: pd.DataFrame,
    *,
    min_event_count: int = 20,
    min_baseline_count: int = 30,
) -> pd.DataFrame:
    """Compare complete event outcomes with non-event dates in signal coverage.

    Inference uses an intercept/event-indicator OLS with Newey-West covariance
    and a lag equal to the outcome horizon. Local FDR families are each signal
    series and event kind; global FDR covers every eligible test in the table.
    """

    outcomes = _validate_event_outcomes(event_outcomes)
    signal = _validate_signal_daily(signal_daily, tuple(EVENT_FLAGS))
    prices = _validate_ohlc(ohlc)
    _validate_minimum(min_event_count, "min_event_count")
    _validate_minimum(min_baseline_count, "min_baseline_count")
    if outcomes.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    price_groups = {
        str(index_id): group.sort_values("date").reset_index(drop=True)
        for index_id, group in prices.groupby("index_id", sort=True)
    }
    records: list[dict[str, object]] = []
    group_columns = [
        *SERIES_COLUMNS,
        "event_kind",
        "index_id",
        "index_name",
        "horizon",
    ]
    for key, group in outcomes.groupby(group_columns, sort=True):
        identity = dict(zip(group_columns, key))
        horizon = int(identity["horizon"])
        index_id = str(identity["index_id"])
        series_mask = np.logical_and.reduce(
            [
                signal[column].astype(str).eq(str(identity[column]))
                for column in SERIES_COLUMNS
            ]
        )
        coverage = signal.loc[series_mask, "date"]
        if coverage.empty:
            raise ValueError("event outcomes reference a missing signal series")
        index_prices = price_groups.get(index_id)
        if index_prices is None:
            raise ValueError(f"event outcomes reference a missing index: {index_id}")

        daily = _daily_outcomes(
            index_prices,
            pd.DatetimeIndex(coverage.sort_values().unique()),
            horizon,
        )
        event_dates = set(pd.to_datetime(group["event_date"]))
        daily["is_event"] = daily["date"].isin(event_dates)
        complete_events = group[group["complete_window"]].copy()
        if complete_events.duplicated("event_date").any():
            raise ValueError("event outcomes contain duplicate event dates in a group")

        for outcome_name in OUTCOME_NAMES:
            event_values = pd.to_numeric(
                complete_events[outcome_name], errors="coerce"
            ).dropna()
            baseline_values = pd.to_numeric(
                daily.loc[~daily["is_event"], outcome_name], errors="coerce"
            ).dropna()
            eligible = (
                len(event_values) >= min_event_count
                and len(baseline_values) >= min_baseline_count
            )
            inference = {
                "mean_difference": (
                    event_values.mean() - baseline_values.mean()
                    if len(event_values) and len(baseline_values)
                    else np.nan
                ),
                "hac_standard_error": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
                "hac_p_value": np.nan,
            }
            if eligible:
                inference = _newey_west_event_effect(
                    daily[["date", "is_event", outcome_name]].dropna(
                        subset=[outcome_name]
                    ),
                    outcome_name=outcome_name,
                    lag=horizon,
                )
            records.append(
                {
                    **identity,
                    "outcome_name": outcome_name,
                    "event_count": len(event_values),
                    "event_mean": event_values.mean(),
                    "event_median": event_values.median(),
                    "event_q25": event_values.quantile(0.25),
                    "event_q75": event_values.quantile(0.75),
                    "baseline_count": len(baseline_values),
                    "baseline_mean": baseline_values.mean(),
                    **inference,
                    "hac_lag": horizon,
                    "local_fdr_q_value": np.nan,
                    "global_fdr_q_value": np.nan,
                    "inference_eligible": eligible,
                }
            )

    result = pd.DataFrame(records, columns=METRIC_COLUMNS)
    local_keys = [*SERIES_COLUMNS, "event_kind"]
    for _, positions in result.groupby(local_keys, sort=True).groups.items():
        positions = list(positions)
        result.loc[positions, "local_fdr_q_value"] = _benjamini_hochberg(
            result.loc[positions, "hac_p_value"]
        )
    result["global_fdr_q_value"] = _benjamini_hochberg(result["hac_p_value"])
    return result.sort_values(
        [*SERIES_COLUMNS, "event_kind", "index_id", "horizon", "outcome_name"]
    ).reset_index(drop=True)


def _event_outcome(
    prices: pd.DataFrame,
    position: int | None,
    horizon: int,
) -> dict[str, object]:
    if position is None:
        return {
            "horizon": horizon,
            "event_date_available": False,
            "available_future_bars": 0,
            "complete_window": False,
            "window_end_date": pd.NaT,
            "close": np.nan,
            "terminal_return": np.nan,
            "max_up": np.nan,
            "max_down": np.nan,
        }
    close = float(prices.iloc[position]["close"])
    future = prices.iloc[position + 1 : position + horizon + 1]
    complete = len(future) == horizon
    return {
        "horizon": horizon,
        "event_date_available": True,
        "available_future_bars": len(future),
        "complete_window": complete,
        "window_end_date": future.iloc[-1]["date"] if complete else pd.NaT,
        "close": close,
        "terminal_return": (
            float(future.iloc[-1]["close"]) / close - 1.0 if complete else np.nan
        ),
        "max_up": (
            float(future["high"].max()) / close - 1.0 if complete else np.nan
        ),
        "max_down": (
            float(future["low"].min()) / close - 1.0 if complete else np.nan
        ),
    }


def _daily_outcomes(
    prices: pd.DataFrame,
    coverage_dates: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    records = []
    positions = {
        pd.Timestamp(date): position for position, date in enumerate(prices["date"])
    }
    for date in coverage_dates:
        position = positions.get(pd.Timestamp(date))
        outcome = _event_outcome(prices, position, horizon)
        records.append(
            {
                "date": pd.Timestamp(date),
                "terminal_return": outcome["terminal_return"],
                "max_up": outcome["max_up"],
                "max_down": outcome["max_down"],
            }
        )
    return pd.DataFrame(records)


def _newey_west_event_effect(
    frame: pd.DataFrame,
    *,
    outcome_name: str,
    lag: int,
) -> dict[str, float]:
    ordered = frame.sort_values("date")
    y = ordered[outcome_name].to_numpy(dtype=float)
    event = ordered["is_event"].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(ordered)), event])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta
    bread = np.linalg.inv(x.T @ x)
    xu = x * residual[:, None]
    meat = xu.T @ xu
    max_lag = min(int(lag), len(y) - 1)
    for offset in range(1, max_lag + 1):
        weight = 1.0 - offset / (max_lag + 1.0)
        cross = xu[offset:].T @ xu[:-offset]
        meat += weight * (cross + cross.T)
    covariance = bread @ meat @ bread
    covariance *= len(y) / (len(y) - x.shape[1])
    variance = max(float(covariance[1, 1]), 0.0)
    standard_error = sqrt(variance)
    effect = float(beta[1])
    if standard_error == 0.0:
        p_value = 0.0 if effect != 0.0 else 1.0
    else:
        p_value = erfc(abs(effect / standard_error) / sqrt(2.0))
    return {
        "mean_difference": effect,
        "hac_standard_error": standard_error,
        "ci95_lower": effect - 1.96 * standard_error,
        "ci95_upper": effect + 1.96 * standard_error,
        "hac_p_value": p_value,
    }


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = pd.to_numeric(values, errors="coerce").dropna().sort_values()
    count = len(valid)
    if not count:
        return result
    adjusted = valid.to_numpy(dtype=float) * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = np.clip(adjusted, 0.0, 1.0)
    return result


def _validate_event_kinds(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("event_kinds must be a non-empty tuple")
    if len(set(values)) != len(values) or any(value not in EVENT_FLAGS for value in values):
        raise ValueError(f"event_kinds must be unique members of {sorted(EVENT_FLAGS)}")
    return values


def _validate_horizons(values: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("horizons must be a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("horizons must contain positive integers")
    if values != tuple(sorted(set(values))):
        raise ValueError("horizons must be unique and increasing")
    return values


def _validate_minimum(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_signal_daily(
    signal_daily: pd.DataFrame,
    event_kinds: tuple[str, ...],
) -> pd.DataFrame:
    if not isinstance(signal_daily, pd.DataFrame):
        raise TypeError("signal_daily must be a pandas DataFrame")
    required = {*SERIES_COLUMNS, "date", "episode_id"}
    required.update(EVENT_FLAGS[kind] for kind in event_kinds)
    missing = required.difference(signal_daily.columns)
    if missing:
        raise ValueError(f"signal_daily is missing columns: {sorted(missing)}")
    if signal_daily.empty:
        raise ValueError("signal_daily must not be empty")
    result = signal_daily.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("signal_daily contains an invalid date")
    if result.duplicated([*SERIES_COLUMNS, "date"]).any():
        raise ValueError("signal_daily contains duplicate series dates")
    if not result["direction"].isin(["top", "bottom"]).all():
        raise ValueError("signal direction must be top or bottom")
    for kind in event_kinds:
        flag = EVENT_FLAGS[kind]
        result[flag] = _strict_bool(result[flag], flag)
        selected = result[result[flag]]
        if selected["episode_id"].isna().any() or selected["episode_id"].astype(str).eq("").any():
            raise ValueError(f"{flag} rows must have an episode_id")
        if selected.duplicated([*SERIES_COLUMNS, "episode_id"]).any():
            raise ValueError(f"{flag} must occur at most once per episode")
    return result.sort_values([*SERIES_COLUMNS, "date"]).reset_index(drop=True)


def _validate_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(ohlc, pd.DataFrame):
        raise TypeError("ohlc must be a pandas DataFrame")
    required = {"index_id", "index_name", "date", "high", "low", "close"}
    missing = required.difference(ohlc.columns)
    if missing:
        raise ValueError(f"ohlc is missing columns: {sorted(missing)}")
    if ohlc.empty:
        raise ValueError("ohlc must not be empty")
    result = ohlc.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("ohlc contains an invalid date")
    if result.duplicated(["index_id", "date"]).any():
        raise ValueError("ohlc contains duplicate index dates")
    names_per_index = result.groupby("index_id")["index_name"].nunique()
    if names_per_index.gt(1).any():
        raise ValueError("ohlc contains inconsistent index names")
    numeric = result[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    if (
        not np.isfinite(numeric.to_numpy()).all()
        or not numeric.gt(0).all().all()
        or not numeric["high"].ge(numeric["close"]).all()
        or not numeric["low"].le(numeric["close"]).all()
        or not numeric["high"].ge(numeric["low"]).all()
    ):
        raise ValueError("ohlc contains invalid prices")
    result[["high", "low", "close"]] = numeric
    return result.sort_values(["index_id", "date"]).reset_index(drop=True)


def _validate_event_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("event_outcomes must be a pandas DataFrame")
    missing = set(OUTCOME_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"event_outcomes is missing columns: {sorted(missing)}")
    result = frame.copy()
    if result.empty:
        return result
    result["event_date"] = pd.to_datetime(result["event_date"], errors="coerce")
    if result["event_date"].isna().any():
        raise ValueError("event_outcomes contains an invalid event_date")
    result["complete_window"] = _strict_bool(
        result["complete_window"], "complete_window"
    )
    duplicate_key = [
        *SERIES_COLUMNS,
        "event_kind",
        "episode_id",
        "index_id",
        "horizon",
    ]
    if result.duplicated(duplicate_key).any():
        raise ValueError("event_outcomes contains duplicate event records")
    return result


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    mapping = {"true": True, "false": False}
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only booleans")
    return normalized.map(mapping).astype(bool)
