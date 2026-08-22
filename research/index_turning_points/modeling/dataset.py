"""Assemble the frozen first-pass all-A ML training table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import (
    MA_WINDOWS,
    RANGE_WINDOWS,
    RETURN_WINDOWS,
    VOLATILITY_WINDOWS,
    build_index_features,
    point_in_time_directional_state,
)
from .targets import (
    DEFAULT_HORIZONS,
    add_future_entry_targets,
    build_lobe_intensity_targets,
)


DATASET_VERSION = "all_a_ml_dataset_v1"
TODAY_DATASET_VERSION = "all_a_ml_today_dataset_v1"
FUTURE_ENTRY_TARGET_MODE = "future_entry"
TODAY_TARGET_MODE = "today_strict_lobe_membership"
MODEL_START_DATE = pd.Timestamp("2012-07-05")
MARKET_LEVEL_FEATURES = (
    "breadth_ma20",
    "breadth_ma60",
    "breadth_ma120",
    "new_high_low_net_ratio_60",
    "new_high_low_net_ratio_120",
    "new_high_low_net_ratio_250",
    "limit_hit_net_ratio",
    "limit_close_net_ratio",
    "turnover_ratio_pct_p50",
    "turnover_ratio_pct_cap_weighted_mean",
    "turnover_ge_10pct_ratio",
)
MARKET_CHANGE_WINDOWS = (1, 5, 10)
COVERAGE_FEATURES = {
    "market_base_coverage": "base_valid_count",
    "market_ma20_coverage": "valid_count_ma20",
    "market_ma60_coverage": "valid_count_ma60",
    "market_ma120_coverage": "valid_count_ma120",
    "market_high_low_60_coverage": "valid_count_high_low_60",
    "market_high_low_120_coverage": "valid_count_high_low_120",
    "market_high_low_250_coverage": "valid_count_high_low_250",
    "market_limit_coverage": "valid_count_limit",
    "market_turnover_coverage": "turnover_valid_count",
}
TODAY_FEATURE_COLUMNS = (
    "breadth_ma20",
    "breadth_ma20_change_5d",
    "breadth_ma60",
    "breadth_ma60_change_10d",
    "new_high_low_net_ratio_60",
    "new_high_low_net_ratio_60_change_5d",
    "limit_hit_net_ratio",
    "limit_hit_net_ratio_change_5d",
    "turnover_ratio_pct_p50",
    "turnover_ratio_pct_p50_change_10d",
    "index_close_to_ma60",
    "index_drawdown_60d",
    "index_rebound_60d",
    "index_return_5d",
    "index_volatility_20d",
)


def build_all_a_training_daily(
    market_features: pd.DataFrame,
    index_daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    *,
    threshold: float = 0.10,
    start_date: str | pd.Timestamp = MODEL_START_DATE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Merge causal features and post-hoc targets on the all-A calendar."""

    return _build_all_a_daily(
        market_features,
        index_daily,
        regions,
        lobes,
        threshold=threshold,
        start_date=start_date,
        horizons=horizons,
    )


def build_all_a_today_training_daily(
    market_features: pd.DataFrame,
    index_daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    *,
    threshold: float = 0.10,
    start_date: str | pd.Timestamp = MODEL_START_DATE,
) -> pd.DataFrame:
    """Build the compact current-day strict-lobe probability dataset.

    The frozen strict-lobe membership columns are the binary probability
    targets. Post-hoc intensity is retained only for diagnosis and evaluation;
    future-entry targets and unused V1 features are omitted from the bundle.
    """

    full = _build_all_a_daily(
        market_features,
        index_daily,
        regions,
        lobes,
        threshold=threshold,
        start_date=start_date,
        horizons=None,
    )
    columns = (
        "date",
        *today_feature_columns(),
        "index_price_available",
        "target_available",
        "truth_top_intensity",
        "truth_top_in_strict_lobe",
        "truth_bottom_intensity",
        "truth_bottom_in_strict_lobe",
    )
    return full.loc[:, columns].copy()


def _build_all_a_daily(
    market_features: pd.DataFrame,
    index_daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    *,
    threshold: float,
    start_date: str | pd.Timestamp,
    horizons: tuple[int, ...] | None,
) -> pd.DataFrame:
    """Assemble shared point-in-time context with one requested target family."""

    market = _build_market_features(market_features)
    index_features = build_index_features(index_daily)
    phase = point_in_time_directional_state(index_daily, threshold=threshold)
    intensity = build_lobe_intensity_targets(
        index_daily, regions, lobes, index_id="all_a"
    )
    targets = (
        add_future_entry_targets(intensity, horizons=horizons)
        if horizons is not None
        else intensity
    )
    index_context = index_features.merge(
        phase, on="date", how="inner", validate="one_to_one"
    ).merge(targets, on="date", how="inner", validate="one_to_one")

    result = market.merge(
        index_context, on="date", how="left", validate="one_to_one"
    )
    result["index_price_available"] = (
        result["index_price_available"].astype("boolean").fillna(False).astype(bool)
    )
    result["target_available"] = result["truth_top_intensity"].notna()
    start = pd.Timestamp(start_date)
    result = result[result["date"].ge(start)].copy()
    if result.empty:
        raise ValueError("training table has no rows on or after start_date")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("training table dates must be increasing")
    return result.reset_index(drop=True)


def feature_columns() -> tuple[str, ...]:
    """Return the frozen V1 model feature names in deterministic order."""

    market_changes = tuple(
        f"{column}_change_{window}d"
        for column in MARKET_LEVEL_FEATURES
        for window in MARKET_CHANGE_WINDOWS
    )
    index_returns = tuple(f"index_return_{window}d" for window in RETURN_WINDOWS)
    index_ma = tuple(f"index_close_to_ma{window}" for window in MA_WINDOWS)
    index_ranges = tuple(
        name
        for window in RANGE_WINDOWS
        for name in (
            f"index_drawdown_{window}d",
            f"index_rebound_{window}d",
        )
    )
    index_volatility = tuple(
        f"index_volatility_{window}d" for window in VOLATILITY_WINDOWS
    )
    return (
        *MARKET_LEVEL_FEATURES,
        *market_changes,
        "market_universe_size",
        *COVERAGE_FEATURES,
        "index_price_available",
        *index_returns,
        "index_range_1d",
        *index_ma,
        *index_ranges,
        *index_volatility,
        "index_phase_pending",
        "index_phase_up",
        "index_phase_down",
    )


def today_feature_columns() -> tuple[str, ...]:
    """Return the frozen compact current-day model feature names."""

    return TODAY_FEATURE_COLUMNS


def _build_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("market_features must be a pandas DataFrame")
    required = {
        "date",
        "universe_size",
        *MARKET_LEVEL_FEATURES,
        *COVERAGE_FEATURES.values(),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"market_features is missing columns: {sorted(missing)}")
    source = frame.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    if source["date"].isna().any():
        raise ValueError("market_features contains an invalid date")
    if source.empty or source["date"].duplicated().any():
        raise ValueError("market_features dates must be non-empty and unique")
    source = source.sort_values("date").reset_index(drop=True)

    result = pd.DataFrame({"date": source["date"]})
    for column in MARKET_LEVEL_FEATURES:
        values = pd.to_numeric(source[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float, na_value=np.nan)).all():
            raise ValueError(f"market feature must be finite: {column}")
        result[column] = values
        for window in MARKET_CHANGE_WINDOWS:
            result[f"{column}_change_{window}d"] = values - values.shift(window)

    universe = pd.to_numeric(source["universe_size"], errors="coerce")
    if universe.isna().any() or (universe <= 0).any():
        raise ValueError("universe_size must be positive and finite")
    result["market_universe_size"] = universe
    for output_column, source_column in COVERAGE_FEATURES.items():
        valid = pd.to_numeric(source[source_column], errors="coerce")
        ratio = valid / universe
        if ratio.isna().any() or ((ratio < 0.0) | (ratio > 1.0)).any():
            raise ValueError(f"invalid coverage ratio from {source_column}")
        result[output_column] = ratio
    return result
