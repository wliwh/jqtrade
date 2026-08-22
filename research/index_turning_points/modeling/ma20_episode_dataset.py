"""Build one-row-per-MA20-candidate training data with post-hoc labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..evaluation.region_matching import match_signal_regions
from .episode_targets import (
    OPERATIONAL_WINDOW_TRADE_DAYS,
    build_operational_episode_labels,
)


MA20_EPISODE_DATASET_VERSION = "all_a_ma20_episode_dataset_v1"
MA20_SIGNAL_IDS = {
    "top": "ma20_breadth_reversal_top",
    "bottom": "ma20_breadth_reversal_bottom",
}
MAX_CANDIDATE_GAP_TRADE_DAYS = 252
MA20_EPISODE_FEATURE_COLUMNS = (
    "breadth_ma20",
    "breadth_ma20_change_5d",
    "breadth_ma60",
    "breadth_ma60_change_10d",
    "new_high_low_net_ratio_60",
    "new_high_low_net_ratio_60_change_5d",
    "limit_hit_net_ratio",
    "turnover_ratio_pct_p50",
    "index_close_to_ma60",
    "index_drawdown_60d",
    "index_rebound_60d",
    "index_volatility_20d",
    "candidate_gap_trade_days",
)


@dataclass(frozen=True)
class Ma20EpisodeDatasetResult:
    candidate_episodes: pd.DataFrame
    daily_calendar: pd.DataFrame


def build_ma20_episode_dataset(
    signal_daily: pd.DataFrame,
    signal_episodes: pd.DataFrame,
    feature_daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
) -> Ma20EpisodeDatasetResult:
    """Join causal onset features to operational and legacy region labels."""

    features = _validate_feature_daily(feature_daily)
    calendar = pd.DatetimeIndex(features["date"])
    episodes = _candidate_episodes(signal_episodes, calendar)
    candidates = episodes.rename(
        columns={"episode_id": "candidate_episode_id"}
    ).copy()
    candidates["candidate_year"] = candidates["onset_date"].dt.year
    candidates["candidate_gap_trade_days"] = _candidate_gaps(
        candidates, calendar
    )

    feature_columns = [
        column
        for column in MA20_EPISODE_FEATURE_COLUMNS
        if column != "candidate_gap_trade_days"
    ]
    candidates = candidates.merge(
        features[["date", *feature_columns]],
        left_on="onset_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    if candidates[feature_columns].isna().all(axis=1).any():
        raise ValueError("a candidate onset has no matching feature row")

    operational = build_operational_episode_labels(
        candidates[["candidate_episode_id", "direction", "onset_date"]],
        regions,
        lobes,
        calendar,
        window_trade_days=OPERATIONAL_WINDOW_TRADE_DAYS,
    )
    candidates = candidates.merge(
        operational,
        on="candidate_episode_id",
        how="left",
        validate="one_to_one",
    )
    legacy = _legacy_labels(
        signal_daily,
        candidates,
        regions,
        lobes,
        calendar,
    )
    candidates = candidates.merge(
        legacy,
        on="candidate_episode_id",
        how="left",
        validate="one_to_one",
    )
    candidates = candidates.sort_values(
        ["onset_date", "direction", "candidate_episode_id"]
    ).reset_index(drop=True)

    daily_calendar = pd.DataFrame(
        {
            "date": calendar,
            "universe_size": 1,
            "valid_count": 1,
        }
    )
    return Ma20EpisodeDatasetResult(candidates, daily_calendar)


def ma20_episode_feature_columns() -> tuple[str, ...]:
    return MA20_EPISODE_FEATURE_COLUMNS


def _candidate_episodes(
    frame: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    required = {
        "episode_id",
        "signal_id",
        "direction",
        "onset_date",
        "last_active_date",
        "active_days",
        "status",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"signal_episodes is missing columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("onset_date", "last_active_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if result[["onset_date", "last_active_date"]].isna().any().any():
        raise ValueError("signal_episodes contains invalid dates")
    expected_ids = set(MA20_SIGNAL_IDS.values())
    result = result[
        result["signal_id"].astype(str).isin(expected_ids)
        & result["onset_date"].between(calendar.min(), calendar.max())
    ].copy()
    if result.empty:
        raise ValueError("signal_episodes contains no MA20 candidates in coverage")
    expected_by_direction = result["direction"].map(MA20_SIGNAL_IDS)
    if not result["signal_id"].astype(str).eq(expected_by_direction).all():
        raise ValueError("MA20 signal_id and direction are inconsistent")
    if result["episode_id"].astype(str).duplicated().any():
        raise ValueError("candidate episode_id must be unique")
    return result[
        [
            "episode_id",
            "signal_id",
            "direction",
            "onset_date",
            "last_active_date",
            "active_days",
            "status",
        ]
    ].rename(
        columns={
            "signal_id": "source_signal_id",
            "last_active_date": "source_last_active_date",
            "active_days": "source_active_days",
            "status": "source_episode_status",
        }
    )


def _candidate_gaps(
    candidates: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.Series:
    positions = {pd.Timestamp(date): value for value, date in enumerate(calendar)}
    result = pd.Series(index=candidates.index, dtype=float)
    for _direction, group in candidates.groupby("direction", sort=True):
        ordered = group.sort_values("onset_date")
        values = np.asarray(
            [positions[pd.Timestamp(value)] for value in ordered["onset_date"]],
            dtype=int,
        )
        gaps = np.diff(values, prepend=values[0] - MAX_CANDIDATE_GAP_TRADE_DAYS)
        gaps = np.clip(gaps, 1, MAX_CANDIDATE_GAP_TRADE_DAYS)
        result.loc[ordered.index] = gaps
    return result.astype(int)


def _legacy_labels(
    signal_daily: pd.DataFrame,
    candidates: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    calendar: pd.DatetimeIndex,
) -> pd.DataFrame:
    source = signal_daily.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source = source[
        source["signal_id"].astype(str).isin(MA20_SIGNAL_IDS.values())
        & source["date"].isin(calendar)
    ].copy()
    calendars = pd.DataFrame(
        {"index_id": "all_a", "index_name": "全A", "date": calendar}
    )
    matches = match_signal_regions(
        source,
        regions,
        lobes,
        calendars,
        event_kind="onset",
    )
    episode_rows = matches[matches["episode_id"].astype(str).ne("")].copy()
    episode_rows = episode_rows[
        episode_rows["episode_id"].astype(str).isin(
            candidates["candidate_episode_id"].astype(str)
        )
    ]
    if episode_rows["episode_id"].astype(str).duplicated().any():
        raise ValueError("legacy matching returned duplicate candidate rows")
    primary = episode_rows["match_status"].eq("matched")
    loose = _strict_bool(episode_rows["loose_matched"])
    strict = _strict_bool(episode_rows["strict_matched"])
    legacy = pd.DataFrame(
        {
            "candidate_episode_id": episode_rows["episode_id"].astype(str),
            "legacy_match_status": episode_rows["match_status"].astype(str),
            "target_legacy_window_20d_match": primary,
            "target_legacy_loose_match": primary & loose,
            "target_legacy_strict_match": primary & strict,
            "legacy_region_id": episode_rows["region_id"].fillna("").astype(str),
            "legacy_lead_lag_days": pd.to_numeric(
                episode_rows["lead_lag_days"], errors="coerce"
            ).astype("Int64"),
        }
    )
    missing_ids = set(candidates["candidate_episode_id"]).difference(
        legacy["candidate_episode_id"]
    )
    if missing_ids:
        raise ValueError(f"legacy matching omitted candidates: {sorted(missing_ids)}")
    return legacy


def _validate_feature_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        *(
            column
            for column in MA20_EPISODE_FEATURE_COLUMNS
            if column != "candidate_gap_trade_days"
        ),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"feature_daily is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("feature_daily must not be empty")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("feature_daily contains an invalid date")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("feature_daily dates must be unique and increasing")
    for column in required.difference({"date"}):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _strict_bool(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("legacy match flags must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError("legacy match flags must contain only booleans")
    return normalized.map(mapping).astype(bool)
