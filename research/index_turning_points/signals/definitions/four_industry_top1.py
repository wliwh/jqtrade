"""Causal four-industry MA20 Top1 historical baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_ID = "four_industry_top1"
SIGNAL_VERSION = "four_industry_top1_v2_20211213_20260814"
DIRECTION = "top"
TARGET_IDS = ("bank", "coal", "nonferrous", "steel")
TARGET_NAMES = {
    "bank": "银行",
    "coal": "煤炭",
    "nonferrous": "有色金属",
    "steel": "钢铁",
}
MIN_INDUSTRY_VALID_COUNT = 5


def build_four_industry_top1_signal(
    daily_features: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build the frozen four-industry signal from point-in-time daily fields.

    Each target becomes comparable on its own first valid MA20 rank date. The
    cohort starts at the latest of those dates and requires continuous target
    comparability thereafter. No predecessor or substitute industry is used.
    """

    daily = _validate_daily(daily_features)
    start_dates: dict[str, pd.Timestamp] = {}
    comparable: dict[str, pd.Series] = {}
    for target_id in TARGET_IDS:
        prefix = f"target_{target_id}_"
        target_comparable = (
            daily[prefix + "mapping_count"].eq(1)
            & daily[prefix + "valid_count_ma20"].ge(MIN_INDUSTRY_VALID_COUNT)
            & daily[prefix + "breadth_ma20"].notna()
            & daily[prefix + "rank_ma20"].notna()
        )
        if not target_comparable.any():
            raise ValueError(f"target industry never becomes comparable: {target_id}")
        start_dates[target_id] = pd.Timestamp(
            daily.loc[target_comparable, "date"].iloc[0]
        )
        comparable[target_id] = target_comparable

    comparison_start = max(start_dates.values())
    selected = daily[daily["date"].ge(comparison_start)].copy().reset_index(drop=True)
    for target_id in TARGET_IDS:
        target_comparable = comparable[target_id].loc[
            daily["date"].ge(comparison_start)
        ].reset_index(drop=True)
        if not target_comparable.all():
            missing_dates = selected.loc[~target_comparable, "date"]
            raise ValueError(
                f"target comparability is not continuous after cohort start: "
                f"{target_id} at "
                f"{missing_dates.dt.strftime('%Y-%m-%d').tolist()[:5]}"
            )

    target_flags = pd.DataFrame(
        {
            target_id: _strict_bool(
                selected[f"target_{target_id}_is_top1_ma20"],
                f"target_{target_id}_is_top1_ma20",
            )
            for target_id in TARGET_IDS
        }
    )
    for target_id in TARGET_IDS:
        expected = selected[f"target_{target_id}_rank_ma20"].eq(1.0)
        if not target_flags[target_id].equals(expected):
            raise ValueError(f"stored Top1 flag does not match rank: {target_id}")
    triggered = target_flags.any(axis=1)
    stored_trigger = _strict_bool(
        selected["four_industry_top1_triggered"],
        "four_industry_top1_triggered",
    )
    if not triggered.equals(stored_trigger):
        raise ValueError("four_industry_top1_triggered does not match target ranks")
    expected_ids = target_flags.apply(
        lambda row: "|".join(
            sorted(target_id for target_id in TARGET_IDS if bool(row[target_id]))
        ),
        axis=1,
    )
    stored_ids = selected["four_industry_top1_ids"].fillna("").astype(str)
    if not expected_ids.equals(stored_ids):
        raise ValueError("four_industry_top1_ids does not match target ranks")

    breadth_columns = [f"target_{target_id}_breadth_ma20" for target_id in TARGET_IDS]
    valid_count_columns = [
        f"target_{target_id}_valid_count_ma20" for target_id in TARGET_IDS
    ]
    signal_source = pd.DataFrame(
        {
            "date": selected["date"],
            "signal_id": SIGNAL_ID,
            "direction": DIRECTION,
            "raw_value": selected[breadth_columns].max(axis=1),
            "triggered": triggered,
            "universe_size": len(TARGET_IDS),
            "valid_count": selected[valid_count_columns].notna().sum(axis=1),
            "version": version,
            "comparison_start_date": comparison_start,
            "market_breadth_ma20": selected["breadth_ma20"],
            "ranked_industry_count_ma20": selected["ranked_industry_count_ma20"],
            "top1_tie_count_ma20": selected["top1_tie_count_ma20"],
            "top1_industry_codes_ma20": selected["top1_industry_codes_ma20"],
            "top1_industry_names_ma20": selected["top1_industry_names_ma20"],
            "four_industry_top1_ids": stored_ids,
        }
    )
    for target_id in TARGET_IDS:
        for suffix in (
            "industry_code",
            "industry_name",
            "valid_count_ma20",
            "breadth_ma20",
            "rank_ma20",
            "is_top1_ma20",
        ):
            signal_source[f"target_{target_id}_{suffix}"] = selected[
                f"target_{target_id}_{suffix}"
            ].to_numpy()

    if not np.isfinite(signal_source["raw_value"].to_numpy(dtype=float)).all():
        raise ValueError("four-industry raw_value contains non-finite values")
    event_daily, episodes = build_signal_events(
        signal_source,
        capped_confirmation_n=2,
    )
    metadata = {
        "signal_id": SIGNAL_ID,
        "signal_version": version,
        "direction": DIRECTION,
        "target_ids": list(TARGET_IDS),
        "target_names": TARGET_NAMES,
        "target_start_dates": {
            target_id: start_dates[target_id].strftime("%Y-%m-%d")
            for target_id in TARGET_IDS
        },
        "comparison_start_date": comparison_start.strftime("%Y-%m-%d"),
        "comparison_end_date": selected["date"].iloc[-1].strftime("%Y-%m-%d"),
        "daily_rows": len(event_daily),
        "triggered_days": int(event_daily["triggered"].sum()),
        "episodes": len(episodes),
    }
    return event_daily, episodes, metadata


def _validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("daily_features must be a pandas DataFrame")
    required = {
        "date",
        "breadth_ma20",
        "ranked_industry_count_ma20",
        "top1_tie_count_ma20",
        "top1_industry_codes_ma20",
        "top1_industry_names_ma20",
        "four_industry_top1_triggered",
        "four_industry_top1_ids",
    }
    for target_id in TARGET_IDS:
        required.update(
            f"target_{target_id}_{suffix}"
            for suffix in (
                "mapping_count",
                "industry_code",
                "industry_name",
                "valid_count_ma20",
                "breadth_ma20",
                "rank_ma20",
                "is_top1_ma20",
            )
        )
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily_features is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("daily_features must not be empty")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily_features contains an invalid date")
    if result["date"].duplicated().any():
        raise ValueError("daily_features contains duplicate dates")
    result = result.sort_values("date").reset_index(drop=True)
    for target_id in TARGET_IDS:
        prefix = f"target_{target_id}_"
        for suffix in (
            "mapping_count",
            "valid_count_ma20",
            "breadth_ma20",
            "rank_ma20",
        ):
            result[prefix + suffix] = pd.to_numeric(
                result[prefix + suffix], errors="coerce"
            )
    return result


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only booleans")
    return normalized.map(mapping).astype(bool)
