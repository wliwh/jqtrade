"""Causal point-in-time SW level-1 industry MA20 Top1 signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "single_industry_top1_v1_20170103_20260814"
DIRECTION = "top"
REQUESTED_START_DATE = pd.Timestamp("2017-01-01")
MIN_INDUSTRY_VALID_COUNT = 5


def build_single_industry_top1_signals(
    industry_breadth: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build one independent Top1 series for every point-in-time industry.

    An industry exists only between its first and last observed dates. Historical
    and successor classifications are never joined, and missing outside-lifespan
    dates are not converted to inactive observations.
    """

    industry = _validate_industry_breadth(industry_breadth, start_date=start_date)
    calendar = pd.DatetimeIndex(industry["date"].drop_duplicates().sort_values())
    industry["ranked_industry_count_ma20"] = industry.groupby("date")[
        "rank_eligible_ma20"
    ].transform("sum")
    industry["top1_tie_count_ma20"] = industry.groupby("date")[
        "is_top1_ma20"
    ].transform("sum")

    signal_frames: list[pd.DataFrame] = []
    coverage_records: list[dict[str, object]] = []
    for (industry_code, industry_name), group in industry.groupby(
        ["industry_code", "industry_name"], sort=True
    ):
        group = group.sort_values("date").reset_index(drop=True)
        _validate_continuous_lifespan(
            group,
            calendar,
            industry_code=industry_code,
            industry_name=industry_name,
        )
        signal_id = f"single_industry_top1_{industry_code}"
        signal_frames.append(
            pd.DataFrame(
                {
                    "date": group["date"],
                    "signal_id": signal_id,
                    "direction": DIRECTION,
                    "raw_value": group["breadth_ma20"],
                    "triggered": group["is_top1_ma20"],
                    "universe_size": group["universe_count"],
                    "valid_count": group["valid_count_ma20"],
                    "version": version,
                    "industry_code": industry_code,
                    "industry_name": industry_name,
                    "industry_rank_ma20": group["rank_ma20"],
                    "industry_breadth_ma20": group["breadth_ma20"],
                    "industry_above_count_ma20": group["above_count_ma20"],
                    "industry_valid_count_ma20": group["valid_count_ma20"],
                    "ranked_industry_count_ma20": group[
                        "ranked_industry_count_ma20"
                    ],
                    "top1_tie_count_ma20": group["top1_tie_count_ma20"],
                    "industry_start_date": group["date"].iloc[0],
                    "industry_end_date": group["date"].iloc[-1],
                }
            )
        )
        coverage_records.append(
            {
                "signal_id": signal_id,
                "industry_code": industry_code,
                "industry_name": industry_name,
                "start_date": group["date"].iloc[0].strftime("%Y-%m-%d"),
                "end_date": group["date"].iloc[-1].strftime("%Y-%m-%d"),
                "daily_rows": len(group),
                "top1_days": int(group["is_top1_ma20"].sum()),
            }
        )

    source = pd.concat(signal_frames, ignore_index=True)
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    episode_counts = episodes.groupby("signal_id").size().to_dict()
    for record in coverage_records:
        record["episodes"] = int(episode_counts.get(record["signal_id"], 0))

    metadata = {
        "signal_version": version,
        "direction": DIRECTION,
        "requested_start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "comparison_start_date": calendar.min().strftime("%Y-%m-%d"),
        "comparison_end_date": calendar.max().strftime("%Y-%m-%d"),
        "trade_dates": len(calendar),
        "industry_count": len(coverage_records),
        "daily_rows": len(daily),
        "triggered_days": int(daily["triggered"].sum()),
        "episodes": len(episodes),
        "industry_coverage": coverage_records,
        "industry_set_eras": _industry_set_eras(industry),
    }
    return daily, episodes, metadata


def _validate_industry_breadth(
    frame: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("industry_breadth must be a pandas DataFrame")
    required = {
        "date",
        "industry_code",
        "industry_name",
        "universe_count",
        "above_count_ma20",
        "valid_count_ma20",
        "breadth_ma20",
        "rank_eligible_ma20",
        "rank_ma20",
        "is_top1_ma20",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"industry_breadth is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("industry_breadth must not be empty")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("industry_breadth contains an invalid date")
    requested_start = pd.Timestamp(start_date)
    result = result[result["date"].ge(requested_start)].copy()
    if result.empty:
        raise ValueError("industry_breadth has no rows on or after start_date")

    for column in ("industry_code", "industry_name"):
        values = result[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must contain non-empty values")
        result[column] = values.astype(str).str.strip()
    if result.duplicated(["date", "industry_code", "industry_name"]).any():
        raise ValueError("industry_breadth contains duplicate date-industry rows")
    if result.groupby("industry_code")["industry_name"].nunique().gt(1).any():
        raise ValueError("an industry code maps to multiple names in the sample")

    for column in (
        "universe_count",
        "above_count_ma20",
        "valid_count_ma20",
        "breadth_ma20",
        "rank_ma20",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("rank_eligible_ma20", "is_top1_ma20"):
        result[column] = _strict_bool(result[column], column)

    integer_columns = ("universe_count", "above_count_ma20", "valid_count_ma20")
    for column in integer_columns:
        values = result[column]
        if (
            values.isna().any()
            or values.lt(0).any()
            or np.not_equal(values, np.floor(values)).any()
        ):
            raise ValueError(f"{column} must contain non-negative integers")
        result[column] = values.astype(int)
    if result["valid_count_ma20"].gt(result["universe_count"]).any():
        raise ValueError("valid_count_ma20 must not exceed universe_count")
    if result["above_count_ma20"].gt(result["valid_count_ma20"]).any():
        raise ValueError("above_count_ma20 must not exceed valid_count_ma20")

    comparable = (
        result["rank_eligible_ma20"]
        & result["valid_count_ma20"].ge(MIN_INDUSTRY_VALID_COUNT)
        & result["breadth_ma20"].notna()
        & result["rank_ma20"].notna()
    )
    if not comparable.all():
        bad = result.loc[
            ~comparable, ["date", "industry_code", "industry_name"]
        ].head(5)
        raise ValueError(
            "industry is not continuously comparable in its observed lifespan: "
            f"{bad.to_dict(orient='records')}"
        )
    breadth = result["breadth_ma20"].to_numpy(dtype=float)
    if not np.isfinite(breadth).all() or ((breadth < 0) | (breadth > 1)).any():
        raise ValueError("breadth_ma20 must contain finite values between zero and one")
    if not result["is_top1_ma20"].equals(result["rank_ma20"].eq(1.0)):
        raise ValueError("stored Top1 flag does not match rank_ma20")
    return result.sort_values(["date", "industry_code", "industry_name"]).reset_index(
        drop=True
    )


def _validate_continuous_lifespan(
    group: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    industry_code: str,
    industry_name: str,
) -> None:
    actual = pd.DatetimeIndex(group["date"])
    expected = calendar[(calendar >= actual.min()) & (calendar <= actual.max())]
    if not actual.equals(expected):
        missing = expected.difference(actual).strftime("%Y-%m-%d").tolist()[:5]
        raise ValueError(
            "industry observation is not continuous within its lifespan: "
            f"{industry_code}/{industry_name}, missing={missing}"
        )


def _industry_set_eras(industry: pd.DataFrame) -> list[dict[str, object]]:
    daily_sets = (
        industry.groupby("date", sort=True)["industry_code"]
        .agg(lambda values: tuple(sorted(values)))
        .reset_index(name="industry_codes")
    )
    daily_sets["era"] = daily_sets["industry_codes"].ne(
        daily_sets["industry_codes"].shift()
    ).cumsum()
    name_by_code = (
        industry[["industry_code", "industry_name"]]
        .drop_duplicates()
        .set_index("industry_code")["industry_name"]
        .to_dict()
    )
    records = []
    for _, group in daily_sets.groupby("era", sort=True):
        codes = list(group["industry_codes"].iloc[0])
        records.append(
            {
                "start_date": group["date"].iloc[0].strftime("%Y-%m-%d"),
                "end_date": group["date"].iloc[-1].strftime("%Y-%m-%d"),
                "trade_dates": len(group),
                "industry_count": len(codes),
                "industry_codes": codes,
                "industry_names": [name_by_code[code] for code in codes],
            }
        )
    return records


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
