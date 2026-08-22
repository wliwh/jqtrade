"""Post-hoc intensity and future-entry targets for top/bottom models."""

from __future__ import annotations

from math import isfinite

import numpy as np
import pandas as pd


DEFAULT_HORIZONS = (5, 10, 20)
INTENSITY_COLUMNS = {
    "top": "truth_top_intensity",
    "bottom": "truth_bottom_intensity",
}
MEMBERSHIP_COLUMNS = {
    "top": "truth_top_in_strict_lobe",
    "bottom": "truth_bottom_in_strict_lobe",
}


def build_lobe_intensity_targets(
    daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    *,
    index_id: str = "all_a",
) -> pd.DataFrame:
    """Score each strict core lobe from zero to 100 on its index calendar.

    Top lobes use daily highs and bottom lobes use daily lows. Every lobe's
    representative extreme scores 100. Dates outside strict lobes remain zero;
    in particular, gaps between the lobes of an M/W region are not filled.
    """

    prices = _validate_daily(daily)
    selected_regions = _validate_regions(regions, index_id=index_id)
    selected_lobes = _validate_lobes(lobes, index_id=index_id)
    selected_lobes = selected_lobes.merge(
        selected_regions[["region_id", "event_type", "price_band_pct"]],
        on=["region_id", "event_type"],
        how="inner",
        validate="many_to_one",
    )
    if selected_lobes.empty:
        raise ValueError(f"no strict lobes found for index_id={index_id!r}")

    values = {
        direction: np.zeros(len(prices), dtype=float)
        for direction in INTENSITY_COLUMNS
    }
    memberships = {
        direction: np.zeros(len(prices), dtype=bool)
        for direction in INTENSITY_COLUMNS
    }
    for row in selected_lobes.itertuples(index=False):
        direction = str(row.event_type)
        price_column = "high" if direction == "top" else "low"
        start = pd.Timestamp(row.lobe_start)
        end = pd.Timestamp(row.lobe_end)
        positions = np.flatnonzero((prices.index >= start) & (prices.index <= end))
        if not len(positions):
            raise ValueError(f"lobe has no dates in index calendar: {row.lobe_id}")
        if prices.index[positions[0]] != start or prices.index[positions[-1]] != end:
            raise ValueError(f"lobe boundary is missing from index calendar: {row.lobe_id}")

        representative = float(row.representative_price)
        band = float(row.price_band_pct)
        if not isfinite(representative) or representative <= 0.0:
            raise ValueError(f"invalid representative price: {row.lobe_id}")
        if not isfinite(band) or not 0.0 < band < 1.0:
            raise ValueError(f"invalid price band: {row.lobe_id}")

        observed = prices.iloc[positions][price_column].to_numpy(dtype=float)
        if direction == "top":
            relative_gap = (representative - observed) / representative
        else:
            relative_gap = (observed - representative) / representative
        if (relative_gap < -1e-10).any():
            raise ValueError(
                f"representative price is not the lobe extreme: {row.lobe_id}"
            )
        scores = 100.0 * np.clip(1.0 - relative_gap / band, 0.0, 1.0)
        values[direction][positions] = np.maximum(
            values[direction][positions], scores
        )
        memberships[direction][positions] = True

    result = pd.DataFrame({"date": prices.index})
    for direction, column in INTENSITY_COLUMNS.items():
        result[column] = values[direction]
        result[MEMBERSHIP_COLUMNS[direction]] = memberships[direction]
    return result


def add_future_entry_targets(
    intensity_daily: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Add nested future strict-lobe entry labels with incomplete tails as NA."""

    _validate_horizons(horizons)
    required = {
        "date",
        *INTENSITY_COLUMNS.values(),
        *MEMBERSHIP_COLUMNS.values(),
    }
    missing = required.difference(intensity_daily.columns)
    if missing:
        raise ValueError(f"intensity_daily is missing columns: {sorted(missing)}")
    result = intensity_daily.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("intensity_daily contains an invalid date")
    if result["date"].duplicated().any() or not result["date"].is_monotonic_increasing:
        raise ValueError("intensity_daily dates must be unique and increasing")

    n_rows = len(result)
    for horizon in horizons:
        complete = np.arange(n_rows) + horizon < n_rows
        result[f"target_complete_{horizon}d"] = complete
        for direction, intensity_column in INTENSITY_COLUMNS.items():
            intensity = pd.to_numeric(
                result[intensity_column], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.isfinite(intensity).all() or ((intensity < 0) | (intensity > 100)).any():
                raise ValueError(f"{intensity_column} must be finite and in [0, 100]")
            membership = _strict_bool(
                result[MEMBERSHIP_COLUMNS[direction]],
                MEMBERSHIP_COLUMNS[direction],
            ).to_numpy(dtype=bool)
            target = pd.Series(pd.NA, index=result.index, dtype="boolean")
            for position in np.flatnonzero(complete):
                target.iloc[position] = bool(
                    membership[position : position + horizon + 1].any()
                )
            result[f"target_{direction}_within_{horizon}d"] = target
    return result


def _validate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    missing = {"high", "low"}.difference(daily.columns)
    if missing:
        raise ValueError(f"daily is missing columns: {sorted(missing)}")
    if not isinstance(daily.index, pd.DatetimeIndex):
        raise TypeError("daily must use a DatetimeIndex")
    if daily.empty or not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be non-empty, unique and increasing")
    result = daily[["high", "low"]].copy()
    for column in ("high", "low"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("daily high/low must be finite")
    if (result <= 0.0).any().any() or (result["high"] < result["low"]).any():
        raise ValueError("daily must contain positive, valid high/low prices")
    return result


def _validate_regions(regions: pd.DataFrame, *, index_id: str) -> pd.DataFrame:
    required = {"region_id", "index_id", "event_type", "price_band_pct"}
    missing = required.difference(regions.columns)
    if missing:
        raise ValueError(f"regions is missing columns: {sorted(missing)}")
    result = regions[regions["index_id"].eq(index_id)].copy()
    if result.empty:
        raise ValueError(f"regions has no rows for index_id={index_id!r}")
    if result["region_id"].duplicated().any():
        raise ValueError("region_id must be unique")
    if not result["event_type"].isin(INTENSITY_COLUMNS).all():
        raise ValueError("region event_type must be top or bottom")
    return result


def _validate_lobes(lobes: pd.DataFrame, *, index_id: str) -> pd.DataFrame:
    required = {
        "region_id",
        "lobe_id",
        "index_id",
        "event_type",
        "lobe_start",
        "lobe_end",
        "representative_price",
    }
    missing = required.difference(lobes.columns)
    if missing:
        raise ValueError(f"lobes is missing columns: {sorted(missing)}")
    result = lobes[lobes["index_id"].eq(index_id)].copy()
    for column in ("lobe_start", "lobe_end"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if result.empty or result[["lobe_start", "lobe_end"]].isna().any().any():
        raise ValueError(f"lobes has no valid rows for index_id={index_id!r}")
    if result["lobe_id"].duplicated().any():
        raise ValueError("lobe_id must be unique")
    if (result["lobe_start"] > result["lobe_end"]).any():
        raise ValueError("lobe_start must not be after lobe_end")
    if not result["event_type"].isin(INTENSITY_COLUMNS).all():
        raise ValueError("lobe event_type must be top or bottom")
    return result


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if not isinstance(horizons, tuple) or not horizons:
        raise ValueError("horizons must be a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if any(value <= 0 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    if horizons != tuple(sorted(set(horizons))):
        raise ValueError("horizons must be unique and increasing")


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError(f"{name} must be boolean")
    return normalized.eq("true")
