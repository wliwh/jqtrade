"""Causal all-A MA20 breadth-versus-price divergence signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..events import build_signal_events


SIGNAL_VERSION = "breadth_price_divergence_v1_20120104_20260814"
SIGNAL_ID = "all_a_ma20_breadth_price_divergence_top"
REQUESTED_START_DATE = pd.Timestamp("2012-01-01")
ROLLING_HIGH_WINDOW = 60
MIN_PRICE_OBSERVATIONS = 59
PRICE_NEAR_HIGH_THRESHOLD = 0.02
DIVERGENCE_THRESHOLD = 0.20
MAX_MISSING_PRICE_DATES = 2


def build_breadth_price_divergence_signal(
    daily_features: pd.DataFrame,
    index_prices: pd.DataFrame,
    *,
    version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Build one top signal from all-A price and MA20 breadth high distances."""

    market = _validate_daily_features(daily_features, start_date=start_date)
    prices = _validate_index_prices(index_prices)
    market = market.merge(prices, on="date", how="left", validate="one_to_one")
    missing_dates = market.loc[market["index_close"].isna(), "date"]
    if len(missing_dates) > MAX_MISSING_PRICE_DATES:
        raise ValueError(
            "index_prices has too many missing dates inside the breadth calendar: "
            f"{missing_dates.dt.strftime('%Y-%m-%d').tolist()}"
        )

    rolling_close = market["index_close"].rolling(
        ROLLING_HIGH_WINDOW, min_periods=1
    )
    market["index_price_valid_count_60"] = rolling_close.count().astype(int)
    market["index_close_high_60"] = rolling_close.max()
    market["breadth_ma20_high_60"] = market["breadth_ma20"].rolling(
        ROLLING_HIGH_WINDOW,
        min_periods=ROLLING_HIGH_WINDOW,
    ).max()
    position_available = pd.Series(
        np.arange(len(market)) >= ROLLING_HIGH_WINDOW - 1,
        index=market.index,
    )
    market["comparison_available"] = (
        position_available
        & market["index_close"].notna()
        & market["index_price_valid_count_60"].ge(MIN_PRICE_OBSERVATIONS)
        & market["breadth_ma20_high_60"].gt(0)
    )
    market["index_price_distance_from_high_60"] = (
        1.0 - market["index_close"].divide(market["index_close_high_60"])
    ).where(market["comparison_available"])
    market["breadth_distance_from_high_60"] = (
        1.0 - market["breadth_ma20"].divide(market["breadth_ma20_high_60"])
    ).where(market["comparison_available"])
    market["breadth_price_divergence"] = (
        market["breadth_distance_from_high_60"]
        - market["index_price_distance_from_high_60"]
    ).where(market["comparison_available"])
    triggered = (
        market["comparison_available"]
        & market["index_price_distance_from_high_60"].le(
            PRICE_NEAR_HIGH_THRESHOLD
        )
        & market["breadth_price_divergence"].ge(DIVERGENCE_THRESHOLD)
    )

    source = pd.DataFrame(
        {
            "date": market["date"],
            "signal_id": SIGNAL_ID,
            "direction": "top",
            "raw_value": market["breadth_price_divergence"],
            "triggered": triggered,
            "universe_size": market["universe_size"],
            "valid_count": market["valid_count_ma20"],
            "version": version,
            "index_id": "all_a",
            "index_name": "全A",
            "index_close": market["index_close"],
            "index_close_high_60": market["index_close_high_60"],
            "breadth_ma20": market["breadth_ma20"],
            "breadth_ma20_high_60": market["breadth_ma20_high_60"],
            "index_price_distance_from_high_60": market[
                "index_price_distance_from_high_60"
            ],
            "breadth_distance_from_high_60": market[
                "breadth_distance_from_high_60"
            ],
            "breadth_price_divergence": market["breadth_price_divergence"],
            "comparison_available": market["comparison_available"],
            "index_price_valid_count_60": market[
                "index_price_valid_count_60"
            ],
            "above_count_ma20": market["above_count_ma20"],
            "valid_count_ma20": market["valid_count_ma20"],
        }
    )
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    metadata = {
        "signal_version": version,
        "requested_start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "comparison_start_date": market["date"].min().strftime("%Y-%m-%d"),
        "comparison_end_date": market["date"].max().strftime("%Y-%m-%d"),
        "first_available_date": market.loc[
            market["comparison_available"], "date"
        ].iloc[0].strftime("%Y-%m-%d"),
        "trade_dates": len(market),
        "comparison_available_dates": int(market["comparison_available"].sum()),
        "comparison_unavailable_dates": int(
            (~market["comparison_available"]).sum()
        ),
        "missing_index_price_dates": missing_dates.dt.strftime(
            "%Y-%m-%d"
        ).tolist(),
        "triggered_days": int(triggered.sum()),
        "episodes": len(episodes),
    }
    return daily, episodes, metadata


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
        "above_count_ma20",
        "valid_count_ma20",
        "breadth_ma20",
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
    if len(result) < ROLLING_HIGH_WINDOW:
        raise ValueError("daily_features does not cover the rolling-high window")

    for column in ("universe_size", "above_count_ma20", "valid_count_ma20"):
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
    if result["valid_count_ma20"].eq(0).any():
        raise ValueError("valid_count_ma20 must be positive")
    if result["valid_count_ma20"].gt(result["universe_size"]).any():
        raise ValueError("valid_count_ma20 must not exceed universe_size")
    if result["above_count_ma20"].gt(result["valid_count_ma20"]).any():
        raise ValueError("above_count_ma20 must not exceed valid_count_ma20")
    breadth = pd.to_numeric(result["breadth_ma20"], errors="coerce")
    if (
        breadth.isna().any()
        or not np.isfinite(breadth.to_numpy(dtype=float)).all()
        or breadth.lt(0).any()
        or breadth.gt(1).any()
    ):
        raise ValueError("breadth_ma20 must contain finite values from zero to one")
    expected = result["above_count_ma20"].divide(result["valid_count_ma20"])
    if not np.allclose(breadth, expected, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "breadth_ma20 does not match above_count_ma20/valid_count_ma20"
        )
    result["breadth_ma20"] = breadth.astype(float)
    return result.reset_index(drop=True)


def _validate_index_prices(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("index_prices must be a pandas DataFrame")
    required = {"date", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"index_prices is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("index_prices must not be empty")
    result = frame[["date", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["index_close"] = pd.to_numeric(result.pop("close"), errors="coerce")
    if result["date"].isna().any():
        raise ValueError("index_prices contains an invalid date")
    if result["date"].duplicated().any():
        raise ValueError("index_prices contains duplicate dates")
    if not result["date"].is_monotonic_increasing:
        raise ValueError("index_prices dates must be strictly increasing")
    close = result["index_close"].to_numpy(dtype=float)
    if not np.isfinite(close).all() or (close <= 0).any():
        raise ValueError("index close must contain finite positive values")
    return result
