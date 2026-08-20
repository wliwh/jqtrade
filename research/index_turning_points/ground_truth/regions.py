"""Post-hoc top and bottom regions built from directional-change anchors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
import pandas as pd


REGION_COLUMNS = [
    "region_id",
    "index_id",
    "index_name",
    "event_type",
    "scale",
    "status",
    "eligible",
    "region_start",
    "region_end",
    "region_start_position",
    "region_end_position",
    "anchor_date",
    "anchor_position",
    "anchor_price",
    "first_core_date",
    "last_core_date",
    "lobe_count",
    "confirmation_date",
    "confirmation_position",
    "confirmation_price",
    "confirmation_lag",
    "reversal_return",
    "threshold",
    "price_band_pct",
    "price_band_fraction_of_threshold",
    "max_price_band_pct",
    "max_side_days",
    "max_lobe_gap",
    "label_version",
]

LOBE_COLUMNS = [
    "region_id",
    "lobe_id",
    "index_id",
    "index_name",
    "event_type",
    "scale",
    "lobe_number",
    "lobe_start",
    "lobe_end",
    "lobe_start_position",
    "lobe_end_position",
    "representative_date",
    "representative_position",
    "representative_price",
    "relative_anchor_price",
    "core_days",
    "gap_from_previous_lobe",
    "small_pivot_count",
    "small_pivot_dates",
    "label_version",
]


def _validate_windows(values: tuple[int, ...], name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive integers")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be unique and increasing")


@dataclass(frozen=True)
class RegionProtocol:
    """Frozen phase-A parameters shared by region labels and later scoring."""

    label_version: str
    price_band_fraction_of_threshold: float
    max_price_band_pct: float
    max_side_days: int
    max_lobe_gap: int
    prediction_windows: tuple[int, ...]
    confirmation_windows: tuple[int, ...]
    capped_confirmation_n: int

    def __post_init__(self) -> None:
        if not isinstance(self.label_version, str) or not self.label_version.strip():
            raise ValueError("label_version must be a non-empty string")
        fraction = float(self.price_band_fraction_of_threshold)
        if not isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("price band fraction must be between 0 and 1")
        maximum = float(self.max_price_band_pct)
        if not isfinite(maximum) or not 0.0 < maximum < 1.0:
            raise ValueError("max_price_band_pct must be between 0 and 1")
        if isinstance(self.max_side_days, bool) or not isinstance(
            self.max_side_days, int
        ):
            raise ValueError("max_side_days must be a positive integer")
        if self.max_side_days <= 0:
            raise ValueError("max_side_days must be a positive integer")
        if isinstance(self.max_lobe_gap, bool) or not isinstance(
            self.max_lobe_gap, int
        ):
            raise ValueError("max_lobe_gap must be a non-negative integer")
        if self.max_lobe_gap < 0:
            raise ValueError("max_lobe_gap must be a non-negative integer")
        _validate_windows(self.prediction_windows, "prediction_windows")
        _validate_windows(self.confirmation_windows, "confirmation_windows")
        if isinstance(self.capped_confirmation_n, bool) or not isinstance(
            self.capped_confirmation_n, int
        ):
            raise ValueError("capped_confirmation_n must be a positive integer")
        if self.capped_confirmation_n <= 0:
            raise ValueError("capped_confirmation_n must be a positive integer")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready protocol values without mutating tuple fields."""

        values = asdict(self)
        values["prediction_windows"] = list(self.prediction_windows)
        values["confirmation_windows"] = list(self.confirmation_windows)
        return values

    def resolve_price_band_pct(self, threshold: float) -> float:
        """Return the frozen threshold-relative band with its absolute cap."""

        value = float(threshold)
        if not isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("threshold must be between 0 and 1")
        return min(
            value * float(self.price_band_fraction_of_threshold),
            float(self.max_price_band_pct),
        )


DEFAULT_REGION_PROTOCOL = RegionProtocol(
    label_version="top_bottom_regions_v2",
    price_band_fraction_of_threshold=0.2,
    max_price_band_pct=0.02,
    max_side_days=20,
    max_lobe_gap=10,
    prediction_windows=(5, 10, 20),
    confirmation_windows=(5, 10, 20),
    capped_confirmation_n=2,
)


def build_turning_point_regions(
    daily: pd.DataFrame,
    medium_labels: pd.DataFrame,
    *,
    index_id: str,
    index_name: str,
    small_labels: pd.DataFrame | None = None,
    scale: str = "medium",
    protocol: RegionProtocol = DEFAULT_REGION_PROTOCOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build confirmed macro regions and their non-contiguous core lobes.

    This is deliberately a post-hoc labeler. It may use bars on both sides of a
    canonical medium directional-change anchor. Adjacent anchors divide time at
    their midpoint, which prevents top and bottom envelopes from overlapping.
    """

    _validate_daily(daily)
    labels = _validate_labels(medium_labels, daily, "medium_labels")
    small = _prepare_small_labels(small_labels, daily)
    if labels.empty:
        return _empty_regions(), _empty_lobes()

    positions = labels["anchor_position"].astype(int).tolist()
    region_records: list[dict[str, object]] = []
    lobe_records: list[dict[str, object]] = []

    for label_number, row in labels.iterrows():
        if row["status"] != "confirmed" or not _truthy(row["eligible"]):
            continue

        event_type = str(row["event_type"])
        anchor_position = int(row["anchor_position"])
        anchor_price = float(row["anchor_price"])
        threshold = float(row["threshold"])
        price_band_pct = protocol.resolve_price_band_pct(threshold)
        if not 0.0 < price_band_pct < 1.0:
            raise ValueError("derived price_band_pct must be between 0 and 1")

        cell_start = 0
        if label_number > 0:
            cell_start = (positions[label_number - 1] + anchor_position) // 2 + 1
        cell_end = len(daily) - 1
        if label_number + 1 < len(labels):
            cell_end = (anchor_position + positions[label_number + 1]) // 2
        search_start = max(cell_start, anchor_position - protocol.max_side_days)
        search_end = min(cell_end, anchor_position + protocol.max_side_days)

        price_column = "high" if event_type == "top" else "low"
        prices = daily[price_column].iloc[search_start : search_end + 1]
        if event_type == "top":
            cutoff = anchor_price * (1.0 - price_band_pct)
            near_extreme = prices.ge(cutoff)
        else:
            cutoff = anchor_price * (1.0 + price_band_pct)
            near_extreme = prices.le(cutoff)

        spans = _contiguous_spans(near_extreme.to_numpy(dtype=bool), search_start)
        anchor_lobe = next(
            (
                number
                for number, (start, end) in enumerate(spans)
                if start <= anchor_position <= end
            ),
            None,
        )
        if anchor_lobe is None:
            raise ValueError("canonical anchor is outside its derived price band")

        first_lobe = last_lobe = anchor_lobe
        while first_lobe > 0:
            gap = spans[first_lobe][0] - spans[first_lobe - 1][1] - 1
            if gap > protocol.max_lobe_gap:
                break
            first_lobe -= 1
        while last_lobe + 1 < len(spans):
            gap = spans[last_lobe + 1][0] - spans[last_lobe][1] - 1
            if gap > protocol.max_lobe_gap:
                break
            last_lobe += 1
        selected_spans = spans[first_lobe : last_lobe + 1]

        anchor_date = pd.Timestamp(row["anchor_date"])
        region_id = (
            f"{index_id}_{scale}_{event_type}_{anchor_date.strftime('%Y%m%d')}"
        )
        previous_end: int | None = None
        for lobe_number, (lobe_start, lobe_end) in enumerate(selected_spans, start=1):
            lobe_prices = daily[price_column].iloc[lobe_start : lobe_end + 1]
            representative_price = float(
                lobe_prices.max() if event_type == "top" else lobe_prices.min()
            )
            representative_matches = np.flatnonzero(
                np.isclose(
                    lobe_prices.to_numpy(dtype=float),
                    representative_price,
                    rtol=1e-12,
                    atol=1e-12,
                )
            )
            representative_position = lobe_start + int(representative_matches[-1])
            small_pivots = _small_pivots_in_lobe(
                small,
                event_type,
                lobe_start,
                lobe_end,
            )
            lobe_records.append(
                {
                    "region_id": region_id,
                    "lobe_id": f"{region_id}_lobe_{lobe_number:02d}",
                    "index_id": index_id,
                    "index_name": index_name,
                    "event_type": event_type,
                    "scale": scale,
                    "lobe_number": lobe_number,
                    "lobe_start": daily.index[lobe_start],
                    "lobe_end": daily.index[lobe_end],
                    "lobe_start_position": lobe_start,
                    "lobe_end_position": lobe_end,
                    "representative_date": daily.index[representative_position],
                    "representative_position": representative_position,
                    "representative_price": representative_price,
                    "relative_anchor_price": representative_price / anchor_price - 1.0,
                    "core_days": lobe_end - lobe_start + 1,
                    "gap_from_previous_lobe": (
                        pd.NA
                        if previous_end is None
                        else lobe_start - previous_end - 1
                    ),
                    "small_pivot_count": len(small_pivots),
                    "small_pivot_dates": "|".join(
                        pd.Timestamp(value).strftime("%Y-%m-%d")
                        for value in small_pivots["anchor_date"]
                    ),
                    "label_version": protocol.label_version,
                }
            )
            previous_end = lobe_end

        region_start, region_end = selected_spans[0][0], selected_spans[-1][1]
        region_records.append(
            {
                "region_id": region_id,
                "index_id": index_id,
                "index_name": index_name,
                "event_type": event_type,
                "scale": scale,
                "status": "confirmed",
                "eligible": True,
                "region_start": daily.index[region_start],
                "region_end": daily.index[region_end],
                "region_start_position": region_start,
                "region_end_position": region_end,
                "anchor_date": anchor_date,
                "anchor_position": anchor_position,
                "anchor_price": anchor_price,
                "first_core_date": daily.index[selected_spans[0][0]],
                "last_core_date": daily.index[selected_spans[-1][1]],
                "lobe_count": len(selected_spans),
                "confirmation_date": pd.Timestamp(row["confirmation_date"]),
                "confirmation_position": int(row["confirmation_position"]),
                "confirmation_price": float(row["confirmation_price"]),
                "confirmation_lag": int(row["confirmation_lag"]),
                "reversal_return": float(row["reversal_return"]),
                "threshold": threshold,
                "price_band_pct": price_band_pct,
                "price_band_fraction_of_threshold": (
                    protocol.price_band_fraction_of_threshold
                ),
                "max_price_band_pct": protocol.max_price_band_pct,
                "max_side_days": protocol.max_side_days,
                "max_lobe_gap": protocol.max_lobe_gap,
                "label_version": protocol.label_version,
            }
        )

    return (
        pd.DataFrame(region_records, columns=REGION_COLUMNS),
        pd.DataFrame(lobe_records, columns=LOBE_COLUMNS),
    )


def _validate_daily(daily: pd.DataFrame) -> None:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    missing = {"high", "low"}.difference(daily.columns)
    if missing:
        raise ValueError(f"daily is missing columns: {sorted(missing)}")
    if daily.empty:
        raise ValueError("daily must not be empty")
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("daily index must be unique and increasing")
    prices = daily[["high", "low"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError("daily high and low must be finite and positive")
    if (daily["high"] < daily["low"]).any():
        raise ValueError("daily high must be greater than or equal to low")


def _validate_labels(
    labels: pd.DataFrame,
    daily: pd.DataFrame,
    name: str,
) -> pd.DataFrame:
    if not isinstance(labels, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    required = {
        "event_type",
        "status",
        "eligible",
        "anchor_date",
        "anchor_position",
        "anchor_price",
        "confirmation_date",
        "confirmation_position",
        "confirmation_price",
        "confirmation_lag",
        "reversal_return",
        "threshold",
    }
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if labels.empty:
        return labels.copy().reset_index(drop=True)

    result = labels.copy().reset_index(drop=True)
    if not result["event_type"].isin(["top", "bottom"]).all():
        raise ValueError(f"{name} contains an invalid event_type")
    positions = pd.to_numeric(result["anchor_position"], errors="coerce")
    if positions.isna().any() or not positions.is_monotonic_increasing:
        raise ValueError(f"{name} anchor positions must be increasing integers")
    if not positions.is_unique or (positions % 1.0).ne(0.0).any():
        raise ValueError(f"{name} anchor positions must be unique integers")
    result["anchor_position"] = positions.astype(int)
    if (result["anchor_position"] < 0).any() or (
        result["anchor_position"] >= len(daily)
    ).any():
        raise ValueError(f"{name} anchor position is outside daily data")
    anchor_dates = pd.to_datetime(result["anchor_date"])
    expected_dates = pd.DatetimeIndex(
        [daily.index[position] for position in result["anchor_position"]]
    )
    if not anchor_dates.reset_index(drop=True).equals(pd.Series(expected_dates)):
        raise ValueError(f"{name} anchor dates do not match anchor positions")
    if (result["event_type"].eq(result["event_type"].shift())).any():
        raise ValueError(f"{name} event types must alternate")
    return result


def _prepare_small_labels(
    labels: pd.DataFrame | None,
    daily: pd.DataFrame,
) -> pd.DataFrame:
    columns = ["event_type", "status", "eligible", "anchor_date", "anchor_position"]
    if labels is None or labels.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(labels, pd.DataFrame):
        raise TypeError("small_labels must be a pandas DataFrame")
    missing = set(columns).difference(labels.columns)
    if missing:
        raise ValueError(f"small_labels is missing columns: {sorted(missing)}")
    result = labels[columns].copy()
    result["anchor_position"] = pd.to_numeric(
        result["anchor_position"], errors="coerce"
    )
    if result["anchor_position"].isna().any():
        raise ValueError("small_labels anchor positions must be integers")
    result["anchor_position"] = result["anchor_position"].astype(int)
    if (result["anchor_position"] < 0).any() or (
        result["anchor_position"] >= len(daily)
    ).any():
        raise ValueError("small_labels anchor position is outside daily data")
    return result.reset_index(drop=True)


def _contiguous_spans(mask: np.ndarray, start_position: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    span_start: int | None = None
    for offset, active in enumerate(mask):
        position = start_position + offset
        if active and span_start is None:
            span_start = position
        if span_start is not None and (not active or offset == len(mask) - 1):
            span_end = position if active and offset == len(mask) - 1 else position - 1
            spans.append((span_start, span_end))
            span_start = None
    return spans


def _small_pivots_in_lobe(
    small_labels: pd.DataFrame,
    event_type: str,
    start: int,
    end: int,
) -> pd.DataFrame:
    if small_labels.empty:
        return small_labels
    eligible = small_labels["eligible"].map(_truthy)
    return small_labels[
        small_labels["event_type"].eq(event_type)
        & small_labels["status"].eq("confirmed")
        & eligible
        & small_labels["anchor_position"].between(start, end)
    ]


def _truthy(value: object) -> bool:
    return False if pd.isna(value) else bool(value)


def _empty_regions() -> pd.DataFrame:
    return pd.DataFrame(columns=REGION_COLUMNS)


def _empty_lobes() -> pd.DataFrame:
    return pd.DataFrame(columns=LOBE_COLUMNS)
