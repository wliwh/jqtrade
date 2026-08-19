import pandas as pd
import pytest

from research.index_turning_points.regions import (
    DEFAULT_REGION_PROTOCOL,
    RegionProtocol,
    build_turning_point_regions,
)


def _daily(extremes, event_type):
    index = pd.bdate_range("2020-01-01", periods=len(extremes))
    values = pd.Series(extremes, index=index, dtype=float)
    if event_type == "top":
        high = values
        low = values - 2.0
    else:
        low = values
        high = values + 2.0
    return pd.DataFrame(
        {"open": (high + low) / 2.0, "high": high, "low": low, "close": (high + low) / 2.0},
        index=index,
    )


def _label(daily, position, event_type, *, eligible=True, status="confirmed"):
    price_column = "high" if event_type == "top" else "low"
    confirmation_position = min(position + 1, len(daily) - 1)
    anchor_price = float(daily.iloc[position][price_column])
    confirmation_price = float(
        daily.iloc[confirmation_position]["low" if event_type == "top" else "high"]
    )
    return {
        "event_type": event_type,
        "status": status,
        "eligible": eligible,
        "anchor_date": daily.index[position],
        "anchor_position": position,
        "anchor_price": anchor_price,
        "confirmation_date": (
            daily.index[confirmation_position] if status == "confirmed" else pd.NaT
        ),
        "confirmation_position": (
            confirmation_position if status == "confirmed" else pd.NA
        ),
        "confirmation_price": confirmation_price if status == "confirmed" else pd.NA,
        "confirmation_lag": (
            confirmation_position - position if status == "confirmed" else pd.NA
        ),
        "reversal_return": (
            confirmation_price / anchor_price - 1.0
            if status == "confirmed"
            else pd.NA
        ),
        "threshold": 0.10,
    }


def _three_point_labels(daily, event_type):
    opposite = "bottom" if event_type == "top" else "top"
    return pd.DataFrame(
        [
            _label(daily, 0, opposite, eligible=False),
            _label(daily, 3, event_type),
            _label(daily, 10, opposite, status="unconfirmed", eligible=False),
        ]
    )


def _protocol(**overrides):
    values = {
        "label_version": "test_regions_v1",
        "price_band_fraction_of_threshold": 0.25,
        "max_price_band_pct": 0.02,
        "max_side_days": 10,
        "max_lobe_gap": 2,
        "prediction_windows": (5, 10, 20),
        "confirmation_windows": (5, 10, 20),
        "capped_confirmation_n": 2,
    }
    values.update(overrides)
    return RegionProtocol(**values)


def test_default_protocol_freezes_phase_a_parameters():
    assert DEFAULT_REGION_PROTOCOL.label_version == "top_bottom_regions_v2"
    assert DEFAULT_REGION_PROTOCOL.price_band_fraction_of_threshold == 0.20
    assert DEFAULT_REGION_PROTOCOL.max_price_band_pct == 0.02
    assert DEFAULT_REGION_PROTOCOL.max_side_days == 20
    assert DEFAULT_REGION_PROTOCOL.max_lobe_gap == 10
    assert DEFAULT_REGION_PROTOCOL.prediction_windows == (5, 10, 20)
    assert DEFAULT_REGION_PROTOCOL.confirmation_windows == (5, 10, 20)
    assert DEFAULT_REGION_PROTOCOL.capped_confirmation_n == 2


@pytest.mark.parametrize(
    "event_type,extremes",
    [
        ("top", [90, 92, 96, 100, 96, 95, 97.8, 94, 92, 90, 88]),
        ("bottom", [110, 108, 104, 100, 104, 105, 102.2, 106, 108, 110, 112]),
    ],
)
def test_absolute_price_band_cap_excludes_marginal_secondary_lobe(
    event_type,
    extremes,
):
    daily = _daily(extremes, event_type)

    regions, lobes = build_turning_point_regions(
        daily,
        _three_point_labels(daily, event_type),
        index_id="test",
        index_name="测试指数",
        protocol=_protocol(
            price_band_fraction_of_threshold=0.25,
            max_price_band_pct=0.02,
        ),
    )

    assert regions.iloc[0]["price_band_pct"] == pytest.approx(0.02)
    assert regions.iloc[0]["max_price_band_pct"] == pytest.approx(0.02)
    assert regions.iloc[0]["lobe_count"] == 1
    assert len(lobes) == 1


@pytest.mark.parametrize(
    "event_type,extremes",
    [
        ("top", [90, 92, 96, 100, 96, 95, 99, 94, 92, 90, 88]),
        ("bottom", [110, 108, 104, 100, 104, 105, 101, 106, 108, 110, 112]),
    ],
)
def test_m_top_and_w_bottom_keep_two_core_lobes(event_type, extremes):
    daily = _daily(extremes, event_type)
    medium = _three_point_labels(daily, event_type)
    small = pd.DataFrame(
        [
            _label(daily, 3, event_type),
            _label(daily, 6, event_type),
        ]
    )

    regions, lobes = build_turning_point_regions(
        daily,
        medium,
        index_id="test",
        index_name="测试指数",
        small_labels=small,
        protocol=_protocol(),
    )

    assert len(regions) == 1
    assert regions.iloc[0]["event_type"] == event_type
    assert regions.iloc[0]["anchor_date"] == daily.index[3]
    assert regions.iloc[0]["confirmation_date"] == daily.index[4]
    assert regions.iloc[0]["confirmation_lag"] == 1
    assert regions.iloc[0]["region_start"] == daily.index[3]
    assert regions.iloc[0]["region_end"] == daily.index[6]
    assert regions.iloc[0]["lobe_count"] == 2
    assert lobes["lobe_start"].tolist() == [daily.index[3], daily.index[6]]
    assert lobes["small_pivot_count"].tolist() == [1, 1]


def test_plateau_is_one_multi_day_core_lobe():
    daily = _daily([90, 92, 96, 100, 99, 99.5, 95, 93, 91, 89, 87], "top")
    regions, lobes = build_turning_point_regions(
        daily,
        _three_point_labels(daily, "top"),
        index_id="test",
        index_name="测试指数",
        protocol=_protocol(),
    )

    assert regions.iloc[0]["lobe_count"] == 1
    assert regions.iloc[0]["region_start"] == daily.index[3]
    assert regions.iloc[0]["region_end"] == daily.index[5]
    assert lobes.iloc[0]["core_days"] == 3
    assert lobes.iloc[0]["representative_date"] == daily.index[3]


def test_gap_above_limit_does_not_join_a_distant_near_extreme():
    daily = _daily([90, 92, 96, 100, 95, 94, 93, 99, 92, 90, 88], "top")
    regions, lobes = build_turning_point_regions(
        daily,
        _three_point_labels(daily, "top"),
        index_id="test",
        index_name="测试指数",
        protocol=_protocol(max_lobe_gap=2),
    )

    assert regions.iloc[0]["lobe_count"] == 1
    assert regions.iloc[0]["region_end"] == daily.index[3]
    assert len(lobes) == 1


def test_adjacent_medium_regions_use_non_overlapping_midpoint_cells():
    daily = pd.DataFrame(
        {
            "high": [82, 90, 95, 100, 94, 88, 80, 72, 82, 91, 96, 99, 94, 85, 77],
            "low": [80, 88, 93, 98, 92, 86, 78, 70, 80, 89, 94, 97, 92, 83, 75],
        },
        index=pd.bdate_range("2020-01-01", periods=15),
    )
    labels = pd.DataFrame(
        [
            _label(daily, 0, "bottom", eligible=False),
            _label(daily, 3, "top"),
            _label(daily, 7, "bottom"),
            _label(daily, 11, "top"),
            _label(daily, 14, "bottom", status="unconfirmed", eligible=False),
        ]
    )

    regions, _ = build_turning_point_regions(
        daily,
        labels,
        index_id="test",
        index_name="测试指数",
        protocol=_protocol(max_side_days=20, max_lobe_gap=10),
    )

    assert regions["event_type"].tolist() == ["top", "bottom", "top"]
    assert all(
        left < right
        for left, right in zip(
            regions["region_end_position"].iloc[:-1],
            regions["region_start_position"].iloc[1:],
        )
    )


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"price_band_fraction_of_threshold": 0}, "price band"),
        ({"max_price_band_pct": 0}, "max_price_band_pct"),
        ({"max_side_days": 0}, "max_side_days"),
        ({"max_lobe_gap": -1}, "max_lobe_gap"),
        ({"prediction_windows": (10, 5)}, "prediction_windows"),
        ({"capped_confirmation_n": 0}, "capped_confirmation_n"),
    ],
)
def test_protocol_rejects_invalid_frozen_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _protocol(**kwargs)
