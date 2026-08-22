import numpy as np
import pandas as pd

from research.index_turning_points.ground_truth.regions import (
    DEFAULT_REGION_PROTOCOL,
)
from research.index_turning_points.modeling.ma20_episode_dataset import (
    MA20_EPISODE_FEATURE_COLUMNS,
    build_ma20_episode_dataset,
)
from research.index_turning_points.signals.events import build_signal_events


def _signal_bundle(calendar):
    frames = []
    for signal_id, direction, onset_position in (
        ("ma20_breadth_reversal_top", "top", 20),
        ("ma20_breadth_reversal_bottom", "bottom", 40),
    ):
        triggered = np.zeros(len(calendar), dtype=bool)
        triggered[onset_position] = True
        frames.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "signal_id": signal_id,
                    "direction": direction,
                    "raw_value": triggered.astype(float),
                    "triggered": triggered,
                    "universe_size": 100,
                    "valid_count": 90,
                    "version": "ma20_test_v1",
                }
            )
        )
    return build_signal_events(pd.concat(frames, ignore_index=True))


def _region(region_id, direction, anchor, start, end):
    return {
        "region_id": region_id,
        "index_id": "all_a",
        "index_name": "全A",
        "event_type": direction,
        "status": "confirmed",
        "eligible": True,
        "region_start": start,
        "region_end": end,
        "anchor_date": anchor,
        "lobe_count": 1,
        "label_version": DEFAULT_REGION_PROTOCOL.label_version,
    }


def test_builds_one_row_per_ma20_candidate_with_five_day_operational_target():
    calendar = pd.bdate_range("2020-01-01", periods=100)
    signal_daily, signal_episodes = _signal_bundle(calendar)
    feature_columns = [
        column
        for column in MA20_EPISODE_FEATURE_COLUMNS
        if column != "candidate_gap_trade_days"
    ]
    feature_daily = pd.DataFrame({"date": calendar})
    for number, column in enumerate(feature_columns, start=1):
        feature_daily[column] = np.arange(len(calendar), dtype=float) + number
    regions = pd.DataFrame(
        [
            _region("old", "top", pd.Timestamp("2019-12-20"), pd.Timestamp("2019-12-20"), pd.Timestamp("2019-12-20")),
            _region("top_r", "top", calendar[25], calendar[25], calendar[25]),
            _region("bottom_r", "bottom", calendar[50], calendar[50], calendar[50]),
        ]
    )
    lobes = pd.DataFrame(
        {
            "lobe_id": ["old_lobe", "top_lobe", "bottom_lobe"],
            "region_id": ["old", "top_r", "bottom_r"],
            "lobe_start": [pd.Timestamp("2019-12-20"), calendar[25], calendar[50]],
            "lobe_end": [pd.Timestamp("2019-12-20"), calendar[25], calendar[50]],
        }
    )

    result = build_ma20_episode_dataset(
        signal_daily,
        signal_episodes,
        feature_daily,
        regions,
        lobes,
    )
    candidates = result.candidate_episodes.set_index("direction")

    assert len(candidates) == 2
    assert bool(candidates.loc["top", "target_operational_match"])
    assert not bool(candidates.loc["bottom", "target_operational_match"])
    assert candidates.loc["top", "operational_lead_lag_days"] == -5
    assert candidates["operational_window_trade_days"].eq(5).all()
    assert candidates["target_legacy_window_20d_match"].all()
    assert candidates["candidate_gap_trade_days"].eq(252).all()
    assert len(result.daily_calendar) == len(calendar)
