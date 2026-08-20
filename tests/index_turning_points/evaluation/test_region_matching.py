import pandas as pd
import pytest

from research.index_turning_points.evaluation.region_matching import (
    match_signal_regions,
    summarize_region_matches,
)
from research.index_turning_points.ground_truth.regions import DEFAULT_REGION_PROTOCOL
from research.index_turning_points.signals.events import build_signal_events


def _calendar(periods=70):
    return pd.DataFrame(
        {
            "index_id": "test_index",
            "index_name": "测试指数",
            "date": pd.bdate_range("2020-01-01", periods=periods),
        }
    )


def _signal(calendar, events, direction="top"):
    frame = pd.DataFrame(
        {
            "date": calendar["date"],
            "signal_id": "test_signal",
            "direction": direction,
            "version": "test_v1",
            "episode_id": pd.NA,
            "event_onset": False,
            "event_capped_confirmation": False,
            "event_exit": False,
        }
    )
    for number, position in enumerate(events, start=1):
        frame.loc[position, "episode_id"] = f"episode_{number}"
        frame.loc[position, "event_onset"] = True
    return frame


def _region(calendar, region_id, anchor, start=None, end=None, event_type="top"):
    start = anchor if start is None else start
    end = anchor if end is None else end
    return {
        "region_id": region_id,
        "index_id": "test_index",
        "index_name": "测试指数",
        "event_type": event_type,
        "status": "confirmed",
        "eligible": True,
        "region_start": calendar.loc[start, "date"],
        "region_end": calendar.loc[end, "date"],
        "anchor_date": calendar.loc[anchor, "date"],
        "lobe_count": 1,
        "label_version": DEFAULT_REGION_PROTOCOL.label_version,
    }


def _lobe(calendar, region_id, start, end):
    return {
        "region_id": region_id,
        "lobe_id": f"{region_id}_lobe_01",
        "lobe_start": calendar.loc[start, "date"],
        "lobe_end": calendar.loc[end, "date"],
    }


def test_one_to_one_priority_preserves_duplicates_false_alarms_and_misses():
    calendar = _calendar()
    signal = _signal(calendar, events=[5, 9, 10, 26, 69])
    regions = pd.DataFrame(
        [
            _region(calendar, "r1", anchor=10, start=9, end=11),
            _region(calendar, "r2", anchor=30),
            _region(calendar, "r3", anchor=45),
        ]
    )
    lobes = pd.DataFrame(
        [
            _lobe(calendar, "r1", 10, 10),
            _lobe(calendar, "r2", 30, 30),
            _lobe(calendar, "r3", 45, 45),
        ]
    )

    matches = match_signal_regions(signal, regions, lobes, calendar)

    assert matches["match_status"].value_counts().to_dict() == {
        "matched": 2,
        "duplicate_alarm": 2,
        "false_alarm": 1,
        "missed_region": 1,
    }
    r1 = matches[(matches["region_id"].eq("r1")) & matches["primary_match"]].iloc[0]
    assert r1["episode_id"] == "episode_3"
    assert r1["match_level"] == "core_lobe"
    assert bool(r1["strict_matched"])
    assert bool(r1["loose_matched"])
    r2 = matches[(matches["region_id"].eq("r2")) & matches["primary_match"]].iloc[0]
    assert r2["episode_id"] == "episode_4"
    assert r2["match_level"] == "prediction_window"
    assert r2["timing"] == "prediction"
    assert r2["lead_lag_days"] == -4
    assert r2["window_days"] == 5
    assert not bool(r2["strict_matched"])
    assert not bool(r2["loose_matched"])
    assert matches.loc[
        matches["match_status"].eq("missed_region"), "region_id"
    ].tolist() == ["r3"]
    assert set(
        matches.loc[
            matches["match_status"].eq("duplicate_alarm"), "match_level"
        ]
    ) == {"envelope", "prediction_window"}


def test_same_level_anchor_tie_uses_stable_region_id():
    calendar = _calendar(30)
    signal = _signal(calendar, events=[10])
    regions = pd.DataFrame(
        [
            _region(calendar, "a_region", anchor=8),
            _region(calendar, "b_region", anchor=12),
        ]
    )
    lobes = pd.DataFrame(
        [
            _lobe(calendar, "a_region", 8, 8),
            _lobe(calendar, "b_region", 12, 12),
        ]
    )

    matches = match_signal_regions(signal, regions, lobes, calendar)

    primary = matches[matches["primary_match"]].iloc[0]
    assert primary["region_id"] == "a_region"
    assert primary["lead_lag_days"] == 2
    assert matches.loc[
        matches["match_status"].eq("missed_region"), "region_id"
    ].tolist() == ["b_region"]


def test_top_signal_never_matches_bottom_region():
    calendar = _calendar(30)
    signal = _signal(calendar, events=[10], direction="top")
    regions = pd.DataFrame(
        [_region(calendar, "bottom_region", anchor=10, event_type="bottom")]
    )
    lobes = pd.DataFrame([_lobe(calendar, "bottom_region", 10, 10)])

    matches = match_signal_regions(signal, regions, lobes, calendar)

    assert matches["match_status"].tolist() == ["false_alarm"]


def test_short_episode_exit_confirmation_can_be_matched_separately():
    calendar = _calendar(30)
    source = pd.DataFrame(
        {
            "date": calendar["date"],
            "signal_id": "test_signal",
            "direction": "top",
            "raw_value": 0.0,
            "triggered": [False] * 9 + [True] + [False] * 20,
            "universe_size": 100,
            "valid_count": 90,
            "version": "test_v1",
        }
    )
    signal, _ = build_signal_events(source, capped_confirmation_n=2)
    regions = pd.DataFrame([_region(calendar, "r1", anchor=10)])
    lobes = pd.DataFrame([_lobe(calendar, "r1", 10, 10)])

    matches = match_signal_regions(
        signal,
        regions,
        lobes,
        calendar,
        event_kind="capped_confirmation",
    )

    assert matches["match_status"].tolist() == ["matched"]
    assert matches.iloc[0]["event_date"] == calendar.loc[10, "date"]
    assert matches.iloc[0]["match_level"] == "core_lobe"


def test_metrics_separate_strict_loose_and_window_scopes():
    calendar = _calendar()
    signal = _signal(calendar, events=[10, 26, 69])
    regions = pd.DataFrame(
        [
            _region(calendar, "r1", anchor=10),
            _region(calendar, "r2", anchor=30),
            _region(calendar, "r3", anchor=45),
        ]
    )
    lobes = pd.DataFrame(
        [
            _lobe(calendar, "r1", 10, 10),
            _lobe(calendar, "r2", 30, 30),
            _lobe(calendar, "r3", 45, 45),
        ]
    )
    matches = match_signal_regions(signal, regions, lobes, calendar)

    metrics = summarize_region_matches(matches).set_index("match_scope")

    assert metrics.loc["strict", "matched_region_count"] == 1
    assert metrics.loc["loose", "matched_region_count"] == 1
    assert metrics.loc["window", "matched_region_count"] == 2
    assert metrics.loc["window", "region_recall"] == pytest.approx(2 / 3)
    assert metrics.loc["window", "episode_precision"] == pytest.approx(2 / 3)


def test_rejects_signal_calendar_gaps():
    calendar = _calendar(30)
    signal = _signal(calendar, events=[10]).drop(index=5).reset_index(drop=True)
    regions = pd.DataFrame([_region(calendar, "r1", anchor=10)])
    lobes = pd.DataFrame([_lobe(calendar, "r1", 10, 10)])

    with pytest.raises(ValueError, match="calendar mismatch"):
        match_signal_regions(signal, regions, lobes, calendar)


def test_uses_aligned_signal_calendar_when_one_index_source_has_date_anomalies():
    canonical = _calendar(30)
    signal = _signal(canonical, events=[10])
    anomalous = canonical.drop(index=[5]).copy()
    anomalous.loc[len(anomalous)] = {
        "index_id": "test_index",
        "index_name": "测试指数",
        "date": pd.Timestamp("2020-01-11"),
    }
    reference = canonical.assign(
        index_id="reference_index",
        index_name="参考指数",
    )
    calendars = pd.concat([anomalous, reference], ignore_index=True)
    regions = pd.DataFrame([_region(canonical, "r1", anchor=10)])
    lobes = pd.DataFrame([_lobe(canonical, "r1", 10, 10)])

    matches = match_signal_regions(signal, regions, lobes, calendars)

    selected = matches[matches["index_id"].eq("test_index")]
    assert selected["match_status"].tolist() == ["matched"]
    assert selected.iloc[0]["lead_lag_days"] == 0
