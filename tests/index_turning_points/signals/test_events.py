import pandas as pd
import pytest

from research.index_turning_points.signals.events import build_signal_events


def _daily(triggered):
    dates = pd.bdate_range("2020-01-01", periods=len(triggered))
    return pd.DataFrame(
        {
            "date": dates,
            "signal_id": "test_signal",
            "direction": "top",
            "raw_value": range(len(triggered)),
            "triggered": triggered,
            "universe_size": 100,
            "valid_count": 90,
            "version": "test_v1",
        }
    )


def test_builds_onset_exit_continuation_and_capped_confirmation():
    daily, episodes = build_signal_events(
        _daily([False, True, False, True, True, True]),
        capped_confirmation_n=2,
    )

    assert daily["episode_stage"].tolist() == [
        "inactive",
        "onset",
        "exit",
        "onset",
        "continuation",
        "continuation",
    ]
    assert daily["event_onset"].tolist() == [False, True, False, True, False, False]
    assert daily["event_exit"].tolist() == [False, False, True, False, False, False]
    assert daily["event_capped_confirmation"].tolist() == [
        False,
        False,
        True,
        False,
        True,
        False,
    ]
    assert daily["capped_confirmation_reason"].tolist() == [
        "",
        "",
        "short_episode_exit",
        "",
        "nth_active_day",
        "",
    ]
    assert episodes["active_days"].tolist() == [1, 3]
    assert episodes["status"].tolist() == ["closed", "active"]
    assert episodes["confirmation_status"].tolist() == ["confirmed", "confirmed"]
    assert episodes["capped_confirmation_date"].tolist() == [
        daily.loc[2, "date"],
        daily.loc[4, "date"],
    ]


def test_sample_tail_short_episode_stays_pending_without_backfill():
    daily, episodes = build_signal_events(_daily([False, True]), capped_confirmation_n=2)

    assert not daily["event_capped_confirmation"].any()
    assert episodes.iloc[0]["status"] == "active"
    assert episodes.iloc[0]["confirmation_status"] == "pending"
    assert pd.isna(episodes.iloc[0]["capped_confirmation_date"])


def test_daily_events_are_invariant_to_every_sample_truncation():
    source = _daily([False, True, False, True, True, True, False, True])
    full, _ = build_signal_events(source, capped_confirmation_n=2)
    causal_columns = [
        "date",
        "triggered",
        "episode_id",
        "episode_number",
        "episode_day",
        "episode_stage",
        "event_onset",
        "event_continuation",
        "event_exit",
        "event_capped_confirmation",
        "capped_confirmation_reason",
    ]

    for length in range(1, len(source) + 1):
        truncated, _ = build_signal_events(
            source.iloc[:length],
            capped_confirmation_n=2,
        )
        pd.testing.assert_frame_equal(
            truncated[causal_columns].reset_index(drop=True),
            full.iloc[:length][causal_columns].reset_index(drop=True),
        )


def test_multiple_signal_series_have_independent_stable_episode_ids():
    first = _daily([True, False])
    second = first.assign(signal_id="other_signal", direction="bottom")

    daily, episodes = build_signal_events(pd.concat([second, first], ignore_index=True))

    assert len(episodes) == 2
    assert episodes["episode_id"].is_unique
    assert set(episodes["direction"]) == {"top", "bottom"}
    assert daily.groupby(["signal_id", "direction"])["episode_number"].max().eq(1).all()


@pytest.mark.parametrize(
    "change,message",
    [
        ({"triggered": ["unknown", False]}, "invalid boolean"),
        ({"valid_count": [101, 90]}, "must not exceed"),
        ({"direction": ["up", "up"]}, "top or bottom"),
    ],
)
def test_rejects_invalid_point_in_time_inputs(change, message):
    source = _daily([True, False]).assign(**change)

    with pytest.raises(ValueError, match=message):
        build_signal_events(source)
