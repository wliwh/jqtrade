import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.single_industry_top1 import (
    build_single_industry_top1_signals,
)


def _industry_rows():
    dates = pd.bdate_range("2020-01-01", periods=6)
    records = []
    eras = [
        [("old", "旧行业"), ("stable", "存续行业")],
        [("new_a", "新行业甲"), ("new_b", "新行业乙"), ("stable", "存续行业")],
    ]
    ranks = [
        {"old": 2, "stable": 1},
        {"old": 1, "stable": 2},
        {"old": 1, "stable": 1},
        {"new_a": 1, "new_b": 3, "stable": 2},
        {"new_a": 2, "new_b": 1, "stable": 3},
        {"new_a": 3, "new_b": 2, "stable": 1},
    ]
    for position, date in enumerate(dates):
        members = eras[0] if position < 3 else eras[1]
        for code, name in members:
            rank = ranks[position][code]
            valid_count = 10
            above_count = max(0, 11 - rank * 3)
            records.append(
                {
                    "date": date,
                    "industry_code": code,
                    "industry_name": name,
                    "universe_count": 12,
                    "above_count_ma20": above_count,
                    "valid_count_ma20": valid_count,
                    "breadth_ma20": above_count / valid_count,
                    "rank_eligible_ma20": True,
                    "rank_ma20": float(rank),
                    "is_top1_ma20": rank == 1,
                }
            )
    return pd.DataFrame(records)


def test_uses_each_point_in_time_industry_only_inside_its_lifespan():
    daily, episodes, metadata = build_single_industry_top1_signals(
        _industry_rows(), start_date="2020-01-01"
    )

    old = daily[daily["industry_code"].eq("old")]
    new_a = daily[daily["industry_code"].eq("new_a")]
    stable = daily[daily["industry_code"].eq("stable")]
    assert len(old) == 3
    assert len(new_a) == 3
    assert len(stable) == 6
    assert old["date"].max() < new_a["date"].min()
    assert metadata["industry_count"] == 4
    assert [era["industry_count"] for era in metadata["industry_set_eras"]] == [2, 3]

    old_episode = episodes[episodes["signal_id"].eq("single_industry_top1_old")]
    assert len(old_episode) == 1
    assert old_episode.iloc[0]["status"] == "active"
    assert pd.isna(old_episode.iloc[0]["exit_date"])
    assert not old["event_exit"].any()


def test_ties_are_independent_top1_triggers_and_raw_value_is_industry_breadth():
    daily, _, _ = build_single_industry_top1_signals(
        _industry_rows(), start_date="2020-01-01"
    )
    tied_date = pd.Timestamp("2020-01-03")
    tied = daily[daily["date"].eq(tied_date)].sort_values("industry_code")

    assert tied["triggered"].tolist() == [True, True]
    assert tied["top1_tie_count_ma20"].eq(2).all()
    assert tied["raw_value"].equals(tied["industry_breadth_ma20"])


def test_events_are_invariant_when_point_in_time_input_is_truncated():
    source = _industry_rows()
    full, _, _ = build_single_industry_top1_signals(
        source, start_date="2020-01-01"
    )
    cutoff = source["date"].drop_duplicates().sort_values().iloc[4]
    truncated, _, _ = build_single_industry_top1_signals(
        source[source["date"].le(cutoff)], start_date="2020-01-01"
    )
    event_columns = [
        "date",
        "signal_id",
        "triggered",
        "episode_id",
        "episode_stage",
        "event_onset",
        "event_continuation",
        "event_exit",
        "event_capped_confirmation",
    ]

    expected = full[
        full["date"].le(cutoff) & full["signal_id"].isin(truncated["signal_id"])
    ][event_columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, truncated[event_columns].reset_index(drop=True))


def test_rejects_an_observation_gap_inside_an_industry_lifespan():
    source = _industry_rows()
    gap_date = source["date"].drop_duplicates().sort_values().iloc[2]
    source = source[
        ~(source["industry_code"].eq("stable") & source["date"].eq(gap_date))
    ]

    with pytest.raises(ValueError, match="not continuous within its lifespan"):
        build_single_industry_top1_signals(source, start_date="2020-01-01")


def test_rejects_top1_flag_that_does_not_match_rank():
    source = _industry_rows()
    source.loc[source.index[0], "is_top1_ma20"] = True

    with pytest.raises(ValueError, match="does not match rank"):
        build_single_industry_top1_signals(source, start_date="2020-01-01")
