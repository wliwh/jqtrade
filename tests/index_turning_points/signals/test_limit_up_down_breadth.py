import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.limit_up_down_breadth import (
    _causal_historical_midrank,
    build_limit_up_down_breadth_signals,
)


def _limit_rows(values, *, start="2020-01-01", valid_count=1000):
    records = []
    for date, value in zip(pd.bdate_range(start, periods=len(values)), values):
        net_count = int(round(value * valid_count))
        up_hit = max(net_count, 0)
        down_hit = max(-net_count, 0)
        up_close = up_hit
        down_close = down_hit
        records.append(
            {
                "date": date,
                "universe_size": valid_count + 100,
                "valid_count_limit": valid_count,
                "limit_price_missing_count": 3,
                "limit_up_hit_count": up_hit,
                "limit_down_hit_count": down_hit,
                "limit_hit_net_count": net_count,
                "limit_up_hit_ratio": up_hit / valid_count,
                "limit_down_hit_ratio": down_hit / valid_count,
                "limit_hit_net_ratio": net_count / valid_count,
                "limit_up_close_count": up_close,
                "limit_down_close_count": down_close,
                "limit_close_net_count": net_count,
                "limit_up_close_ratio": up_close / valid_count,
                "limit_down_close_ratio": down_close / valid_count,
                "limit_close_net_ratio": net_count / valid_count,
            }
        )
    return pd.DataFrame(records)


def _top_reversal_rows(*, include_exit=True):
    values = [-0.60 + 0.01 * index for index in range(120)]
    values.extend([0.58, 0.0, 0.0, 0.0, 0.0, 0.35])
    if include_exit:
        values.append(0.0)
    return _limit_rows(values)


def _bottom_reversal_rows(*, include_exit=True):
    values = [-0.59 + 0.01 * index for index in range(120)]
    values.extend([-0.58, 0.0, 0.0, 0.0, 0.0, -0.35])
    if include_exit:
        values.append(0.0)
    return _limit_rows(values)


def test_midrank_strictly_excludes_current_and_reaches_boundaries():
    values = pd.Series([0.0] * 120 + [-1.0, 0.0, 1.0])

    ranks, history_counts = _causal_historical_midrank(
        values,
        history_window=250,
        min_history=120,
    )

    assert ranks.iloc[:120].isna().all()
    assert ranks.iloc[120] == 0.0
    assert ranks.iloc[121] == pytest.approx(61 / 121)
    assert ranks.iloc[122] == 1.0
    assert history_counts.iloc[120:].tolist() == [120, 121, 122]


def test_midrank_uses_prior_250_trade_days_and_excludes_missing_from_denominator():
    values = pd.Series([0.0] * 250 + [np.nan, 1.0])

    ranks, history_counts = _causal_historical_midrank(
        values,
        history_window=250,
        min_history=120,
    )

    assert pd.isna(ranks.iloc[250])
    assert history_counts.iloc[250:].tolist() == [250, 249]
    assert ranks.iloc[251] == 1.0


def test_warmup_is_excluded_and_top_reversal_uses_five_trade_day_change():
    source = _top_reversal_rows()

    daily, episodes, metadata = build_limit_up_down_breadth_signals(
        source,
        start_date="2020-01-01",
    )

    top = daily[daily["direction"].eq("top")].reset_index(drop=True)
    assert top["date"].iloc[0] == source["date"].iloc[120]
    assert top["change_available"].iloc[:5].eq(False).all()
    assert not top["triggered"].iloc[:5].any()
    assert top.loc[top["triggered"], "date"].tolist() == [
        source["date"].iloc[125]
    ]
    assert top.loc[5, "limit_score"] >= 0.75
    assert top.loc[5, "limit_score_change_5d"] <= -0.10
    assert top["raw_value"].equals(top["limit_score"])
    assert len(episodes) == 1
    assert episodes.iloc[0]["direction"] == "top"
    assert metadata["comparison_start_date"] == source["date"].iloc[120].strftime(
        "%Y-%m-%d"
    )


def test_bottom_reversal_uses_symmetric_thresholds():
    source = _bottom_reversal_rows()

    daily, episodes, _ = build_limit_up_down_breadth_signals(
        source,
        start_date="2020-01-01",
    )

    bottom = daily[daily["direction"].eq("bottom")].reset_index(drop=True)
    assert bottom.loc[bottom["triggered"], "date"].tolist() == [
        source["date"].iloc[125]
    ]
    assert bottom.loc[5, "limit_score"] <= 0.25
    assert bottom.loc[5, "limit_score_change_5d"] >= 0.10
    assert len(episodes) == 1
    assert episodes.iloc[0]["direction"] == "bottom"


def test_missing_feature_row_is_kept_inactive_and_not_added_to_history():
    source = _top_reversal_rows()
    missing_position = 123
    source.loc[missing_position, "valid_count_limit"] = 0
    source.loc[missing_position, "limit_price_missing_count"] = source.loc[
        missing_position, "universe_size"
    ]
    count_columns = [column for column in source if column.endswith("_count")]
    ratio_columns = [column for column in source if column.endswith("_ratio")]
    source.loc[missing_position, count_columns] = 0
    source.loc[missing_position, ratio_columns] = np.nan

    daily, _, _ = build_limit_up_down_breadth_signals(
        source,
        start_date="2020-01-01",
    )

    row = daily[
        daily["date"].eq(source["date"].iloc[missing_position])
        & daily["direction"].eq("top")
    ].iloc[0]
    assert not row["quality_available"]
    assert not row["score_available"]
    assert not row["change_available"]
    assert not row["triggered"]
    later = daily[
        daily["date"].eq(source["date"].iloc[missing_position + 1])
        & daily["direction"].eq("top")
    ].iloc[0]
    assert later["limit_rank_history_count"] == 123


def test_daily_scores_and_events_are_invariant_to_sample_truncation():
    source = _top_reversal_rows()
    full, _, _ = build_limit_up_down_breadth_signals(
        source,
        start_date="2020-01-01",
    )
    causal_columns = [
        "date",
        "signal_id",
        "direction",
        "raw_value",
        "limit_hit_net_rank250",
        "limit_close_net_rank250",
        "limit_score_change_5d",
        "triggered",
        "episode_id",
        "episode_stage",
        "event_onset",
        "event_continuation",
        "event_exit",
        "event_capped_confirmation",
    ]

    for length in range(121, len(source) + 1):
        truncated, _, _ = build_limit_up_down_breadth_signals(
            source.iloc[:length],
            start_date="2020-01-01",
        )
        cutoff = source["date"].iloc[length - 1]
        expected = full[full["date"].le(cutoff)][causal_columns].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(
            truncated[causal_columns].reset_index(drop=True),
            expected,
        )


def test_sample_tail_single_day_episode_stays_pending_without_backfill():
    source = _top_reversal_rows(include_exit=False)

    daily, episodes, _ = build_limit_up_down_breadth_signals(
        source,
        start_date="2020-01-01",
    )

    top = episodes[episodes["direction"].eq("top")].iloc[0]
    assert top["active_days"] == 1
    assert top["status"] == "active"
    assert top["confirmation_status"] == "pending"
    assert pd.isna(top["capped_confirmation_date"])
    assert not daily["event_capped_confirmation"].any()


@pytest.mark.parametrize(
    "change,message",
    [
        ({"limit_hit_net_count": 1}, "does not match"),
        ({"limit_up_hit_ratio": 0.123}, "does not match"),
        ({"limit_hit_net_ratio": np.nan}, "finite values"),
        ({"limit_hit_net_ratio": "not-a-number"}, "non-numeric"),
        ({"valid_count_limit": 1101}, "must not exceed universe_size"),
    ],
)
def test_rejects_internally_inconsistent_limit_inputs(change, message):
    source = _top_reversal_rows()
    for column, value in change.items():
        source.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        build_limit_up_down_breadth_signals(source, start_date="2020-01-01")
