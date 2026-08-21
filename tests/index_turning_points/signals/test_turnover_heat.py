import numpy as np
import pandas as pd
import pytest

from research.index_turning_points.signals.definitions.turnover_heat import (
    build_turnover_heat_signal,
    causal_historical_midrank,
)


def _daily_rows(order_values):
    dates = pd.bdate_range("2020-01-01", periods=len(order_values))
    records = []
    for date, order_value in zip(dates, order_values):
        if pd.isna(order_value):
            records.append(
                {
                    "date": date,
                    "universe_size": 1200,
                    "turnover_valid_count": 0,
                    "turnover_cap_weight_valid_count": 0,
                    "turnover_ratio_pct_mean": np.nan,
                    "turnover_ratio_pct_p25": np.nan,
                    "turnover_ratio_pct_p50": np.nan,
                    "turnover_ratio_pct_p75": np.nan,
                    "turnover_ratio_pct_p90": np.nan,
                    "turnover_ratio_pct_p95": np.nan,
                    "turnover_ratio_pct_cap_weighted_mean": np.nan,
                    "turnover_ge_5pct_count": 0,
                    "turnover_ge_5pct_ratio": np.nan,
                    "turnover_ge_10pct_count": 0,
                    "turnover_ge_10pct_ratio": np.nan,
                    "turnover_ge_20pct_count": 0,
                    "turnover_ge_20pct_ratio": np.nan,
                }
            )
            continue

        value = float(order_value)
        ge_10_count = int(order_value)
        ge_5_count = min(1000, ge_10_count + 100)
        ge_20_count = max(0, ge_10_count - 100)
        records.append(
            {
                "date": date,
                "universe_size": 1200,
                "turnover_valid_count": 1000,
                "turnover_cap_weight_valid_count": 900,
                "turnover_ratio_pct_mean": value + 0.1,
                "turnover_ratio_pct_p25": max(0.0, value - 0.5),
                "turnover_ratio_pct_p50": value,
                "turnover_ratio_pct_p75": value + 0.5,
                "turnover_ratio_pct_p90": value + 1.0,
                "turnover_ratio_pct_p95": value + 1.5,
                "turnover_ratio_pct_cap_weighted_mean": value + 0.25,
                "turnover_ge_5pct_count": ge_5_count,
                "turnover_ge_5pct_ratio": ge_5_count / 1000,
                "turnover_ge_10pct_count": ge_10_count,
                "turnover_ge_10pct_ratio": ge_10_count / 1000,
                "turnover_ge_20pct_count": ge_20_count,
                "turnover_ge_20pct_ratio": ge_20_count / 1000,
            }
        )
    return pd.DataFrame(records)


def _reversal_values():
    return [*range(120), 200, 201, 202, 203, 204, 100, 101, 300]


def test_midrank_excludes_current_and_gives_half_weight_to_ties():
    values = pd.Series([1.0] * 121)

    ranked = causal_historical_midrank(values)

    assert ranked["rank"].iloc[:120].isna().all()
    assert ranked.loc[120, "history_count"] == 120
    assert ranked.loc[120, "rank"] == pytest.approx(0.5)


def test_midrank_uses_valid_history_inside_the_prior_250_trade_dates():
    values = pd.Series(
        [1.0, *([0.0] * 125), *([np.nan] * 10), *([0.0] * 115), 1.0, 0.5]
    )

    ranked = causal_historical_midrank(values)

    assert ranked.iloc[-1]["history_count"] == 240
    assert ranked.iloc[-1]["rank"] == pytest.approx(239 / 240)


def test_builds_top_reversal_and_starts_at_first_available_score():
    source = _daily_rows(_reversal_values())

    daily, episodes, metadata = build_turnover_heat_signal(
        source, start_date="2020-01-01"
    )

    assert daily["date"].iloc[0] == source["date"].iloc[120]
    assert daily.loc[daily["triggered"], "date"].tolist() == [
        source["date"].iloc[125],
        source["date"].iloc[126],
    ]
    assert daily["change_available"].iloc[:5].eq(False).all()
    assert daily["turnover_score_change_5d"].iloc[:5].isna().all()
    assert daily["raw_value"].equals(daily["turnover_score"])
    assert daily["valid_count"].eq(900).all()
    assert len(episodes) == 1
    assert episodes.iloc[0]["active_days"] == 2
    assert episodes.iloc[0]["capped_confirmation_reason"] == "nth_active_day"
    assert metadata["first_score_available_date"] == source[
        "date"
    ].iloc[120].strftime("%Y-%m-%d")
    assert metadata["first_change_available_date"] == source[
        "date"
    ].iloc[125].strftime("%Y-%m-%d")


def test_post_warmup_missing_component_is_retained_but_inactive():
    source = _daily_rows(_reversal_values())
    missing_date = source["date"].iloc[126]
    source.loc[126, "turnover_cap_weight_valid_count"] = 0
    source.loc[126, "turnover_ratio_pct_cap_weighted_mean"] = np.nan

    daily, _, metadata = build_turnover_heat_signal(
        source, start_date="2020-01-01"
    )

    row = daily[daily["date"].eq(missing_date)].iloc[0]
    assert not row["quality_available"]
    assert not row["change_available"]
    assert not row["triggered"]
    assert pd.isna(row["turnover_score"])
    assert row["valid_count"] == 0
    assert metadata["quality_unavailable_dates"] == 1


def test_daily_values_and_events_are_invariant_when_input_is_truncated():
    source = _daily_rows(_reversal_values())
    full, _, _ = build_turnover_heat_signal(source, start_date="2020-01-01")
    cutoff = source["date"].iloc[126]
    truncated, _, _ = build_turnover_heat_signal(
        source[source["date"].le(cutoff)], start_date="2020-01-01"
    )
    columns = [
        "date",
        "raw_value",
        "turnover_ratio_pct_p50_rank250",
        "turnover_ratio_pct_cap_weighted_mean_rank250",
        "turnover_ge_10pct_ratio_rank250",
        "turnover_score_change_5d",
        "quality_available",
        "change_available",
        "triggered",
        "episode_id",
        "episode_stage",
        "event_onset",
        "event_continuation",
        "event_exit",
        "event_capped_confirmation",
    ]

    expected = full[full["date"].le(cutoff)][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        expected,
        truncated[columns].reset_index(drop=True),
    )


def test_single_active_day_at_sample_tail_remains_pending():
    source = _daily_rows(_reversal_values()[:-2])

    daily, episodes, _ = build_turnover_heat_signal(
        source, start_date="2020-01-01"
    )

    assert daily["triggered"].iloc[-1]
    assert daily["event_onset"].iloc[-1]
    assert not daily["event_capped_confirmation"].iloc[-1]
    assert episodes.iloc[0]["status"] == "active"
    assert episodes.iloc[0]["confirmation_status"] == "pending"
    assert pd.isna(episodes.iloc[0]["capped_confirmation_date"])


def test_rejects_ratio_that_does_not_match_count():
    source = _daily_rows(_reversal_values())
    source.loc[0, "turnover_ge_10pct_ratio"] += 0.01

    with pytest.raises(ValueError, match="does not match"):
        build_turnover_heat_signal(source, start_date="2020-01-01")


def test_rejects_nonmonotonic_turnover_quantiles():
    source = _daily_rows(_reversal_values())
    source.loc[0, "turnover_ratio_pct_p25"] = 10.0

    with pytest.raises(ValueError, match="quantiles must be nondecreasing"):
        build_turnover_heat_signal(source, start_date="2020-01-01")
