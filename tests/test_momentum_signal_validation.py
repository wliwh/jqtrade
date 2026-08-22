import inspect

import numpy as np
import pandas as pd
import pytest

import research.momentum_signal_validation.p1_jq_signal_validation as p1_module
from research.momentum_signal_validation.p1_jq_signal_validation import (
    build_config,
    build_forward_returns,
    compute_signal_panels,
    cross_sectional_ic,
    run_p1,
    summarize_series,
)


def _synthetic_close(periods=180, assets=6):
    dates = pd.bdate_range("2020-01-01", periods=periods)
    time = np.arange(periods, dtype=float)
    data = {}
    for number in range(assets):
        slope = 0.0005 + number * 0.00035
        cycle = 0.003 * np.sin(time / 13.0 + number)
        data["asset_%d" % number] = 100.0 * np.exp(slope * time + cycle)
    return pd.DataFrame(data, index=dates)


def test_signal_uses_only_prices_through_signal_date():
    close = _synthetic_close()
    changed = close.copy()
    cutoff = close.index[90]
    changed.loc[changed.index > cutoff] *= 7.0
    original = compute_signal_panels(close, (25,))
    revised = compute_signal_panels(changed, (25,))
    for name in ("formation_return", "annualized_slope", "r2", "slope_x_r2"):
        pd.testing.assert_series_equal(
            original[(name, 25)].loc[cutoff],
            revised[(name, 25)].loc[cutoff],
        )


def test_log_linear_prices_have_unit_r2_and_ordered_slope_scores():
    dates = pd.bdate_range("2020-01-01", periods=40)
    time = np.arange(40, dtype=float)
    close = pd.DataFrame(
        {
            "slow": 100.0 * np.exp(0.001 * time),
            "fast": 100.0 * np.exp(0.003 * time),
        },
        index=dates,
    )
    signals = compute_signal_panels(close, (25,))
    last = dates[-1]
    assert signals[("r2", 25)].loc[last, "slow"] == pytest.approx(1.0)
    assert signals[("r2", 25)].loc[last, "fast"] == pytest.approx(1.0)
    assert (
        signals[("slope_x_r2", 25)].loc[last, "fast"]
        > signals[("slope_x_r2", 25)].loc[last, "slow"]
    )


def test_forward_return_starts_at_signal_close_and_ends_h_bars_later():
    close = pd.DataFrame(
        {"asset": [100.0, 110.0, 121.0, 133.1]},
        index=pd.bdate_range("2020-01-01", periods=4),
    )
    outcomes = build_forward_returns(close, (1, 2))
    assert outcomes[1].iloc[0, 0] == pytest.approx(0.10)
    assert outcomes[2].iloc[0, 0] == pytest.approx(0.21)
    assert np.isnan(outcomes[2].iloc[-1, 0])


def test_rank_ic_is_positive_when_signal_and_outcome_order_match():
    dates = pd.bdate_range("2020-01-01", periods=2)
    columns = ["asset_%d" % number for number in range(6)]
    signal = pd.DataFrame([np.arange(6), np.arange(6)], index=dates, columns=columns)
    outcome = signal * 0.01
    daily = cross_sectional_ic(signal, outcome, min_assets=6)
    assert daily["rank_ic"].tolist() == pytest.approx([1.0, 1.0])
    assert daily["pearson_ic"].tolist() == pytest.approx([1.0, 1.0])


def test_hac_summary_keeps_descriptive_and_inference_fields():
    result = summarize_series(pd.Series([0.1, 0.2, 0.0, 0.3]), hac_lag=2)
    assert result["n"] == 4
    assert result["mean"] == pytest.approx(0.15)
    assert result["hac_lag"] == 2
    assert result["hac_standard_error"] >= 0.0
    assert 0.0 <= result["p_value"] <= 1.0


def test_active_p1_has_one_local_calculation_interface():
    assert list(inspect.signature(run_p1).parameters) == [
        "close_panels",
        "config",
        "verbose",
    ]
    assert not [name for name in dir(p1_module) if name.startswith("fetch_")]
    assert "jqdata" not in inspect.getsource(p1_module)


def test_progress_reporter_prints_percentage_stage_and_elapsed_time(capsys):
    progress = p1_module._build_progress_reporter(total_steps=2)
    progress("started", advance=False)
    progress("first stage")
    progress("finished")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert "0%" in lines[0] and "started" in lines[0]
    assert "50%" in lines[1] and "first stage" in lines[1]
    assert "100%" in lines[2] and "finished" in lines[2]


def test_end_to_end_local_run_returns_research_tables_only(capsys):
    base_close = _synthetic_close(periods=220, assets=6)
    close_panels = {}
    universes = {}
    for universe_group in ("broad", "industry_sw_l1", "style"):
        panel = base_close.copy()
        panel.columns = [
            "%s_%s" % (universe_group, column) for column in panel.columns
        ]
        close_panels[universe_group] = panel
        universes[universe_group] = {
            "version": "%s_test_v1" % universe_group,
            "members": {column: column for column in panel.columns},
            "min_assets": 6,
        }
    config = build_config(
        {
            "universes": universes,
            "research_start": "2020-01-01",
            "research_end": "2020-11-03",
            "periods": {
                "development": ("2020-01-01", "2020-05-31"),
                "validation": ("2020-06-01", "2020-08-31"),
                "locked_oos": ("2020-09-01", "2020-11-03"),
            },
            "lookbacks": (10, 25, 40),
            "horizons": (1, 5),
            "primary_lookback": 25,
            "primary_horizon": 5,
        }
    )
    results = run_p1(close_panels, config=config, verbose=True)
    output = capsys.readouterr().out
    assert set(results) == {
        "protocol",
        "universe_coverage",
        "ic_summary",
        "ic_daily",
        "parameter_plateau",
        "group_returns",
        "group_diagnostics",
        "topk_summary",
        "r2_double_sort",
        "r2_quality_spreads",
        "yearly_primary_ic",
    }
    assert set(results["ic_summary"]["universe_group"]) == {
        "broad",
        "industry_sw_l1",
        "style",
    }
    assert len(results["ic_summary"]) == 3 * 4 * 3 * 2 * 4 * 2 * 2
    assert set(results["ic_daily"]["signal"]) == {"slope_x_r2"}
    assert set(results["ic_daily"]["lookback"]) == {25}
    assert "[P1 100%" in output


def test_local_run_rejects_missing_or_extra_universe_groups():
    panel = _synthetic_close(periods=40, assets=6)
    with pytest.raises(ValueError, match="exactly match"):
        run_p1({"broad": panel}, verbose=False)
