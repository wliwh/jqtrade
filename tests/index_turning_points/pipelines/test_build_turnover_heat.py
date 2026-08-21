import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import build_turnover_heat as pipeline


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_turnover_heat_bundle_from_first_score_date(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert set(outputs) == {"signal_daily", "signal_episodes", "manifest"}
    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2012-07-05"
    assert daily["date"].max() == "2026-08-14"
    assert daily["signal_id"].unique().tolist() == [
        "all_a_turnover_heat_top"
    ]
    assert daily["direction"].unique().tolist() == ["top"]
    assert len(daily) == 3429
    assert len(episodes) == 51
    assert manifest["signal_version"] == "turnover_heat_v1_20120705_20260814"
    assert manifest["comparison"]["history_start_date"] == "2012-01-04"
    assert manifest["comparison"]["comparison_start_date"] == "2012-07-05"
    assert manifest["comparison"]["first_change_available_date"] == "2012-07-12"
    assert manifest["definition"]["rank_history_window_trade_days"] == 250
    assert manifest["definition"]["rank_minimum_valid_history_days"] == 120
    assert manifest["definition"]["rank_excludes_current_date"] is True
    assert manifest["definition"]["capped_confirmation_n"] == 2
    assert manifest["counts"]["triggered_days"] == 70
    assert manifest["counts"]["episodes"] == 51
    assert manifest["inputs"]["data_version"] == "all_a_p1_inputs_v2"
    assert daily["change_available"].iloc[:5].eq(False).all()
    for column in (
        "turnover_ratio_pct_mean",
        "turnover_ratio_pct_p25",
        "turnover_ratio_pct_p75",
        "turnover_ratio_pct_p90",
        "turnover_ratio_pct_p95",
        "turnover_ge_5pct_ratio",
        "turnover_ge_10pct_ratio",
        "turnover_ge_20pct_ratio",
    ):
        assert column in daily.columns
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_refuses_to_overwrite_nonempty_bundle(tmp_path):
    output_dir = tmp_path / "signal"
    output_dir.mkdir()
    sentinel = output_dir / "existing.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep"
