import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import (
    build_multi_period_ma_breadth as pipeline,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_multi_period_ma_breadth_bundle(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert set(outputs) == {"signal_daily", "signal_episodes", "manifest"}
    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2012-01-04"
    assert daily["date"].max() == "2026-08-14"
    assert daily["signal_id"].nunique() == 2
    assert len(daily) == 7098
    assert manifest["counts"]["triggered_days_by_direction"] == {
        "bottom": 140,
        "top": 108,
    }
    assert manifest["counts"]["episodes_by_direction"] == {
        "bottom": 63,
        "top": 43,
    }
    assert len(episodes) == 106
    assert manifest["definition"]["change_lookback_trade_days"] == 5
    assert manifest["inputs"]["data_version"] == "all_a_p1_inputs_v2"
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "signal"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)
