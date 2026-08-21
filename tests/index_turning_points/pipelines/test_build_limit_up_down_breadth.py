import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import (
    build_limit_up_down_breadth as pipeline,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_limit_up_down_breadth_bundle(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert set(outputs) == {"signal_daily", "signal_episodes", "manifest"}
    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2012-07-05"
    assert daily["date"].max() == "2026-08-14"
    assert daily["signal_id"].nunique() == 2
    assert len(daily) == 6858
    assert daily.groupby("direction").head(5)["change_available"].eq(False).all()
    assert manifest["counts"]["triggered_days_by_direction"] == {
        "bottom": 23,
        "top": 53,
    }
    assert manifest["counts"]["episodes_by_direction"] == {
        "bottom": 23,
        "top": 52,
    }
    assert len(episodes) == 75
    rank_definition = manifest["definition"]["historical_rank"]
    assert rank_definition["strictly_excludes_current"] is True
    assert rank_definition["history_window_trade_days"] == 250
    assert rank_definition["minimum_valid_history_days"] == 120
    assert manifest["definition"]["capped_confirmation_n"] == 2
    assert manifest["inputs"]["data_version"] == "all_a_p1_inputs_v2"
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_rejects_nonempty_output_directory_without_overwriting(tmp_path):
    output_dir = tmp_path / "signal"
    output_dir.mkdir()
    existing = output_dir / "existing.txt"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert existing.read_text(encoding="utf-8") == "keep"
    assert sorted(path.name for path in output_dir.iterdir()) == ["existing.txt"]
