import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import (
    build_new_high_low_period_decomposition as pipeline,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_six_series_bundle(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2012-01-04"
    assert daily["date"].max() == "2026-08-14"
    assert daily["signal_id"].nunique() == 6
    assert len(daily) == 21294
    expected_counts = {
        (60, "top"): (176, 98),
        (60, "bottom"): (97, 73),
        (120, "top"): (103, 48),
        (120, "bottom"): (54, 43),
        (250, "top"): (61, 24),
        (250, "bottom"): (32, 27),
    }
    actual_counts = {
        (record["high_low_window"], record["direction"]): (
            record["triggered_days"],
            record["episodes"],
        )
        for record in manifest["counts"]["series_counts"]
    }
    assert actual_counts == expected_counts
    assert manifest["counts"]["triggered_days"] == 523
    assert len(episodes) == manifest["counts"]["episodes"] == 313
    assert manifest["inputs"]["data_version"] == "all_a_p1_inputs_v2"
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "signal"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)
