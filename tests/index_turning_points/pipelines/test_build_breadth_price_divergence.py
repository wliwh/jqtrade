import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import (
    build_breadth_price_divergence as pipeline,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_breadth_price_divergence_bundle(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(
        pipeline.DEFAULT_INPUT_DIR,
        pipeline.DEFAULT_VIPDOC,
        output_dir,
    )

    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2012-01-04"
    assert daily["date"].max() == "2026-08-14"
    assert len(daily) == 3549
    assert manifest["comparison"]["first_available_date"] == "2012-04-06"
    assert manifest["comparison"]["missing_index_price_dates"] == [
        "2017-04-10",
        "2017-06-19",
    ]
    assert manifest["counts"]["comparison_available_dates"] == 3476
    assert manifest["counts"]["triggered_days"] == 531
    assert len(episodes) == manifest["counts"]["episodes"] == 140
    assert manifest["inputs"]["data_version"] == "all_a_p1_inputs_v2"
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "signal"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(
            pipeline.DEFAULT_INPUT_DIR,
            pipeline.DEFAULT_VIPDOC,
            output_dir,
        )
