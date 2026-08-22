import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.modeling.dataset import feature_columns
from research.index_turning_points.pipelines import build_ml_dataset as pipeline


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_all_a_ml_daily_bundle(tmp_path):
    output_dir = tmp_path / "modeling"

    outputs = pipeline.run_pipeline(
        pipeline.DEFAULT_INPUT_DIR,
        pipeline.DEFAULT_GROUND_TRUTH_DIR,
        pipeline.DEFAULT_VIPDOC,
        output_dir,
    )

    daily = pd.read_csv(outputs["training_daily"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2012-07-05"
    assert daily["date"].max() == "2026-08-14"
    assert len(daily) == 3429
    assert set(feature_columns()).issubset(daily.columns)
    assert not any("triggered" in column for column in daily.columns)
    assert set(daily["index_phase_pti"].dropna().unique()) == {
        "pending",
        "up",
        "down",
    }
    assert daily["truth_top_intensity"].max() == 100.0
    assert daily["truth_bottom_intensity"].max() == 100.0
    assert manifest["dataset_version"] == "all_a_ml_dataset_v1"
    assert manifest["definition"]["directional_change_threshold"] == 0.10
    assert manifest["counts"]["target_available_dates"] < len(daily)
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "modeling"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(
            pipeline.DEFAULT_INPUT_DIR,
            pipeline.DEFAULT_GROUND_TRUTH_DIR,
            pipeline.DEFAULT_VIPDOC,
            output_dir,
        )
