import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import build_four_industry_top1 as pipeline


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_v2_four_industry_signal_bundle(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert set(outputs) == {"signal_daily", "signal_episodes", "manifest"}
    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert len(daily) == 1133
    assert len(episodes) == 105
    assert daily["date"].iloc[0] == "2021-12-13"
    assert daily["date"].iloc[-1] == "2026-08-14"
    assert daily["triggered"].sum() == 472
    assert daily["valid_count"].eq(4).all()
    assert manifest["comparison"]["target_start_dates"] == {
        "bank": "2014-02-21",
        "coal": "2021-12-13",
        "nonferrous": "2012-01-04",
        "steel": "2012-01-04",
    }
    assert manifest["definition"]["substitution_policy"] == (
        "No predecessor or substitute industry."
    )
    assert manifest["inputs"]["data_version"] == "all_a_p1_inputs_v2"
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])


def test_pipeline_rejects_nonempty_output_directory(tmp_path):
    output_dir = tmp_path / "signal"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)
