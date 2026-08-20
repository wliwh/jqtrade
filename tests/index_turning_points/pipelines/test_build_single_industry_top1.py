import hashlib
import json

import pandas as pd
import pytest

from research.index_turning_points.pipelines import build_single_industry_top1 as pipeline


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pipeline_builds_point_in_time_single_industry_bundle(tmp_path):
    output_dir = tmp_path / "signal"

    outputs = pipeline.run_pipeline(pipeline.DEFAULT_INPUT_DIR, output_dir)

    assert set(outputs) == {"signal_daily", "signal_episodes", "manifest"}
    daily = pd.read_csv(outputs["signal_daily"])
    episodes = pd.read_csv(outputs["signal_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert daily["date"].min() == "2017-01-03"
    assert daily["date"].max() == "2026-08-14"
    assert daily["signal_id"].nunique() == 32
    assert len(daily) == 68779
    assert set(daily.loc[daily["date"].eq("2021-12-10"), "industry_name"]) == {
        item["industry_name"]
        for item in manifest["comparison"]["industry_coverage"]
        if item["start_date"] == "2017-01-03"
    }
    assert "采掘I" in set(daily.loc[daily["date"].eq("2021-12-10"), "industry_name"])
    assert "采掘I" not in set(daily.loc[daily["date"].eq("2021-12-13"), "industry_name"])
    assert "煤炭I" in set(daily.loc[daily["date"].eq("2021-12-13"), "industry_name"])
    assert [
        era["industry_count"]
        for era in manifest["comparison"]["industry_set_eras"]
    ] == [28, 31]
    assert len(episodes) == manifest["counts"]["episodes"]
    assert manifest["definition"]["substitution_policy"].startswith(
        "No current-industry backfill"
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
