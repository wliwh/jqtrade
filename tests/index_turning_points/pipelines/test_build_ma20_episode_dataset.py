import hashlib
import json

import pandas as pd

from research.index_turning_points.modeling.ma20_episode_dataset import (
    MA20_EPISODE_DATASET_VERSION,
    MA20_EPISODE_FEATURE_COLUMNS,
)
from research.index_turning_points.pipelines import build_ma20_episode_dataset


def test_pipeline_builds_five_day_ma20_candidate_bundle(tmp_path):
    output_dir = tmp_path / "episode_dataset"

    outputs = build_ma20_episode_dataset.run_pipeline(
        build_ma20_episode_dataset.DEFAULT_SIGNAL_DIR,
        build_ma20_episode_dataset.DEFAULT_FEATURE_DATASET_DIR,
        build_ma20_episode_dataset.DEFAULT_GROUND_TRUTH_DIR,
        output_dir,
    )

    candidates = pd.read_csv(outputs["candidate_episodes"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["dataset_version"] == MA20_EPISODE_DATASET_VERSION
    assert manifest["definition"]["operational_window_trade_days"] == 5
    assert manifest["definition"]["feature_columns"] == list(
        MA20_EPISODE_FEATURE_COLUMNS
    )
    assert len(candidates) == 161
    assert candidates["operational_window_trade_days"].eq(5).all()
    assert candidates.groupby("direction")["target_operational_match"].sum().to_dict() == {
        "bottom": 23,
        "top": 19,
    }
    for record in manifest["outputs"]:
        path = output_dir / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
