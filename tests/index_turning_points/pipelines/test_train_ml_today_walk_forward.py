import json

import pandas as pd
import pytest

from research.index_turning_points.modeling.dataset import TODAY_DATASET_VERSION
from research.index_turning_points.modeling.today_training import (
    TODAY_TRAINING_VERSION,
    WalkForwardResult,
)
from research.index_turning_points.pipelines import train_ml_today_walk_forward
from research.index_turning_points.pipelines.signal_bundle import (
    output_frame_record,
    write_manifest,
)


def _dataset_bundle(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    frame = pd.DataFrame({"date": ["2020-01-02"]})
    path = dataset_dir / "training_daily.csv"
    frame.to_csv(path, index=False)
    write_manifest(
        dataset_dir / "manifest.json",
        {
            "dataset_version": TODAY_DATASET_VERSION,
            "outputs": [output_frame_record(path, frame, dataset_dir)],
        },
    )
    return dataset_dir


def _fake_result():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-02", "2019-01-03"]),
            "signal_id": ["ml_today_elastic_net"] * 2,
            "model_id": ["elastic_net"] * 2,
            "direction": ["top"] * 2,
            "version": [TODAY_TRAINING_VERSION] * 2,
            "test_year": [2019, 2019],
            "triggered": [False, True],
        }
    )
    return WalkForwardResult(
        signal_daily=daily,
        signal_episodes=pd.DataFrame({"episode_id": ["one"]}),
        probability_metrics=pd.DataFrame({"brier_score": [0.2]}),
        folds=pd.DataFrame({"fold_id": ["wf_2019"]}),
        thresholds=pd.DataFrame({"threshold": [50.0]}),
        fit_audit=pd.DataFrame({"model_fit_status": ["fitted"]}),
    )


def test_pipeline_writes_today_probability_bundle(tmp_path, monkeypatch):
    dataset_dir = _dataset_bundle(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        train_ml_today_walk_forward,
        "run_today_walk_forward_training",
        lambda *args, **kwargs: _fake_result(),
    )

    outputs = train_ml_today_walk_forward.run_pipeline(dataset_dir, output_dir)

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["training_version"] == TODAY_TRAINING_VERSION
    assert manifest["definition"]["target_semantics"] == (
        "Probability that the current date belongs to the direction's "
        "frozen strict lobe."
    )
    assert manifest["definition"]["score_formula"] == (
        "100 * pred_probability_today"
    )
    assert manifest["definition"]["feature_columns"]
    assert manifest["counts"]["oos_daily_rows"] == 2
    assert len(manifest["outputs"]) == 6
    with pytest.raises(FileExistsError):
        train_ml_today_walk_forward.run_pipeline(dataset_dir, output_dir)
