import json

import pandas as pd
import pytest

from research.index_turning_points.modeling.dataset import TODAY_DATASET_VERSION
from research.index_turning_points.modeling.today_calibrated_training import (
    TODAY_CALIBRATED_TRAINING_VERSION,
    CalibratedWalkForwardResult,
)
from research.index_turning_points.pipelines import (
    train_ml_today_calibrated_walk_forward,
)
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
            "signal_id": ["ml_today_calibrated_elastic_net"] * 2,
            "model_id": ["elastic_net"] * 2,
            "direction": ["top"] * 2,
            "version": [TODAY_CALIBRATED_TRAINING_VERSION] * 2,
            "test_year": [2019, 2019],
            "probability_status": ["calibrated"] * 2,
            "triggered": [False, True],
        }
    )
    return CalibratedWalkForwardResult(
        signal_daily=daily,
        signal_episodes=pd.DataFrame({"episode_id": ["one"]}),
        probability_metrics=pd.DataFrame(
            {"brier_score": [0.2], "expected_calibration_error": [0.1]}
        ),
        calibration_reliability=pd.DataFrame(
            {"bin_number": [0], "rows": [2]}
        ),
        folds=pd.DataFrame({"fold_id": ["wf_cal3_2019"]}),
        thresholds=pd.DataFrame({"entry_probability": [0.5]}),
        fit_audit=pd.DataFrame({"model_fit_status": ["fitted"]}),
    )


def test_pipeline_writes_v2_calibration_bundle(tmp_path, monkeypatch):
    dataset_dir = _dataset_bundle(tmp_path)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        train_ml_today_calibrated_walk_forward,
        "run_calibrated_today_walk_forward_training",
        lambda *args, **kwargs: _fake_result(),
    )

    outputs = train_ml_today_calibrated_walk_forward.run_pipeline(
        dataset_dir, output_dir
    )

    assert "calibration_reliability" in outputs
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["training_version"] == TODAY_CALIBRATED_TRAINING_VERSION
    assert manifest["definition"]["calibration_year_count"] == 3
    assert manifest["definition"]["minimum_calibration_events"] == {
        "positive": 5,
        "negative": 30,
    }
    assert manifest["definition"]["alert_policy"] == {
        "minimum_entry_probability": 0.5,
        "exit_probability": 0.3,
        "cooldown_trade_days": 10,
        "insufficient_calibration_alerts": False,
    }
    assert len(manifest["outputs"]) == 7
    with pytest.raises(FileExistsError):
        train_ml_today_calibrated_walk_forward.run_pipeline(
            dataset_dir, output_dir
        )
