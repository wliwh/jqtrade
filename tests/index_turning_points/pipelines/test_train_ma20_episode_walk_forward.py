import json

import pandas as pd

from research.index_turning_points.modeling.ma20_episode_dataset import (
    MA20_EPISODE_DATASET_VERSION,
)
from research.index_turning_points.modeling.ma20_episode_training import (
    MA20_EPISODE_TRAINING_VERSION,
    Ma20EpisodeWalkForwardResult,
)
from research.index_turning_points.pipelines import train_ma20_episode_walk_forward
from research.index_turning_points.pipelines.signal_bundle import (
    output_frame_record,
    write_manifest,
)


def _dataset_bundle(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    candidates = pd.DataFrame(
        {"candidate_episode_id": ["one"], "onset_date": ["2019-01-02"]}
    )
    calendar = pd.DataFrame(
        {"date": ["2019-01-02"], "universe_size": [1], "valid_count": [1]}
    )
    candidate_path = dataset_dir / "candidate_episodes.csv"
    calendar_path = dataset_dir / "daily_calendar.csv"
    candidates.to_csv(candidate_path, index=False)
    calendar.to_csv(calendar_path, index=False)
    write_manifest(
        dataset_dir / "manifest.json",
        {
            "dataset_version": MA20_EPISODE_DATASET_VERSION,
            "outputs": [
                output_frame_record(candidate_path, candidates, dataset_dir),
                output_frame_record(calendar_path, calendar, dataset_dir),
            ],
        },
    )
    return dataset_dir


def _fake_result():
    predictions = pd.DataFrame(
        {
            "candidate_episode_id": ["one", "two"],
            "test_year": [2019, 2019],
            "direction": ["top", "bottom"],
            "final_alert": [True, False],
        }
    )
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2019-01-02", "2019-01-02"]),
            "signal_id": ["ma20_episode_ml_l2_logistic"] * 2,
            "direction": ["top", "bottom"],
            "version": [MA20_EPISODE_TRAINING_VERSION] * 2,
            "test_year": [2019, 2019],
            "triggered": [True, False],
        }
    )
    return Ma20EpisodeWalkForwardResult(
        signal_daily=daily,
        signal_episodes=pd.DataFrame({"episode_id": ["episode_one"]}),
        candidate_predictions=predictions,
        probability_metrics=pd.DataFrame({"brier_score": [0.2]}),
        calibration_reliability=pd.DataFrame({"bin_number": [0], "rows": [2]}),
        folds=pd.DataFrame({"fold_id": ["episode_wf_2019"]}),
        thresholds=pd.DataFrame({"threshold_probability": [0.3]}),
        fit_audit=pd.DataFrame({"model_fit_status": ["fitted"]}),
    )


def test_pipeline_writes_episode_probability_and_filter_audits(tmp_path, monkeypatch):
    dataset_dir = _dataset_bundle(tmp_path)
    output_dir = tmp_path / "training"
    monkeypatch.setattr(
        train_ma20_episode_walk_forward,
        "run_ma20_episode_walk_forward_training",
        lambda *args, **kwargs: _fake_result(),
    )

    outputs = train_ma20_episode_walk_forward.run_pipeline(dataset_dir, output_dir)

    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert manifest["training_version"] == MA20_EPISODE_TRAINING_VERSION
    assert manifest["definition"]["operational_window_trade_days"] == 5
    assert manifest["definition"]["probability_output"] == (
        "pred_probability_episode_match"
    )
    assert manifest["definition"]["filter_policy"]["minimum_match_recall"] == 0.6
    assert len(manifest["outputs"]) == 8
    assert "candidate_predictions" in outputs
