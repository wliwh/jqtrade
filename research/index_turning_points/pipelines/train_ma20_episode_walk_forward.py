"""Train an immutable walk-forward filter for MA20 candidate episodes."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from ..modeling.episode_filter import (
    DEFAULT_ANNUAL_CANDIDATE_BUDGET,
    DEFAULT_MIN_MATCH_RECALL,
    DEFAULT_MIN_SELECTED_CANDIDATES,
)
from ..modeling.episode_targets import (
    OPERATIONAL_LABEL_VERSION,
    OPERATIONAL_WINDOW_TRADE_DAYS,
)
from ..modeling.ma20_episode_dataset import (
    MA20_EPISODE_DATASET_VERSION,
    ma20_episode_feature_columns,
)
from ..modeling.ma20_episode_training import (
    CALIBRATION_BIN_COUNT,
    FIRST_OOF_YEAR,
    FIRST_TEST_YEAR,
    MA20_EPISODE_MODEL_ID,
    MA20_EPISODE_TRAINING_VERSION,
    MIN_CALIBRATION_NEGATIVES,
    MIN_CALIBRATION_POSITIVES,
    run_ma20_episode_walk_forward_training,
)
from ..modeling.models import RANDOM_STATE
from ..modeling.walk_forward import DEFAULT_BOUNDARY_GAP
from .signal_bundle import (
    input_file_record,
    logic_records,
    output_frame_record,
    require_empty_output_dir,
    sha256_file,
    write_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "modeling"
    / "all_a_ma20_episode_dataset_v1_20120705_20260814"
)
CANDIDATE_EPISODES_PATH = "candidate_episodes.csv"
DAILY_CALENDAR_PATH = "daily_calendar.csv"


def run_pipeline(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    first_oof_year: int = FIRST_OOF_YEAR,
    first_test_year: int = FIRST_TEST_YEAR,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
    annual_candidate_budget: int = DEFAULT_ANNUAL_CANDIDATE_BUDGET,
    version: str = MA20_EPISODE_TRAINING_VERSION,
) -> dict[str, Path]:
    """Verify the candidate bundle and write OOS probabilities and alerts."""

    dataset_dir = Path(dataset_dir)
    output_dir = require_empty_output_dir(output_dir)
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("dataset_version") != MA20_EPISODE_DATASET_VERSION:
        raise ValueError(f"episode training requires {MA20_EPISODE_DATASET_VERSION}")
    candidates, candidate_record = _load_dataset_frame(
        dataset_dir, dataset_manifest, CANDIDATE_EPISODES_PATH
    )
    calendar, calendar_record = _load_dataset_frame(
        dataset_dir, dataset_manifest, DAILY_CALENDAR_PATH
    )
    result = run_ma20_episode_walk_forward_training(
        candidates,
        calendar,
        first_oof_year=first_oof_year,
        first_test_year=first_test_year,
        boundary_gap=boundary_gap,
        annual_candidate_budget=annual_candidate_budget,
        version=version,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "signal_daily": (output_dir / "oos_signal_daily.csv", result.signal_daily),
        "signal_episodes": (
            output_dir / "oos_signal_episodes.csv",
            result.signal_episodes,
        ),
        "candidate_predictions": (
            output_dir / "candidate_predictions.csv",
            result.candidate_predictions,
        ),
        "probability_metrics": (
            output_dir / "probability_metrics.csv",
            result.probability_metrics,
        ),
        "calibration_reliability": (
            output_dir / "calibration_reliability.csv",
            result.calibration_reliability,
        ),
        "folds": (output_dir / "folds.csv", result.folds),
        "thresholds": (output_dir / "thresholds.csv", result.thresholds),
        "fit_audit": (output_dir / "fit_audit.csv", result.fit_audit),
    }
    outputs = {name: path for name, (path, _frame) in frames.items()}
    outputs["manifest"] = output_dir / "manifest.json"
    for path, frame in frames.values():
        frame.to_csv(path, index=False)

    predictions = result.candidate_predictions
    manifest = {
        "training_version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Expanding walk-forward probability that an existing MA20 candidate "
            "episode hits the frozen operational top/bottom region, followed by "
            "a causal candidate filter; not a daily top probability or strategy."
        ),
        "definition": {
            "index_id": "all_a",
            "model_id": MA20_EPISODE_MODEL_ID,
            "model_parameters": {
                "estimator": "sklearn.linear_model.LogisticRegression",
                "penalty": "l2",
                "C": 0.1,
                "solver": "lbfgs",
                "class_weight": None,
                "max_iter": 2000,
                "preprocessing": ["median_imputer", "standard_scaler"],
                "random_state": RANDOM_STATE,
            },
            "runtime": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "observation_time": "After the MA20 candidate onset day's close.",
            "probability_output": "pred_probability_episode_match",
            "score_formula": "100 * pred_probability_episode_match",
            "target": "target_operational_match",
            "target_semantics": (
                "Conditional on an MA20 candidate: one-to-one same-direction "
                "strict-core or +/-5-trading-day anchor match."
            ),
            "operational_label_version": OPERATIONAL_LABEL_VERSION,
            "operational_window_trade_days": OPERATIONAL_WINDOW_TRADE_DAYS,
            "first_oof_year": first_oof_year,
            "first_test_year": first_test_year,
            "boundary_gap_trade_days": boundary_gap,
            "minimum_calibration_events": {
                "positive": MIN_CALIBRATION_POSITIVES,
                "negative": MIN_CALIBRATION_NEGATIVES,
            },
            "calibration": (
                "Historical causal OOF sigmoid when event minimums are met; "
                "otherwise prior shift and pass through every MA20 candidate."
            ),
            "calibration_bin_count": CALIBRATION_BIN_COUNT,
            "filter_policy": {
                "annual_candidate_budget": annual_candidate_budget,
                "minimum_selected_candidates": DEFAULT_MIN_SELECTED_CANDIDATES,
                "minimum_match_recall": DEFAULT_MIN_MATCH_RECALL,
                "objective": "maximum 95% Wilson precision lower bound",
                "no_feasible_threshold": "pass through all MA20 candidates",
                "alert_duration": "one onset trading day",
            },
            "feature_columns": list(ma20_episode_feature_columns()),
        },
        "inputs": {
            "dataset_manifest": input_file_record(dataset_manifest_path),
            "source_files": [candidate_record, calendar_record],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "modeling" / "models.py",
                PROJECT_DIR / "modeling" / "calibration.py",
                PROJECT_DIR / "modeling" / "episode_filter.py",
                PROJECT_DIR / "modeling" / "ma20_episode_training.py",
                PROJECT_DIR / "signals" / "events.py",
                PROJECT_DIR / "docs" / "ma20_episode_ml_v1_spec.md",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(path, frame, output_dir)
            for path, frame in frames.values()
        ],
        "counts": {
            "candidate_predictions": len(predictions),
            "oos_daily_rows": len(result.signal_daily),
            "oos_episodes": len(result.signal_episodes),
            "metric_rows": len(result.probability_metrics),
            "calibration_reliability_rows": len(result.calibration_reliability),
            "fold_rows": len(result.folds),
            "threshold_rows": len(result.thresholds),
            "fit_rows": len(result.fit_audit),
            "test_years": sorted(int(value) for value in predictions["test_year"].unique()),
            "candidate_predictions_by_direction": {
                direction: int(predictions["direction"].eq(direction).sum())
                for direction in ("top", "bottom")
            },
            "retained_alerts_by_direction": {
                direction: int(
                    predictions.loc[
                        predictions["direction"].eq(direction), "final_alert"
                    ].sum()
                )
                for direction in ("top", "bottom")
            },
        },
    }
    write_manifest(outputs["manifest"], manifest)
    return outputs


def _load_dataset_frame(
    dataset_dir: Path,
    manifest: dict[str, object],
    relative_path: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    record = next(
        (
            value
            for value in manifest.get("outputs", [])
            if str(value.get("path")) == relative_path
        ),
        None,
    )
    if record is None:
        raise ValueError(f"dataset manifest is missing {relative_path}")
    path = dataset_dir / relative_path
    digest = sha256_file(path)
    if digest != record.get("sha256"):
        raise ValueError(f"{relative_path} hash mismatch")
    encoding = str(record.get("encoding", "utf-8"))
    frame = pd.read_csv(path, encoding=encoding)
    if len(frame) != record.get("rows") or list(frame.columns) != record.get("columns"):
        raise ValueError(f"{relative_path} shape mismatch")
    return frame, {
        "source": MA20_EPISODE_DATASET_VERSION,
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": encoding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = run_pipeline(args.dataset_dir, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
