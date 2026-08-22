"""Train immutable multi-year calibrated all-A nowcast probability V2."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from ..modeling.alert_policy import (
    DEFAULT_COOLDOWN_DAYS,
    DEFAULT_EXIT_PROBABILITY,
    DEFAULT_MIN_ENTRY_PROBABILITY,
)
from ..modeling.alerts import DEFAULT_ANNUAL_EPISODE_BUDGET
from ..modeling.dataset import TODAY_DATASET_VERSION, today_feature_columns
from ..modeling.models import RANDOM_STATE
from ..modeling.today_calibrated_training import (
    CALIBRATION_BIN_COUNT,
    DEFAULT_CALIBRATION_YEAR_COUNT,
    DEFAULT_FIRST_TEST_YEAR,
    MIN_CALIBRATION_NEGATIVES,
    MIN_CALIBRATION_POSITIVES,
    TODAY_CALIBRATED_MODEL_IDS,
    TODAY_CALIBRATED_TRAINING_VERSION,
    run_calibrated_today_walk_forward_training,
)
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
TRAINING_DAILY_PATH = "training_daily.csv"


def run_pipeline(
    dataset_dir: Path | str,
    output_dir: Path | str,
    *,
    model_ids: tuple[str, ...] = TODAY_CALIBRATED_MODEL_IDS,
    first_test_year: int = DEFAULT_FIRST_TEST_YEAR,
    calibration_year_count: int = DEFAULT_CALIBRATION_YEAR_COUNT,
    boundary_gap: int = DEFAULT_BOUNDARY_GAP,
    annual_episode_budget: int = DEFAULT_ANNUAL_EPISODE_BUDGET,
    version: str = TODAY_CALIBRATED_TRAINING_VERSION,
) -> dict[str, Path]:
    """Verify V1 data and write V2 test-year probabilities and audits."""

    dataset_dir = Path(dataset_dir)
    output_dir = require_empty_output_dir(output_dir)
    dataset_manifest_path = dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("dataset_version") != TODAY_DATASET_VERSION:
        raise ValueError(
            f"calibrated current-day training requires {TODAY_DATASET_VERSION}"
        )
    training_daily, input_record = _load_training_daily(
        dataset_dir, dataset_manifest
    )
    result = run_calibrated_today_walk_forward_training(
        training_daily,
        model_ids=model_ids,
        first_test_year=first_test_year,
        calibration_year_count=calibration_year_count,
        boundary_gap=boundary_gap,
        annual_episode_budget=annual_episode_budget,
        version=version,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "signal_daily": (output_dir / "oos_signal_daily.csv", result.signal_daily),
        "signal_episodes": (
            output_dir / "oos_signal_episodes.csv",
            result.signal_episodes,
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

    manifest = {
        "training_version": version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Retrospective three-year-calibrated expanding walk-forward "
            "estimation of current strict-lobe membership, with a frozen "
            "hysteresis/cooldown alert policy; not a trading strategy."
        ),
        "definition": {
            "index_id": "all_a",
            "models": list(model_ids),
            "model_parameters": {
                "elastic_net": {
                    "estimator": "sklearn.linear_model.LogisticRegression",
                    "penalty": "elasticnet",
                    "solver": "saga",
                    "l1_ratio": 0.5,
                    "C": 0.05,
                    "class_weight": "balanced",
                    "max_iter": 3000,
                },
                "shallow_gbdt": {
                    "estimator": "sklearn.ensemble.GradientBoostingClassifier",
                    "n_estimators": 60,
                    "learning_rate": 0.03,
                    "max_depth": 2,
                    "min_samples_leaf": 20,
                    "subsample": 0.8,
                    "sample_weight": "balanced",
                },
                "random_state": RANDOM_STATE,
            },
            "runtime": {
                "python_executable": sys.executable,
                "python_version": platform.python_version(),
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "observation_time": "After the current trading day's close.",
            "target_semantics": (
                "Probability that the current date belongs to the direction's "
                "frozen strict lobe."
            ),
            "score_formula": "100 * pred_probability_today",
            "first_test_year": first_test_year,
            "calibration_year_count": calibration_year_count,
            "boundary_gap_trade_days": boundary_gap,
            "minimum_calibration_events": {
                "positive": MIN_CALIBRATION_POSITIVES,
                "negative": MIN_CALIBRATION_NEGATIVES,
            },
            "calibration": (
                "Validation-only three-year sigmoid when event minimums are "
                "met; otherwise ranking-preserving logit prior shift to the "
                "smoothed training prevalence and no formal alerts."
            ),
            "calibration_bin_count": CALIBRATION_BIN_COUNT,
            "annual_episode_budget": annual_episode_budget,
            "calibration_episode_budget": (
                annual_episode_budget * calibration_year_count
            ),
            "alert_policy": {
                "minimum_entry_probability": DEFAULT_MIN_ENTRY_PROBABILITY,
                "exit_probability": DEFAULT_EXIT_PROBABILITY,
                "cooldown_trade_days": DEFAULT_COOLDOWN_DAYS,
                "insufficient_calibration_alerts": False,
            },
            "feature_columns": list(today_feature_columns()),
        },
        "inputs": {
            "dataset_manifest": input_file_record(dataset_manifest_path),
            "source_files": [input_record],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "modeling" / "walk_forward.py",
                PROJECT_DIR / "modeling" / "models.py",
                PROJECT_DIR / "modeling" / "alerts.py",
                PROJECT_DIR / "modeling" / "alert_policy.py",
                PROJECT_DIR / "modeling" / "calibration.py",
                PROJECT_DIR / "modeling" / "training.py",
                PROJECT_DIR / "modeling" / "today_training.py",
                PROJECT_DIR / "modeling" / "today_calibrated_training.py",
                PROJECT_DIR / "signals" / "events.py",
                PROJECT_DIR / "docs" / "ml_today_probability_v2_spec.md",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(path, frame, output_dir)
            for path, frame in frames.values()
        ],
        "counts": {
            "folds": len(result.folds),
            "oos_daily_rows": len(result.signal_daily),
            "oos_episodes": len(result.signal_episodes),
            "metric_rows": len(result.probability_metrics),
            "calibration_reliability_rows": len(result.calibration_reliability),
            "threshold_rows": len(result.thresholds),
            "fit_rows": len(result.fit_audit),
            "test_years": sorted(
                int(value) for value in result.signal_daily["test_year"].unique()
            ),
            "probability_status_rows": {
                str(status): int(count)
                for status, count in result.signal_daily["probability_status"]
                .value_counts()
                .sort_index()
                .items()
            },
            "triggered_days": {
                f"{model_id}:{direction}": int(
                    result.signal_daily.loc[
                        result.signal_daily["model_id"].eq(model_id)
                        & result.signal_daily["direction"].eq(direction),
                        "triggered",
                    ].sum()
                )
                for model_id in model_ids
                for direction in ("top", "bottom")
            },
        },
    }
    write_manifest(outputs["manifest"], manifest)
    return outputs


def _load_training_daily(
    dataset_dir: Path,
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = next(
        (
            record
            for record in manifest.get("outputs", [])
            if str(record.get("path")) == TRAINING_DAILY_PATH
        ),
        None,
    )
    if source is None:
        raise ValueError("dataset manifest is missing training_daily.csv")
    path = dataset_dir / TRAINING_DAILY_PATH
    digest = sha256_file(path)
    if digest != source.get("sha256"):
        raise ValueError("training_daily.csv hash mismatch")
    frame = pd.read_csv(path, encoding=str(source.get("encoding", "utf-8")))
    if len(frame) != source.get("rows") or list(frame.columns) != source.get("columns"):
        raise ValueError("training_daily.csv shape mismatch")
    return frame, {
        "source": TODAY_DATASET_VERSION,
        "path": TRAINING_DAILY_PATH,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": str(source.get("encoding", "utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TODAY_CALIBRATED_MODEL_IDS,
        default=list(TODAY_CALIBRATED_MODEL_IDS),
    )
    parser.add_argument(
        "--training-version", default=TODAY_CALIBRATED_TRAINING_VERSION
    )
    args = parser.parse_args()
    outputs = run_pipeline(
        args.dataset_dir,
        args.output_dir,
        model_ids=tuple(args.models),
        version=args.training_version,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
