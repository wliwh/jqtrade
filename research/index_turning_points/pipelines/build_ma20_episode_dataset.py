"""Build an immutable one-row-per-MA20-candidate episode dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..ground_truth.regions import DEFAULT_REGION_PROTOCOL
from ..modeling.episode_targets import (
    OPERATIONAL_LABEL_VERSION,
    OPERATIONAL_WINDOW_TRADE_DAYS,
)
from ..modeling.ma20_episode_dataset import (
    MA20_EPISODE_DATASET_VERSION,
    MA20_SIGNAL_IDS,
    build_ma20_episode_dataset,
    ma20_episode_feature_columns,
)
from .signal_bundle import (
    input_file_record,
    logic_records,
    output_frame_record,
    require_empty_output_dir,
    sha256_file,
    write_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "signals"
    / "ma_period_breadth_decomposition_v1_20120104_20260814"
)
DEFAULT_FEATURE_DATASET_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "modeling"
    / "all_a_ml_today_dataset_v1_20120705_20260814"
)
DEFAULT_GROUND_TRUTH_DIR = (
    PROJECT_DIR / "artifacts" / "ground_truth" / "index_ohlc_20260814"
)
SIGNAL_VERSION = "ma_period_breadth_decomposition_v1_20120104_20260814"
FEATURE_DATASET_VERSION = "all_a_ml_today_dataset_v1"
SIGNAL_DAILY_PATH = "signal_daily.csv"
SIGNAL_EPISODES_PATH = "signal_episodes.csv"
FEATURE_DAILY_PATH = "training_daily.csv"
REGIONS_PATH = "turning_point_regions.csv"
LOBES_PATH = "turning_point_region_lobes.csv"


def run_pipeline(
    signal_dir: Path | str,
    feature_dataset_dir: Path | str,
    ground_truth_dir: Path | str,
    output_dir: Path | str,
    *,
    dataset_version: str = MA20_EPISODE_DATASET_VERSION,
) -> dict[str, Path]:
    """Verify frozen inputs and write candidate episodes plus their calendar."""

    signal_dir = Path(signal_dir)
    feature_dataset_dir = Path(feature_dataset_dir)
    ground_truth_dir = Path(ground_truth_dir)
    output_dir = require_empty_output_dir(output_dir)

    signal_manifest_path = signal_dir / "manifest.json"
    signal_manifest = _read_manifest(signal_manifest_path)
    if signal_manifest.get("signal_version") != SIGNAL_VERSION:
        raise ValueError(f"episode dataset requires {SIGNAL_VERSION}")
    signal_daily, signal_daily_record = _load_output(
        signal_dir, signal_manifest, SIGNAL_DAILY_PATH, source=SIGNAL_VERSION
    )
    signal_episodes, signal_episodes_record = _load_output(
        signal_dir, signal_manifest, SIGNAL_EPISODES_PATH, source=SIGNAL_VERSION
    )

    feature_manifest_path = feature_dataset_dir / "manifest.json"
    feature_manifest = _read_manifest(feature_manifest_path)
    if feature_manifest.get("dataset_version") != FEATURE_DATASET_VERSION:
        raise ValueError(f"episode dataset requires {FEATURE_DATASET_VERSION}")
    feature_daily, feature_record = _load_output(
        feature_dataset_dir,
        feature_manifest,
        FEATURE_DAILY_PATH,
        source=FEATURE_DATASET_VERSION,
    )

    region_dir = ground_truth_dir / "regions" / DEFAULT_REGION_PROTOCOL.label_version
    ground_truth_manifest_path = region_dir / "manifest.json"
    ground_truth_manifest = _read_manifest(ground_truth_manifest_path)
    if ground_truth_manifest.get("label_version") != DEFAULT_REGION_PROTOCOL.label_version:
        raise ValueError("ground-truth label version does not match the frozen protocol")
    regions, regions_record = _load_output(
        ground_truth_dir,
        ground_truth_manifest,
        REGIONS_PATH,
        source=DEFAULT_REGION_PROTOCOL.label_version,
        match_by_name=True,
    )
    lobes, lobes_record = _load_output(
        ground_truth_dir,
        ground_truth_manifest,
        LOBES_PATH,
        source=DEFAULT_REGION_PROTOCOL.label_version,
        match_by_name=True,
    )

    result = build_ma20_episode_dataset(
        signal_daily,
        signal_episodes,
        feature_daily,
        regions,
        lobes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "candidate_episodes": (
            output_dir / "candidate_episodes.csv",
            result.candidate_episodes,
        ),
        "daily_calendar": (output_dir / "daily_calendar.csv", result.daily_calendar),
    }
    outputs = {name: path for name, (path, _frame) in frames.items()}
    outputs["manifest"] = output_dir / "manifest.json"
    for path, frame in frames.values():
        frame.to_csv(path, index=False)

    candidates = result.candidate_episodes
    manifest = {
        "dataset_version": dataset_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "One row per causal MA20 breadth candidate episode with close-time "
            "features and post-hoc episode-match labels; not a trading strategy."
        ),
        "definition": {
            "index_id": "all_a",
            "candidate_signal_ids": MA20_SIGNAL_IDS,
            "observation_time": "After the candidate onset trading day's close.",
            "target": "target_operational_match",
            "target_semantics": (
                "A one-to-one same-direction match when the candidate onset is "
                "inside a strict core lobe or within +/-5 trading days of the anchor."
            ),
            "operational_label_version": OPERATIONAL_LABEL_VERSION,
            "operational_window_trade_days": OPERATIONAL_WINDOW_TRADE_DAYS,
            "legacy_audit_targets": [
                "target_legacy_window_20d_match",
                "target_legacy_loose_match",
                "target_legacy_strict_match",
            ],
            "feature_columns": list(ma20_episode_feature_columns()),
        },
        "inputs": {
            "signal_manifest": input_file_record(signal_manifest_path),
            "feature_dataset_manifest": input_file_record(feature_manifest_path),
            "ground_truth_manifest": input_file_record(ground_truth_manifest_path),
            "source_files": [
                signal_daily_record,
                signal_episodes_record,
                feature_record,
                regions_record,
                lobes_record,
            ],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "evaluation" / "region_matching.py",
                PROJECT_DIR / "modeling" / "episode_targets.py",
                PROJECT_DIR / "modeling" / "ma20_episode_dataset.py",
                PROJECT_DIR / "docs" / "ma20_episode_ml_v1_spec.md",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(path, frame, output_dir)
            for path, frame in frames.values()
        ],
        "counts": {
            "calendar_dates": len(result.daily_calendar),
            "candidate_episodes": len(candidates),
            "by_direction": {
                direction: int(candidates["direction"].eq(direction).sum())
                for direction in ("top", "bottom")
            },
            "operational_matches": {
                direction: int(
                    candidates.loc[
                        candidates["direction"].eq(direction),
                        "target_operational_match",
                    ].sum()
                )
                for direction in ("top", "bottom")
            },
            "legacy_window_20d_matches": {
                direction: int(
                    candidates.loc[
                        candidates["direction"].eq(direction),
                        "target_legacy_window_20d_match",
                    ].sum()
                )
                for direction in ("top", "bottom")
            },
        },
    }
    write_manifest(outputs["manifest"], manifest)
    return outputs


def _read_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_output(
    base_dir: Path,
    manifest: dict[str, object],
    relative_path: str,
    *,
    source: str,
    match_by_name: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    record = next(
        (
            value
            for value in manifest.get("outputs", [])
            if (
                Path(str(value.get("path"))).name == relative_path
                if match_by_name
                else str(value.get("path")) == relative_path
            )
        ),
        None,
    )
    if record is None:
        raise ValueError(f"manifest is missing output: {relative_path}")
    manifest_path = Path(str(record["path"]))
    path = base_dir / manifest_path
    digest = sha256_file(path)
    if digest != record.get("sha256"):
        raise ValueError(f"input hash mismatch: {manifest_path.as_posix()}")
    encoding = str(record.get("encoding", "utf-8"))
    frame = pd.read_csv(path, encoding=encoding)
    if len(frame) != record.get("rows") or list(frame.columns) != record.get("columns"):
        raise ValueError(f"input shape mismatch: {manifest_path.as_posix()}")
    return frame, {
        "source": source,
        "path": manifest_path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": encoding,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-dir", type=Path, default=DEFAULT_SIGNAL_DIR)
    parser.add_argument(
        "--feature-dataset-dir", type=Path, default=DEFAULT_FEATURE_DATASET_DIR
    )
    parser.add_argument(
        "--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    outputs = run_pipeline(
        args.signal_dir,
        args.feature_dataset_dir,
        args.ground_truth_dir,
        args.output_dir,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
