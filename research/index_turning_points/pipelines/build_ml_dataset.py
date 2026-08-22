"""Build an immutable all-A ML daily dataset bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..adapters.tdx import read_tdx_daily, threshold_for_index
from ..modeling.dataset import (
    DATASET_VERSION,
    FUTURE_ENTRY_TARGET_MODE,
    MODEL_START_DATE,
    TODAY_DATASET_VERSION,
    TODAY_TARGET_MODE,
    build_all_a_today_training_daily,
    build_all_a_training_daily,
    feature_columns,
    today_feature_columns,
)
from ..modeling.targets import DEFAULT_HORIZONS
from .signal_bundle import (
    input_file_record,
    load_verified_frame,
    logic_records,
    output_frame_record,
    require_empty_output_dir,
    sha256_file,
    write_manifest,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "inputs"
    / "all_a_p1_inputs"
    / "all_a_p1_inputs_v2_20120101_20260814"
)
DEFAULT_GROUND_TRUTH_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "ground_truth"
    / "index_ohlc_20260814"
    / "regions"
    / "top_bottom_regions_v2"
)
DEFAULT_VIPDOC = Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc"
DAILY_PATH = "data/daily_market_features.csv"
ALL_A_TDX_PATH = "ds/lday/62#000985.day"
REGIONS_PATH = "turning_point_regions.csv"
LOBES_PATH = "turning_point_region_lobes.csv"
INPUT_DATA_VERSION = "all_a_p1_inputs_v2"
LABEL_VERSION = "top_bottom_regions_v2"


def run_pipeline(
    input_dir: Path | str,
    ground_truth_dir: Path | str,
    vipdoc: Path | str,
    output_dir: Path | str,
    *,
    dataset_version: str | None = None,
    start_date: str | pd.Timestamp = MODEL_START_DATE,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    target_mode: str = FUTURE_ENTRY_TARGET_MODE,
) -> dict[str, Path]:
    """Validate frozen inputs and write one auditable training-day bundle."""

    if target_mode not in {FUTURE_ENTRY_TARGET_MODE, TODAY_TARGET_MODE}:
        raise ValueError(f"unknown target_mode: {target_mode}")
    if dataset_version is None:
        dataset_version = (
            TODAY_DATASET_VERSION
            if target_mode == TODAY_TARGET_MODE
            else DATASET_VERSION
        )

    input_dir = Path(input_dir)
    ground_truth_dir = Path(ground_truth_dir)
    vipdoc = Path(vipdoc)
    output_dir = require_empty_output_dir(output_dir)

    input_manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))
    _validate_market_manifest(input_manifest)
    market, market_record = load_verified_frame(
        input_dir,
        input_manifest,
        DAILY_PATH,
        source_name=INPUT_DATA_VERSION,
    )

    ground_truth_manifest_path = ground_truth_dir / "manifest.json"
    ground_truth_manifest = json.loads(
        ground_truth_manifest_path.read_text(encoding="utf-8")
    )
    _validate_ground_truth_manifest(ground_truth_manifest)
    regions, regions_record = _load_ground_truth_frame(
        ground_truth_dir, ground_truth_manifest, REGIONS_PATH
    )
    lobes, lobes_record = _load_ground_truth_frame(
        ground_truth_dir, ground_truth_manifest, LOBES_PATH
    )
    prices, price_record = _load_all_a_prices(vipdoc, ground_truth_manifest)
    threshold = threshold_for_index("all_a", 0.10)
    if target_mode == TODAY_TARGET_MODE:
        training_daily = build_all_a_today_training_daily(
            market,
            prices,
            regions,
            lobes,
            threshold=threshold,
            start_date=start_date,
        )
        selected_feature_columns = today_feature_columns()
        purpose = (
            "All-A compact point-in-time features plus post-hoc current-day "
            "strict-lobe membership targets; not a fitted model or trading strategy."
        )
        definition = {
            "index_id": "all_a",
            "index_code": "000985.XSHG",
            "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "label_version": LABEL_VERSION,
            "target_mode": TODAY_TARGET_MODE,
            "probability_targets": {
                "top": "truth_top_in_strict_lobe",
                "bottom": "truth_bottom_in_strict_lobe",
            },
            "target_semantics": (
                "Binary membership of the current date in the direction's "
                "frozen strict lobe."
            ),
            "observation_time": "After the current trading day's close.",
            "intensity": (
                "Auxiliary post-hoc 0-100 proximity to each strict lobe's own "
                "representative high/low; never used as the probability target."
            ),
            "feature_columns": list(selected_feature_columns),
        }
    else:
        training_daily = build_all_a_training_daily(
            market,
            prices,
            regions,
            lobes,
            threshold=threshold,
            start_date=start_date,
            horizons=horizons,
        )
        selected_feature_columns = feature_columns()
        purpose = (
            "All-A point-in-time features plus post-hoc strict-lobe targets for "
            "the frozen first ML experiment; not a fitted model or trading strategy."
        )
        definition = {
            "index_id": "all_a",
            "index_code": "000985.XSHG",
            "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "label_version": LABEL_VERSION,
            "horizons_trade_days": list(horizons),
            "directional_change_threshold": threshold,
            "phase_state": (
                "Online pending/up/down maximum-tolerance state; never backfilled "
                "from the post-hoc Plotly phase background."
            ),
            "intensity": (
                "0-100 relative price distance from each strict lobe's own "
                "representative high/low; non-lobe dates are zero."
            ),
            "feature_columns": list(selected_feature_columns),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "training_daily": output_dir / "training_daily.csv",
        "manifest": output_dir / "manifest.json",
    }
    training_daily.to_csv(outputs["training_daily"], index=False)
    manifest = {
        "dataset_version": dataset_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": purpose,
        "definition": definition,
        "inputs": {
            "market_manifest": input_file_record(input_manifest_path),
            "ground_truth_manifest": input_file_record(
                ground_truth_manifest_path
            ),
            "source_files": [
                market_record,
                regions_record,
                lobes_record,
                price_record,
            ],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "modeling" / "targets.py",
                PROJECT_DIR / "modeling" / "features.py",
                PROJECT_DIR / "modeling" / "dataset.py",
                *(
                    [PROJECT_DIR / "docs" / "ml_today_probability_v1_spec.md"]
                    if target_mode == TODAY_TARGET_MODE
                    else []
                ),
                PROJECT_DIR / "ground_truth" / "labels.py",
                PROJECT_DIR / "adapters" / "tdx.py",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(
                outputs["training_daily"], training_daily, output_dir
            )
        ],
        "counts": {
            "trade_dates": len(training_daily),
            "index_price_available_dates": int(
                training_daily["index_price_available"].sum()
            ),
            "target_available_dates": int(training_daily["target_available"].sum()),
            "missing_index_dates": training_daily.loc[
                ~training_daily["index_price_available"], "date"
            ].dt.strftime("%Y-%m-%d").tolist(),
            "positive_intensity_days": {
                "top": int(training_daily["truth_top_intensity"].gt(0).sum()),
                "bottom": int(training_daily["truth_bottom_intensity"].gt(0).sum()),
            },
            "strict_lobe_days": {
                "top": int(
                    training_daily["truth_top_in_strict_lobe"]
                    .astype("boolean")
                    .fillna(False)
                    .sum()
                ),
                "bottom": int(
                    training_daily["truth_bottom_in_strict_lobe"]
                    .astype("boolean")
                    .fillna(False)
                    .sum()
                ),
            },
        },
    }
    if target_mode == FUTURE_ENTRY_TARGET_MODE:
        manifest["counts"]["phase_state_days"] = {
            str(state): int(count)
            for state, count in training_daily["index_phase_pti"]
            .value_counts()
            .sort_index()
            .items()
        }
        manifest["counts"]["complete_target_dates"] = {
            str(horizon): int(
                training_daily[f"target_complete_{horizon}d"]
                .astype("boolean")
                .fillna(False)
                .sum()
            )
            for horizon in horizons
        }
    write_manifest(outputs["manifest"], manifest)
    return outputs


def _validate_market_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("data_version") != INPUT_DATA_VERSION:
        raise ValueError(f"ML dataset requires {INPUT_DATA_VERSION}")
    query = manifest.get("query", {})
    if query.get("universe", {}).get("index") != "000985.XSHG":
        raise ValueError("ML dataset requires point-in-time all-A constituents")
    if tuple(query.get("ma_windows", [])) != (20, 60, 120):
        raise ValueError("ML dataset requires MA20/60/120 breadth")
    if tuple(query.get("high_low_windows", [])) != (60, 120, 250):
        raise ValueError("ML dataset requires 60/120/250-day high-low breadth")


def _validate_ground_truth_manifest(manifest: dict[str, object]) -> None:
    if manifest.get("label_version") != LABEL_VERSION:
        raise ValueError(f"ML dataset requires {LABEL_VERSION}")
    source = next(
        (
            record
            for record in manifest.get("source_files", [])
            if record.get("index_id") == "all_a"
        ),
        None,
    )
    if source is None or source.get("path") != ALL_A_TDX_PATH:
        raise ValueError("ground truth manifest is missing the all-A TDX source")


def _load_ground_truth_frame(
    ground_truth_dir: Path,
    manifest: dict[str, object],
    relative_path: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = next(
        (
            record
            for record in manifest.get("outputs", [])
            if Path(str(record.get("path"))).name == relative_path
        ),
        None,
    )
    if source is None:
        raise ValueError(f"ground truth is missing output: {relative_path}")
    path = ground_truth_dir / relative_path
    digest = sha256_file(path)
    if digest != source.get("sha256"):
        raise ValueError(f"ground truth hash mismatch: {relative_path}")
    frame = pd.read_csv(path, encoding=str(source.get("encoding", "utf-8")))
    if len(frame) != source.get("rows") or list(frame.columns) != source.get("columns"):
        raise ValueError(f"ground truth shape mismatch: {relative_path}")
    return frame, {
        "source": LABEL_VERSION,
        "path": str(source.get("path")),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": str(source.get("encoding", "utf-8")),
    }


def _load_all_a_prices(
    vipdoc: Path,
    ground_truth_manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = vipdoc / ALL_A_TDX_PATH
    digest = sha256_file(path)
    expected = next(
        record
        for record in ground_truth_manifest["source_files"]
        if record.get("index_id") == "all_a"
    )
    if digest != expected.get("sha256"):
        raise ValueError("all-A TDX hash differs from the ground-truth source")
    prices = read_tdx_daily(path, float_prices=True)
    return prices, {
        "source": "TDX all-A index daily file",
        "index_id": "all_a",
        "path": ALL_A_TDX_PATH,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(prices),
        "start_date": prices.index.min().strftime("%Y-%m-%d"),
        "end_date": prices.index.max().strftime("%Y-%m-%d"),
        "encoding": "TDX 32-byte float-price records",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR
    )
    parser.add_argument("--vipdoc", type=Path, default=DEFAULT_VIPDOC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version")
    parser.add_argument(
        "--target-mode",
        choices=(FUTURE_ENTRY_TARGET_MODE, TODAY_TARGET_MODE),
        default=FUTURE_ENTRY_TARGET_MODE,
    )
    parser.add_argument("--start-date", default=MODEL_START_DATE.strftime("%Y-%m-%d"))
    args = parser.parse_args()
    outputs = run_pipeline(
        args.input_dir,
        args.ground_truth_dir,
        args.vipdoc,
        args.output_dir,
        dataset_version=args.dataset_version,
        start_date=args.start_date,
        target_mode=args.target_mode,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
