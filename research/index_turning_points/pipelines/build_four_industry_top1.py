"""Build the immutable stage-E four-industry Top1 signal bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from ..signals.definitions.four_industry_top1 import (
    MIN_INDUSTRY_VALID_COUNT,
    SIGNAL_VERSION,
    TARGET_NAMES,
    build_four_industry_top1_signal,
)
from .signal_bundle import (
    input_file_record,
    load_verified_frame,
    logic_records,
    output_frame_record,
    require_empty_output_dir,
    write_manifest,
    write_signal_frames,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "inputs"
    / "all_a_p1_inputs"
    / "all_a_p1_inputs_v2_20120101_20260814"
)
INPUT_DATA_VERSION = "all_a_p1_inputs_v2"
DAILY_PATH = "data/daily_market_features.csv"
INDUSTRY_PATH = "data/industry_breadth.csv"


def run_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    signal_version: str = SIGNAL_VERSION,
) -> dict[str, Path]:
    """Validate the V2 snapshot and write daily, episode and manifest files."""

    input_dir = Path(input_dir)
    output_dir = require_empty_output_dir(output_dir)
    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    market, market_record = load_verified_frame(
        input_dir, input_manifest, DAILY_PATH
    )
    _, industry_record = load_verified_frame(
        input_dir, input_manifest, INDUSTRY_PATH
    )
    daily, episodes, comparison = build_four_industry_top1_signal(
        market,
        version=signal_version,
    )

    outputs = write_signal_frames(output_dir, daily, episodes)
    manifest = {
        "signal_version": signal_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal four-industry MA20 Top1 historical baseline for stage-C/D "
            "evaluation; not a trading strategy."
        ),
        "definition": {
            "direction": "top",
            "target_names": TARGET_NAMES,
            "rank_window": 20,
            "rank_method": "min descending; ties included",
            "min_industry_valid_count": MIN_INDUSTRY_VALID_COUNT,
            "trigger": (
                "At least one target is Top1 among all rank-eligible point-in-time "
                "SW level-1 industries."
            ),
            "raw_value": "maximum MA20 breadth among the four targets",
            "start_policy": (
                "Each industry starts on its first comparable date; a cohort "
                "starts on the latest member start date."
            ),
            "substitution_policy": "No predecessor or substitute industry.",
            "capped_confirmation_n": 2,
        },
        "comparison": comparison,
        "inputs": {
            "manifest": input_file_record(manifest_path),
            "data_version": input_manifest["data_version"],
            "source_files": [market_record, industry_record],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "signals" / "events.py",
                PROJECT_DIR
                / "signals"
                / "definitions"
                / "four_industry_top1.py",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(outputs["signal_daily"], daily, output_dir),
            output_frame_record(outputs["signal_episodes"], episodes, output_dir),
        ],
        "counts": {
            "daily_rows": len(daily),
            "triggered_days": int(daily["triggered"].sum()),
            "episodes": len(episodes),
            "onsets": int(daily["event_onset"].sum()),
            "capped_confirmations": int(
                daily["event_capped_confirmation"].sum()
            ),
        },
    }
    write_manifest(outputs["manifest"], manifest)
    return outputs


def _validate_input_protocol(manifest: dict[str, object]) -> None:
    if manifest.get("data_version") != INPUT_DATA_VERSION:
        raise ValueError("four-industry signal requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    expected_targets = {
        "bank": "银行",
        "nonferrous": "有色金属",
        "steel": "钢铁",
        "coal": "煤炭",
    }
    if query.get("targets") != expected_targets:
        raise ValueError("input target industries do not match frozen definition")
    if query.get("old_mining_mapped_to_coal") is not False:
        raise ValueError("input must not map old mining to coal")
    if (
        query.get("industry_rank_window") != 20
        or query.get("industry_rank_method") != "min_descending"
        or query.get("min_industry_valid_count") != MIN_INDUSTRY_VALID_COUNT
    ):
        raise ValueError("input industry ranking protocol does not match signal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-version", default=SIGNAL_VERSION)
    args = parser.parse_args()
    outputs = run_pipeline(
        args.input_dir,
        args.output_dir,
        signal_version=args.signal_version,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
