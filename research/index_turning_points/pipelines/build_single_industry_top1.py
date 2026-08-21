"""Build the immutable stage-E single-industry MA20 Top1 signal bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..signals.definitions.single_industry_top1 import (
    MIN_INDUSTRY_VALID_COUNT,
    REQUESTED_START_DATE,
    SIGNAL_VERSION,
    build_single_industry_top1_signals,
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
INDUSTRY_PATH = "data/industry_breadth.csv"


def run_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    signal_version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> dict[str, Path]:
    """Validate the V2 industry snapshot and write one multi-series bundle."""

    input_dir = Path(input_dir)
    output_dir = require_empty_output_dir(output_dir)

    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    industry, source_record = load_verified_frame(
        input_dir, input_manifest, INDUSTRY_PATH
    )
    daily, episodes, comparison = build_single_industry_top1_signals(
        industry,
        version=signal_version,
        start_date=start_date,
    )

    outputs = write_signal_frames(output_dir, daily, episodes)
    manifest = {
        "signal_version": signal_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal point-in-time single-industry MA20 Top1 signals for "
            "stage-C/D evaluation; not a trading strategy."
        ),
        "definition": {
            "direction": "top",
            "rank_window": 20,
            "rank_method": "min descending; ties included",
            "min_industry_valid_count": MIN_INDUSTRY_VALID_COUNT,
            "trigger": "The individual industry is Top1 on that date.",
            "raw_value": "that industry's MA20 breadth",
            "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "industry_identity": "point-in-time industry_code + industry_name",
            "lifespan_policy": (
                "Each industry exists only from its first through last observed "
                "date, with continuous observations required inside that span."
            ),
            "substitution_policy": (
                "No current-industry backfill and no predecessor/successor merge."
            ),
            "capped_confirmation_n": 2,
        },
        "comparison": comparison,
        "inputs": {
            "manifest": input_file_record(manifest_path),
            "data_version": input_manifest["data_version"],
            "source_files": [source_record],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "signals" / "events.py",
                PROJECT_DIR
                / "signals"
                / "definitions"
                / "single_industry_top1.py",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(outputs["signal_daily"], daily, output_dir),
            output_frame_record(outputs["signal_episodes"], episodes, output_dir),
        ],
        "counts": {
            "industry_series": comparison["industry_count"],
            "trade_dates": comparison["trade_dates"],
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
        raise ValueError("single-industry signal requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    if query.get("industry_source") != "get_industry:sw_l1":
        raise ValueError("input must use point-in-time SW level-1 industries")
    if (
        query.get("industry_rank_window") != 20
        or query.get("industry_rank_method") != "min_descending"
        or query.get("min_industry_valid_count") != MIN_INDUSTRY_VALID_COUNT
    ):
        raise ValueError("input industry ranking protocol does not match signal")
    if (
        query.get("ma_comparison_relative_tolerance") != 1e-12
        or query.get("ma_comparison_absolute_tolerance") != 1e-12
    ):
        raise ValueError("input MA comparison tolerance does not match signal")
    export_level = manifest.get("export_level", {})
    if export_level.get("industry_breadth") != (
        "one row per trade date and observed SW level-1 industry"
    ):
        raise ValueError("input industry export level does not match signal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-version", default=SIGNAL_VERSION)
    parser.add_argument("--start-date", default=REQUESTED_START_DATE.strftime("%Y-%m-%d"))
    args = parser.parse_args()
    outputs = run_pipeline(
        args.input_dir,
        args.output_dir,
        signal_version=args.signal_version,
        start_date=args.start_date,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
