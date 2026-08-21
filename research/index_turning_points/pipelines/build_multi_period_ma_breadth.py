"""Build the immutable stage-E multi-period MA breadth signal bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..signals.definitions.multi_period_ma_breadth import (
    CHANGE_LOOKBACK,
    CHANGE_THRESHOLD,
    EXTREME_THRESHOLD,
    MA_WINDOWS,
    REQUESTED_START_DATE,
    SIGNAL_SPECS,
    SIGNAL_VERSION,
    build_multi_period_ma_breadth_signals,
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


def run_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    signal_version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> dict[str, Path]:
    """Validate the V2 daily snapshot and write the two-series bundle."""

    input_dir = Path(input_dir)
    output_dir = require_empty_output_dir(output_dir)

    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    market, source_record = load_verified_frame(
        input_dir, input_manifest, DAILY_PATH
    )
    daily, episodes, comparison = build_multi_period_ma_breadth_signals(
        market,
        version=signal_version,
        start_date=start_date,
    )

    outputs = write_signal_frames(output_dir, daily, episodes)
    manifest = {
        "signal_version": signal_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal all-A MA20/60/120 breadth reversal signals for stage-C/D "
            "evaluation; not a trading strategy."
        ),
        "definition": {
            "signal_series": [
                {"signal_id": signal_id, "direction": direction}
                for signal_id, direction in SIGNAL_SPECS
            ],
            "ma_windows": list(MA_WINDOWS),
            "composite": "equal-weight mean of MA20/60/120 breadth",
            "change_lookback_trade_days": CHANGE_LOOKBACK,
            "top_trigger": (
                "composite >= 0.70 and composite change over 5 trade days <= -0.05"
            ),
            "bottom_trigger": (
                "composite <= 0.30 and composite change over 5 trade days >= 0.05"
            ),
            "extreme_threshold": EXTREME_THRESHOLD,
            "change_threshold": CHANGE_THRESHOLD,
            "raw_value": "equal-weight MA20/60/120 breadth composite",
            "valid_count": "minimum of the three MA valid counts",
            "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
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
                / "multi_period_ma_breadth.py",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(outputs["signal_daily"], daily, output_dir),
            output_frame_record(outputs["signal_episodes"], episodes, output_dir),
        ],
        "counts": {
            "signal_series": len(SIGNAL_SPECS),
            "trade_dates": comparison["trade_dates"],
            "daily_rows": len(daily),
            "triggered_days": int(daily["triggered"].sum()),
            "episodes": len(episodes),
            "onsets": int(daily["event_onset"].sum()),
            "capped_confirmations": int(
                daily["event_capped_confirmation"].sum()
            ),
            "triggered_days_by_direction": comparison[
                "triggered_days_by_direction"
            ],
            "episodes_by_direction": comparison["episodes_by_direction"],
        },
    }
    write_manifest(outputs["manifest"], manifest)
    return outputs


def _validate_input_protocol(manifest: dict[str, object]) -> None:
    if manifest.get("data_version") != INPUT_DATA_VERSION:
        raise ValueError("multi-period MA breadth requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    if tuple(query.get("ma_windows", [])) != MA_WINDOWS:
        raise ValueError("input MA windows do not match signal")
    if (
        query.get("ma_comparison_relative_tolerance") != 1e-12
        or query.get("ma_comparison_absolute_tolerance") != 1e-12
    ):
        raise ValueError("input MA comparison tolerance does not match signal")
    export_level = manifest.get("export_level", {})
    if export_level.get("daily_market_features") != "one row per trade date":
        raise ValueError("input daily export level does not match signal")


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
