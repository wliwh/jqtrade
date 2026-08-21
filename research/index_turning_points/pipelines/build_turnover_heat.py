"""Build the immutable stage-E all-A turnover-heat signal bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..signals.definitions.turnover_heat import (
    CHANGE_LOOKBACK,
    CHANGE_THRESHOLD,
    EXTREME_THRESHOLDS,
    HISTORY_WINDOW,
    MIN_HISTORY,
    RANK_COMPONENTS,
    REQUESTED_START_DATE,
    SCORE_THRESHOLD,
    SIGNAL_ID,
    SIGNAL_VERSION,
    build_turnover_heat_signal,
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
    """Validate the accepted V2 snapshot and write one immutable bundle."""

    input_dir = Path(input_dir)
    output_dir = require_empty_output_dir(output_dir)

    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    market, source_record = load_verified_frame(
        input_dir, input_manifest, DAILY_PATH
    )
    daily, episodes, comparison = build_turnover_heat_signal(
        market,
        version=signal_version,
        start_date=start_date,
    )

    outputs = write_signal_frames(output_dir, daily, episodes)
    manifest = {
        "signal_version": signal_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal all-A turnover-heat top signal for stage-C/D evaluation; "
            "not a trading strategy."
        ),
        "definition": {
            "signal_series": [
                {"signal_id": SIGNAL_ID, "direction": "top"}
            ],
            "rank_components": list(RANK_COMPONENTS),
            "rank_method": (
                "(< current + 0.5 * == current) / valid prior observations"
            ),
            "rank_history_window_trade_days": HISTORY_WINDOW,
            "rank_minimum_valid_history_days": MIN_HISTORY,
            "rank_excludes_current_date": True,
            "composite": "equal-weight mean of the three historical midranks",
            "change_lookback_trade_days": CHANGE_LOOKBACK,
            "top_trigger": (
                "turnover_score >= 0.75 and turnover_score 5-trade-day "
                "change <= -0.10"
            ),
            "score_threshold": SCORE_THRESHOLD,
            "change_threshold": CHANGE_THRESHOLD,
            "raw_value": "turnover_score",
            "valid_count": (
                "minimum of turnover_valid_count and "
                "turnover_cap_weight_valid_count"
            ),
            "comparison_start_policy": (
                "first date on which all three historical ranks are available"
            ),
            "missing_policy": (
                "retain post-warmup dates; any unavailable component makes "
                "quality_available false and cannot trigger"
            ),
            "audit_turnover_extreme_thresholds_pct": list(
                EXTREME_THRESHOLDS
            ),
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
                PROJECT_DIR / "signals" / "definitions" / "turnover_heat.py",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(outputs["signal_daily"], daily, output_dir),
            output_frame_record(
                outputs["signal_episodes"], episodes, output_dir
            ),
        ],
        "counts": {
            "signal_series": 1,
            "trade_dates": comparison["trade_dates"],
            "daily_rows": len(daily),
            "quality_available_dates": comparison[
                "quality_available_dates"
            ],
            "quality_unavailable_dates": comparison[
                "quality_unavailable_dates"
            ],
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
        raise ValueError("turnover heat requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    if tuple(query.get("turnover_extreme_thresholds_pct", [])) != tuple(
        float(value) for value in EXTREME_THRESHOLDS
    ):
        raise ValueError("input turnover extreme thresholds do not match signal")
    if tuple(query.get("valuation_fields", [])) != (
        "turnover_ratio",
        "circulating_market_cap",
    ):
        raise ValueError("input valuation fields do not match turnover signal")
    universe = query.get("universe", {})
    if universe.get("index") != "000985.XSHG":
        raise ValueError("input turnover universe must be all-A 000985.XSHG")
    export_level = manifest.get("export_level", {})
    if export_level.get("daily_market_features") != "one row per trade date":
        raise ValueError("input daily export level does not match signal")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-version", default=SIGNAL_VERSION)
    parser.add_argument(
        "--start-date",
        default=REQUESTED_START_DATE.strftime("%Y-%m-%d"),
    )
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
