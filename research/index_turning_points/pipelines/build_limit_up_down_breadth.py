"""Build the immutable stage-E all-A limit-up/down breadth bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..signals.definitions.limit_up_down_breadth import (
    CHANGE_LOOKBACK,
    CHANGE_THRESHOLD,
    EXTREME_THRESHOLD,
    HISTORY_WINDOW,
    MIN_HISTORY,
    REQUESTED_START_DATE,
    SIGNAL_SPECS,
    SIGNAL_VERSION,
    build_limit_up_down_breadth_signals,
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
    daily, episodes, comparison = build_limit_up_down_breadth_signals(
        market,
        version=signal_version,
        start_date=start_date,
    )

    outputs = write_signal_frames(output_dir, daily, episodes)
    manifest = {
        "signal_version": signal_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal all-A limit-up versus limit-down breadth reversal signals "
            "for stage-C/D evaluation; not a trading strategy."
        ),
        "definition": {
            "signal_series": [
                {"signal_id": signal_id, "direction": direction}
                for signal_id, direction in SIGNAL_SPECS
            ],
            "components": [
                "limit_hit_net_ratio",
                "limit_close_net_ratio",
            ],
            "historical_rank": {
                "method": "midrank: (prior less + 0.5 * prior equal) / n",
                "strictly_excludes_current": True,
                "history_window_trade_days": HISTORY_WINDOW,
                "minimum_valid_history_days": MIN_HISTORY,
            },
            "composite": (
                "equal-weight mean of causal historical midranks for hit-net "
                "and close-net ratios"
            ),
            "change_lookback_trade_days": CHANGE_LOOKBACK,
            "top_trigger": (
                "limit_score >= 0.75 and five-trade-day change <= -0.10"
            ),
            "bottom_trigger": (
                "limit_score <= 0.25 and five-trade-day change >= 0.10"
            ),
            "extreme_threshold": EXTREME_THRESHOLD,
            "change_threshold": CHANGE_THRESHOLD,
            "raw_value": "limit_score",
            "valid_count": "valid_count_limit",
            "missing": (
                "keep the row, set quality/score/change unavailable as "
                "applicable, do not trigger, and exclude it from the rank "
                "denominator while retaining its place in the trading-day window"
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
                PROJECT_DIR
                / "signals"
                / "definitions"
                / "limit_up_down_breadth.py",
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
            "signal_series": len(SIGNAL_SPECS),
            "trade_dates": comparison["trade_dates"],
            "daily_rows": len(daily),
            "quality_available_dates": comparison["quality_available_dates"],
            "score_available_dates": comparison["score_available_dates"],
            "change_available_dates": comparison["change_available_dates"],
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
        raise ValueError("limit-up/down breadth requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    universe = query.get("universe", {})
    if universe.get("index") != "000985.XSHG":
        raise ValueError("input breadth universe must be all-A 000985.XSHG")
    if query.get("price_fq") != "pre":
        raise ValueError("input limit prices must use the pre-adjusted scale")
    price_fields = set(query.get("price_fields", []))
    if not {"close", "high", "low", "high_limit", "low_limit"}.issubset(
        price_fields
    ):
        raise ValueError("input snapshot is missing actual limit-price fields")
    feature_definitions = manifest.get("feature_definitions", {})
    if not feature_definitions.get("limit_hit"):
        raise ValueError("input snapshot is missing the limit-hit definition")
    if not feature_definitions.get("limit_close"):
        raise ValueError("input snapshot is missing the limit-close definition")
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
