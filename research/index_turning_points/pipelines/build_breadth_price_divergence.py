"""Build the immutable stage-E all-A breadth-price divergence bundle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..adapters.tdx import read_tdx_daily
from ..signals.definitions.breadth_price_divergence import (
    DIVERGENCE_THRESHOLD,
    MAX_MISSING_PRICE_DATES,
    MIN_PRICE_OBSERVATIONS,
    PRICE_NEAR_HIGH_THRESHOLD,
    REQUESTED_START_DATE,
    ROLLING_HIGH_WINDOW,
    SIGNAL_VERSION,
    build_breadth_price_divergence_signal,
)
from .signal_bundle import (
    input_file_record,
    load_verified_frame,
    logic_records,
    output_frame_record,
    require_empty_output_dir,
    sha256_file,
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
DEFAULT_VIPDOC = Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc"
INPUT_DATA_VERSION = "all_a_p1_inputs_v2"
DAILY_PATH = "data/daily_market_features.csv"
ALL_A_TDX_PATH = "ds/lday/62#000985.day"
EXPECTED_MISSING_PRICE_DATES = ("2017-04-10", "2017-06-19")


def run_pipeline(
    input_dir: Path | str,
    vipdoc: Path | str,
    output_dir: Path | str,
    *,
    signal_version: str = SIGNAL_VERSION,
    start_date: str | pd.Timestamp = REQUESTED_START_DATE,
) -> dict[str, Path]:
    """Validate the breadth snapshot and TDX all-A close, then write a bundle."""

    input_dir = Path(input_dir)
    vipdoc = Path(vipdoc)
    output_dir = require_empty_output_dir(output_dir)

    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    market, source_record = load_verified_frame(
        input_dir,
        input_manifest,
        DAILY_PATH,
        source_name="all_a_p1_inputs_v2",
    )
    prices, price_record = _load_all_a_prices(vipdoc)
    daily, episodes, comparison = build_breadth_price_divergence_signal(
        market,
        prices,
        version=signal_version,
        start_date=start_date,
    )
    if tuple(comparison["missing_index_price_dates"]) != EXPECTED_MISSING_PRICE_DATES:
        raise ValueError(
            "TDX all-A missing-date set changed; create a new signal version: "
            f"{comparison['missing_index_price_dates']}"
        )

    outputs = write_signal_frames(output_dir, daily, episodes)
    manifest = {
        "signal_version": signal_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Causal all-A MA20 breadth-versus-index-price divergence for "
            "stage-C/D evaluation; not a trading strategy."
        ),
        "definition": {
            "direction": "top",
            "rolling_high_window_trade_days": ROLLING_HIGH_WINDOW,
            "minimum_index_price_observations": MIN_PRICE_OBSERVATIONS,
            "price_near_high_threshold": PRICE_NEAR_HIGH_THRESHOLD,
            "divergence_threshold": DIVERGENCE_THRESHOLD,
            "trigger": (
                "all-A close is within 2% of its 60-day high and MA20 breadth "
                "relative high distance exceeds the price distance by at least 20%"
            ),
            "raw_value": "breadth high distance minus price high distance",
            "missing_price_policy": (
                "No interpolation; missing current close is inactive and each "
                "60-row window requires at least 59 observed closes."
            ),
            "maximum_missing_price_dates": MAX_MISSING_PRICE_DATES,
            "expected_missing_price_dates": list(EXPECTED_MISSING_PRICE_DATES),
            "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "capped_confirmation_n": 2,
        },
        "comparison": comparison,
        "inputs": {
            "breadth_manifest": input_file_record(manifest_path),
            "data_version": input_manifest["data_version"],
            "source_files": [source_record, price_record],
        },
        "logic": logic_records(
            [
                PROJECT_DIR / "signals" / "events.py",
                PROJECT_DIR
                / "signals"
                / "definitions"
                / "breadth_price_divergence.py",
                PROJECT_DIR / "adapters" / "tdx.py",
                Path(__file__),
            ]
        ),
        "outputs": [
            output_frame_record(outputs["signal_daily"], daily, output_dir),
            output_frame_record(outputs["signal_episodes"], episodes, output_dir),
        ],
        "counts": {
            "trade_dates": comparison["trade_dates"],
            "comparison_available_dates": comparison[
                "comparison_available_dates"
            ],
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
        raise ValueError("breadth-price divergence requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    if 20 not in query.get("ma_windows", []):
        raise ValueError("input snapshot does not contain MA20 breadth")
    if (
        query.get("ma_comparison_relative_tolerance") != 1e-12
        or query.get("ma_comparison_absolute_tolerance") != 1e-12
    ):
        raise ValueError("input MA comparison tolerance does not match signal")
    universe = query.get("universe", {})
    if universe.get("index") != "000985.XSHG":
        raise ValueError("input breadth universe must be all-A 000985.XSHG")


def _load_all_a_prices(vipdoc: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    path = vipdoc / ALL_A_TDX_PATH
    digest = sha256_file(path)
    frame = read_tdx_daily(path, float_prices=True).reset_index()
    prices = frame[["date", "close"]].copy()
    return prices, {
        "source": "TDX all-A index daily file",
        "index_id": "all_a",
        "path": ALL_A_TDX_PATH,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(prices),
        "start_date": prices["date"].min().strftime("%Y-%m-%d"),
        "end_date": prices["date"].max().strftime("%Y-%m-%d"),
        "encoding": "TDX 32-byte float-price records",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--vipdoc", type=Path, default=DEFAULT_VIPDOC)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--signal-version", default=SIGNAL_VERSION)
    parser.add_argument("--start-date", default=REQUESTED_START_DATE.strftime("%Y-%m-%d"))
    args = parser.parse_args()
    outputs = run_pipeline(
        args.input_dir,
        args.vipdoc,
        args.output_dir,
        signal_version=args.signal_version,
        start_date=args.start_date,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
