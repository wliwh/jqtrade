"""Render the base viewer or an explicit historical-signal variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ..adapters.legacy_four_industry_v1 import (
    DEFAULT_SIGNAL_PATH,
    load_four_industry_v1_signal,
)
from ..visualization import viewer
from ..visualization.overlays.four_industry import (
    FOUR_INDUSTRY_PAGE,
    add_four_industry_signal,
)
from .signal_bundle import load_verified_frame, sha256_file


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "viewers"
    / "top_bottom_regions_ma20_v1"
    / "index_turning_points_ma20.html"
)
DEFAULT_MA20_INPUT_DIR = (
    PROJECT_DIR
    / "data"
    / "inputs"
    / "all_a_p1_inputs"
    / "all_a_p1_inputs_v2_20120101_20260814"
)
MA20_INPUT_DATA_VERSION = "all_a_p1_inputs_v2"
MA20_DAILY_PATH = "data/daily_market_features.csv"
DEFAULT_MA20_SIGNAL_DIR = (
    PROJECT_DIR
    / "artifacts"
    / "signals"
    / "ma_period_breadth_decomposition_v1_20120104_20260814"
)
MA20_SIGNAL_VERSION = "ma_period_breadth_decomposition_v1_20120104_20260814"
MA20_SIGNAL_DAILY_PATH = "signal_daily.csv"
DEFAULT_FOUR_INDUSTRY_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "viewers"
    / "four_industry_top1_v1"
    / "four_industry_top1_v1__top_bottom_regions_v2.html"
)


def render_base_viewer(
    vipdoc: Path | str,
    output_path: Path | str = DEFAULT_BASE_OUTPUT,
    threshold: float = 0.10,
    *,
    ma20_input_dir: Path | str = DEFAULT_MA20_INPUT_DIR,
    ma20_signal_dir: Path | str = DEFAULT_MA20_SIGNAL_DIR,
) -> Path:
    """Render ground truth above the verified full-A MA20 breadth series."""

    ma20_breadth = load_ma20_breadth(ma20_input_dir)
    ma20_signals = load_ma20_signals(ma20_signal_dir)
    return viewer.write_viewer(
        vipdoc,
        output_path,
        threshold,
        ma20_breadth=ma20_breadth,
        ma20_signals=ma20_signals,
    )


def load_ma20_breadth(input_dir: Path | str) -> pd.DataFrame:
    """Load the immutable full-A daily snapshot after manifest validation."""

    input_dir = Path(input_dir)
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_version") != MA20_INPUT_DATA_VERSION:
        raise ValueError("MA20 viewer requires all_a_p1_inputs_v2")
    export_level = manifest.get("export_level", {})
    if export_level.get("daily_market_features") != "one row per trade date":
        raise ValueError("MA20 viewer requires one daily market row per trade date")
    frame, _ = load_verified_frame(input_dir, manifest, MA20_DAILY_PATH)
    return frame


def load_ma20_signals(signal_dir: Path | str) -> pd.DataFrame:
    """Load the immutable MA-period signal output after manifest validation."""

    signal_dir = Path(signal_dir)
    manifest_path = signal_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signal_version") != MA20_SIGNAL_VERSION:
        raise ValueError("MA20 viewer requires the frozen MA-period signal version")
    record = next(
        (
            output
            for output in manifest.get("outputs", [])
            if output.get("path") == MA20_SIGNAL_DAILY_PATH
        ),
        None,
    )
    if record is None:
        raise ValueError("MA20 signal bundle is missing signal_daily.csv")

    path = signal_dir / MA20_SIGNAL_DAILY_PATH
    if sha256_file(path) != record.get("sha256"):
        raise ValueError("MA20 signal bundle hash mismatch: signal_daily.csv")
    encoding = str(record.get("encoding", "utf-8"))
    frame = pd.read_csv(path, encoding=encoding)
    if len(frame) != record.get("rows"):
        raise ValueError("MA20 signal bundle row count mismatch: signal_daily.csv")
    if list(frame.columns) != record.get("columns"):
        raise ValueError("MA20 signal bundle columns mismatch: signal_daily.csv")
    return frame


def render_four_industry_v1_viewer(
    vipdoc: Path | str,
    signal_path: Path | str = DEFAULT_SIGNAL_PATH,
    output_path: Path | str = DEFAULT_FOUR_INDUSTRY_OUTPUT,
    threshold: float = 0.10,
) -> Path:
    """Render the archived four-industry signal through its legacy adapter."""

    signal = load_four_industry_v1_signal(signal_path)
    panels = viewer.build_viewer_panels(vipdoc, threshold)
    for panel in panels:
        add_four_industry_signal(panel.figure, panel.daily, signal)
    return viewer.write_viewer_panels(
        panels,
        output_path,
        threshold,
        page=FOUR_INDUSTRY_PAGE,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vipdoc",
        type=Path,
        default=Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc",
    )
    parser.add_argument(
        "--variant",
        choices=("base", "four-industry-v1"),
        default="base",
    )
    parser.add_argument("--signal", type=Path, default=DEFAULT_SIGNAL_PATH)
    parser.add_argument(
        "--ma20-input-dir",
        type=Path,
        default=DEFAULT_MA20_INPUT_DIR,
        help="已验收的全 A P1 输入快照目录（仅 base 变体使用）",
    )
    parser.add_argument(
        "--ma20-signal-dir",
        type=Path,
        default=DEFAULT_MA20_SIGNAL_DIR,
        help="冻结的 MA 周期宽度信号 bundle（仅 base 变体使用）",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="基础阈值；每个指数再乘以固定波动倍率",
    )
    args = parser.parse_args()

    if args.variant == "four-industry-v1":
        output = args.output or DEFAULT_FOUR_INDUSTRY_OUTPUT
        result = render_four_industry_v1_viewer(
            args.vipdoc,
            args.signal,
            output,
            args.threshold,
        )
    else:
        output = args.output or DEFAULT_BASE_OUTPUT
        result = render_base_viewer(
            args.vipdoc,
            output,
            args.threshold,
            ma20_input_dir=args.ma20_input_dir,
            ma20_signal_dir=args.ma20_signal_dir,
        )
    print(result)


if __name__ == "__main__":
    main()
