"""Render the base viewer or an explicit historical-signal variant."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..adapters.legacy_four_industry_v1 import (
    DEFAULT_SIGNAL_PATH,
    load_four_industry_v1_signal,
)
from ..visualization import viewer
from ..visualization.overlays.four_industry import (
    FOUR_INDUSTRY_PAGE,
    add_four_industry_signal,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BASE_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "viewers"
    / "top_bottom_regions_v2"
    / "index_turning_points.html"
)
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
) -> Path:
    """Render the ground-truth viewer without a signal overlay."""

    return viewer.write_viewer(vipdoc, output_path, threshold)


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
        result = render_base_viewer(args.vipdoc, output, args.threshold)
    print(result)


if __name__ == "__main__":
    main()
