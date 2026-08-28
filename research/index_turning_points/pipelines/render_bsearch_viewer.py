"""Render eligible source search heat and local index prices as one offline HTML."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..visualization.bsearch_index import write_offline_viewer


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BSEARCH_PATH = PROJECT_DIR / "data" / "inputs" / "bsearch_index" / "bsearch_index.csv"
DEFAULT_REGION_PATH = (
    PROJECT_DIR
    / "artifacts"
    / "ground_truth"
    / "index_ohlc_20260814"
    / "regions"
    / "top_bottom_regions_v2"
    / "turning_point_regions.csv"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR
    / "artifacts"
    / "viewers"
    / "bsearch_index_exploration_v1_6_20110104_20260814"
    / "bsearch_index_exploration.html"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bsearch", type=Path, default=DEFAULT_BSEARCH_PATH)
    parser.add_argument(
        "--vipdoc",
        type=Path,
        default=Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc",
    )
    parser.add_argument("--regions", type=Path, default=DEFAULT_REGION_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    metadata = write_offline_viewer(
        bsearch_path=args.bsearch,
        vipdoc=args.vipdoc,
        region_path=args.regions,
        output_path=args.output,
    )
    manifest = {
        **metadata,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "selection_policy": "retain source keywords only when a meaningful local comparison index exists; exact-name matches use their local index and broad A-share terms use SSE; no outcome-based selection",
        "point_in_time_policy": "Z252 uses log1p(count) standardized against the prior 252 aligned trading days; minimum 60 prior days",
        "page_policy": "The index panel uses local OHLC candlesticks; changing a keyword selects its default local comparison index while the index remains manually overridable; the heat panel shows raw heat and point-in-time Z252 together and supports keyword-scoped manual peak annotations stored in browser localStorage with validated full-bundle JSON export/import; annotation mode keeps click-to-mark, drag-to-pan, wheel-to-zoom, and modebar zoom available together; no correlation or forward-return statistics are rendered on the page",
    }
    manifest_path = args.output.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(manifest_path)


if __name__ == "__main__":
    main()
