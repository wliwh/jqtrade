"""Build one versioned post-hoc ground-truth artifact bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..adapters.tdx import (
    INDEX_SPECS,
    INDEX_THRESHOLD_MULTIPLIERS,
    STANDARD_RECORD,
    read_tdx_daily,
    threshold_for_index,
)
from ..ground_truth.labels import directional_change_labels
from ..ground_truth.outcomes import forward_outcomes
from ..ground_truth.regions import (
    DEFAULT_REGION_PROTOCOL,
    build_turning_point_regions,
)


BASE_THRESHOLDS = (
    ("small", 0.05),
    ("medium", 0.10),
    ("large", 0.20),
)
SSE_2021_MEDIUM_TOP_ANCHORS = (
    "2021-02-18",
    "2021-06-02",
    "2021-09-14",
)

def run_pipeline(vipdoc: Path | str, output_dir: Path | str) -> dict[str, Path]:
    """Read indices and write point labels, regions and forward outcomes."""

    vipdoc = Path(vipdoc)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"ground-truth bundle already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    all_labels = []
    all_outcomes = []
    all_regions = []
    all_region_lobes = []
    source_records = []

    for index_id, index_name, jq_code, relative_path, float_prices in INDEX_SPECS:
        path = vipdoc / relative_path
        raw_rows = path.stat().st_size // STANDARD_RECORD.size
        daily = read_tdx_daily(path, float_prices=float_prices)
        close = daily["close"]
        adjusted_thresholds = {
            level: threshold_for_index(index_id, base_threshold)
            for level, base_threshold in BASE_THRESHOLDS
        }

        source_records.append(
            {
                "index_id": index_id,
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "start_date": daily.index.min().strftime("%Y-%m-%d"),
                "end_date": daily.index.max().strftime("%Y-%m-%d"),
            }
        )

        manifests.append(
            {
                "index_id": index_id,
                "index_name": index_name,
                "jq_code": jq_code,
                "source_file": relative_path,
                "raw_rows": raw_rows,
                "rows": len(daily),
                "duplicates_removed": raw_rows - len(daily),
                "start_date": daily.index.min(),
                "end_date": daily.index.max(),
                "last_close": close.iloc[-1],
                "threshold_multiplier": INDEX_THRESHOLD_MULTIPLIERS[index_id],
                "threshold_small": adjusted_thresholds["small"],
                "threshold_medium": adjusted_thresholds["medium"],
                "threshold_large": adjusted_thresholds["large"],
            }
        )

        labels_by_level = {}
        for threshold_level, threshold in adjusted_thresholds.items():
            labels = directional_change_labels(daily["high"], daily["low"], threshold)
            labels_by_level[threshold_level] = labels
            labels.insert(0, "index_name", index_name)
            labels.insert(0, "index_id", index_id)
            labels.insert(2, "threshold_level", threshold_level)
            all_labels.append(labels)

        regions, region_lobes = build_turning_point_regions(
            daily,
            labels_by_level["medium"],
            index_id=index_id,
            index_name=index_name,
            small_labels=labels_by_level["small"],
        )
        all_regions.append(regions)
        all_region_lobes.append(region_lobes)
        manifests[-1].update(
            {
                "region_label_version": DEFAULT_REGION_PROTOCOL.label_version,
                "region_price_band_pct": DEFAULT_REGION_PROTOCOL.resolve_price_band_pct(
                    adjusted_thresholds["medium"]
                ),
                "region_count": len(regions),
                "top_region_count": int(regions["event_type"].eq("top").sum()),
                "bottom_region_count": int(
                    regions["event_type"].eq("bottom").sum()
                ),
                "multi_lobe_region_count": int(regions["lobe_count"].gt(1).sum()),
            }
        )

        outcomes = forward_outcomes(close).reset_index()
        outcomes.insert(0, "index_name", index_name)
        outcomes.insert(0, "index_id", index_id)
        all_outcomes.append(outcomes)

    region_output_dir = (
        output_dir / "regions" / DEFAULT_REGION_PROTOCOL.label_version
    )
    region_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "manifest": output_dir / "data_manifest.csv",
        "labels": output_dir / "turning_point_labels.csv",
        "outcomes": output_dir / "forward_outcomes.csv",
        "regions": region_output_dir / "turning_point_regions.csv",
        "region_lobes": region_output_dir / "turning_point_region_lobes.csv",
        "region_manifest": region_output_dir / "manifest.json",
    }
    manifest_frame = pd.DataFrame(manifests)
    labels_frame = pd.concat(all_labels, ignore_index=True)
    outcomes_frame = pd.concat(all_outcomes, ignore_index=True)
    regions_frame = pd.concat(all_regions, ignore_index=True)
    lobes_frame = pd.concat(all_region_lobes, ignore_index=True)
    manifest_frame.to_csv(outputs["manifest"], index=False)
    labels_frame.to_csv(outputs["labels"], index=False)
    outcomes_frame.to_csv(outputs["outcomes"], index=False)
    regions_frame.to_csv(outputs["regions"], index=False)
    lobes_frame.to_csv(outputs["region_lobes"], index=False)

    acceptance_checks = _reference_acceptance_checks(regions_frame, manifest_frame)
    region_manifest = {
        "label_version": DEFAULT_REGION_PROTOCOL.label_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Post-hoc top/bottom ground truth; not a point-in-time signal.",
        "protocol": DEFAULT_REGION_PROTOCOL.to_dict(),
        "boundary_policy": (
            "Each medium anchor owns the non-overlapping midpoint cell between "
            "its adjacent directional-change anchors, capped by max_side_days."
        ),
        "source_files": source_records,
        "logic": _logic_records(),
        "point_labels": _artifact_record(outputs["labels"], labels_frame, output_dir),
        "outputs": [
            _artifact_record(outputs["regions"], regions_frame, output_dir),
            _artifact_record(outputs["region_lobes"], lobes_frame, output_dir),
        ],
        "counts": {
            "regions": len(regions_frame),
            "top_regions": int(regions_frame["event_type"].eq("top").sum()),
            "bottom_regions": int(regions_frame["event_type"].eq("bottom").sum()),
            "lobes": len(lobes_frame),
            "multi_lobe_regions": int(regions_frame["lobe_count"].gt(1).sum()),
        },
        "acceptance_checks": acceptance_checks,
    }
    outputs["region_manifest"].write_text(
        json.dumps(region_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(
    path: Path,
    frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "rows": len(frame),
        "columns": list(frame.columns),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "encoding": "utf-8",
    }


def _logic_records() -> dict[str, object]:
    project_dir = Path(__file__).resolve().parents[1]
    paths = [
        project_dir / "ground_truth" / "labels.py",
        project_dir / "ground_truth" / "regions.py",
        project_dir / "ground_truth" / "outcomes.py",
        project_dir / "adapters" / "tdx.py",
        Path(__file__),
    ]
    combined = hashlib.sha256()
    files = []
    for path in paths:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        relative_path = path.relative_to(project_dir).as_posix()
        files.append({"path": relative_path, "sha256": digest})
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(content)
    return {"combined_sha256": combined.hexdigest(), "files": files}


def _reference_acceptance_checks(
    regions: pd.DataFrame,
    manifests: pd.DataFrame,
) -> dict[str, object]:
    sse_manifest = manifests[manifests["index_id"].eq("sse_composite")]
    if sse_manifest.empty:
        return {"sse_2021_medium_tops": {"status": "not_applicable"}}
    source_start = pd.Timestamp(sse_manifest.iloc[0]["start_date"])
    source_end = pd.Timestamp(sse_manifest.iloc[0]["end_date"])
    if source_start > pd.Timestamp("2021-01-01") or source_end < pd.Timestamp(
        "2021-12-31"
    ):
        return {"sse_2021_medium_tops": {"status": "not_applicable"}}

    selected = regions[
        regions["index_id"].eq("sse_composite")
        & regions["event_type"].eq("top")
        & regions["anchor_date"].between("2021-01-01", "2021-12-31")
    ]
    actual = tuple(
        pd.Timestamp(value).strftime("%Y-%m-%d") for value in selected["anchor_date"]
    )
    if actual != SSE_2021_MEDIUM_TOP_ANCHORS:
        raise RuntimeError(
            "SSE 2021 medium top acceptance failed: "
            f"expected {SSE_2021_MEDIUM_TOP_ANCHORS}, got {actual}"
        )
    return {
        "sse_2021_medium_tops": {
            "status": "passed",
            "expected_anchor_dates": list(SSE_2021_MEDIUM_TOP_ANCHORS),
            "actual_anchor_dates": list(actual),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vipdoc",
        type=Path,
        default=Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc",
        help="通达信 vipdoc 目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="新的版本化 ground-truth bundle 目录；不得指向既有 bundle",
    )
    args = parser.parse_args()

    outputs = run_pipeline(args.vipdoc, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
