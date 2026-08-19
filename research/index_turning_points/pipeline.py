"""Small batch pipeline for index turning-point research data."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import directional_change_labels
from .regions import DEFAULT_REGION_PROTOCOL, build_turning_point_regions


BASE_THRESHOLDS = (
    ("small", 0.05),
    ("medium", 0.10),
    ("large", 0.20),
)
HORIZONS = (5, 10, 20, 60)

SSE_2021_MEDIUM_TOP_ANCHORS = (
    "2021-02-18",
    "2021-06-02",
    "2021-09-14",
)

INDEX_SPECS = (
    ("sse_composite", "上证指数", "000001.XSHG", "sh/lday/sh999999.day", False),
    ("csi300", "沪深300", "000300.XSHG", "sh/lday/sh000300.day", False),
    ("csi500", "中证500", "000905.XSHG", "sh/lday/sh000905.day", False),
    ("csi1000", "中证1000", "000852.XSHG", "sh/lday/sh000852.day", False),
    ("cni2000", "国证2000", "399303.XSHE", "sz/lday/sz399303.day", False),
    ("microcap", "微盘股", "TDX.880823", "sh/lday/sh880823.day", False),
    ("all_a", "全A", "000985.XSHG", "ds/lday/62#000985.day", True),
)

# Fixed before signal evaluation. These are intentionally simple volatility
# buckets rather than rolling or outcome-optimized parameters.
INDEX_THRESHOLD_MULTIPLIERS = {
    "sse_composite": 0.8,
    "csi300": 0.9,
    "csi500": 1.1,
    "csi1000": 1.2,
    "cni2000": 1.2,
    "microcap": 1.3,
    "all_a": 1.0,
}

STANDARD_RECORD = struct.Struct("<IIIIIfII")
FLOAT_RECORD = struct.Struct("<IfffffII")
DAILY_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "amount",
    "volume",
    "reserved",
]


def threshold_for_index(index_id: str, base_threshold: float) -> float:
    """Return one fixed index-specific threshold from a common base scale."""

    try:
        multiplier = INDEX_THRESHOLD_MULTIPLIERS[index_id]
    except KeyError as exc:
        raise KeyError(f"missing threshold multiplier for index: {index_id}") from exc

    threshold = round(float(base_threshold) * multiplier, 10)
    if not 0.0 < threshold < 1.0:
        raise ValueError("adjusted threshold must be between 0 and 1")
    return threshold


def read_tdx_daily(path: Path | str, *, float_prices: bool = False) -> pd.DataFrame:
    """Read and minimally validate one 32-byte TDX daily file."""

    path = Path(path)
    data = path.read_bytes()
    record = FLOAT_RECORD if float_prices else STANDARD_RECORD
    if not data or len(data) % record.size:
        raise ValueError(f"invalid TDX daily file size: {path}")

    frame = pd.DataFrame(record.iter_unpack(data), columns=DAILY_COLUMNS)
    if not float_prices:
        price_columns = ["open", "high", "low", "close"]
        frame[price_columns] = frame[price_columns].astype(float).div(100.0)

    try:
        dates = pd.to_datetime(frame.pop("date").astype(str), format="%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid date in TDX daily file: {path}") from exc
    frame.index = dates
    frame.index.name = "date"

    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"dates are not increasing: {path}")

    duplicated = frame.index.duplicated(keep=False)
    if duplicated.any():
        for date, group in frame.loc[duplicated].groupby(level=0, sort=False):
            if len(group.drop_duplicates()) != 1:
                raise ValueError(f"conflicting rows for {date.date()}: {path}")
        frame = frame.loc[~frame.index.duplicated(keep="last")]

    prices = frame[["open", "high", "low", "close"]]
    valid_ohlc = (
        np.isfinite(prices.to_numpy()).all()
        and (prices > 0).all().all()
        and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
        and (frame["high"] >= frame["low"]).all()
    )
    if not valid_ohlc:
        raise ValueError(f"invalid OHLC data: {path}")

    return frame


def forward_outcomes(
    close: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Calculate full-window future downside, upside and terminal return."""

    result = pd.DataFrame({"close": close.astype(float)})
    for horizon in horizons:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizons must contain positive integers")

        future = pd.concat(
            [close.shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        complete = future.notna().all(axis=1)
        returns = future.divide(close, axis=0).sub(1.0)
        result[f"future_max_down_{horizon}d"] = returns.min(axis=1).where(complete)
        result[f"future_max_up_{horizon}d"] = returns.max(axis=1).where(complete)
        result[f"future_return_{horizon}d"] = (
            close.shift(-horizon).divide(close).sub(1.0).where(complete)
        )

    return result


def run_pipeline(vipdoc: Path | str, output_dir: Path | str) -> dict[str, Path]:
    """Read indices and write point labels, regions and forward outcomes."""

    vipdoc = Path(vipdoc)
    output_dir = Path(output_dir)
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
    paths = [
        Path(__file__).with_name("labels.py"),
        Path(__file__).with_name("regions.py"),
        Path(__file__),
    ]
    combined = hashlib.sha256()
    files = []
    for path in paths:
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        files.append({"path": path.name, "sha256": digest})
        combined.update(path.name.encode("utf-8"))
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
        default=Path(__file__).resolve().parent / "artifacts",
        help="CSV 输出目录",
    )
    args = parser.parse_args()

    outputs = run_pipeline(args.vipdoc, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
