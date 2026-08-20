"""Build the immutable stage-E single-industry MA20 Top1 signal bundle."""

from __future__ import annotations

import argparse
import hashlib
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
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"signal bundle already exists: {output_dir}")

    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    industry, source_record = _load_industry_frame(input_dir, input_manifest)
    daily, episodes, comparison = build_single_industry_top1_signals(
        industry,
        version=signal_version,
        start_date=start_date,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "signal_daily": output_dir / "signal_daily.csv",
        "signal_episodes": output_dir / "signal_episodes.csv",
        "manifest": output_dir / "manifest.json",
    }
    daily.to_csv(outputs["signal_daily"], index=False)
    episodes.to_csv(outputs["signal_episodes"], index=False)
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
            "manifest": _input_file_record(manifest_path),
            "data_version": input_manifest["data_version"],
            "source_files": [source_record],
        },
        "logic": _logic_records(),
        "outputs": [
            _output_frame_record(outputs["signal_daily"], daily, output_dir),
            _output_frame_record(outputs["signal_episodes"], episodes, output_dir),
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
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
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


def _load_industry_frame(
    input_dir: Path,
    manifest: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    source = next(
        (
            record
            for record in manifest.get("files", [])
            if str(record.get("path")) == INDUSTRY_PATH
        ),
        None,
    )
    if source is None:
        raise ValueError(f"input snapshot is missing file: {INDUSTRY_PATH}")
    path = input_dir / INDUSTRY_PATH
    digest = _sha256_file(path)
    if digest != source.get("sha256"):
        raise ValueError(f"input snapshot hash mismatch: {INDUSTRY_PATH}")
    encoding = str(source.get("encoding", "utf-8-sig"))
    frame = pd.read_csv(path, encoding=encoding)
    if len(frame) != source.get("rows"):
        raise ValueError(f"input snapshot row count mismatch: {INDUSTRY_PATH}")
    if list(frame.columns) != source.get("columns"):
        raise ValueError(f"input snapshot columns mismatch: {INDUSTRY_PATH}")
    return frame, {
        "path": INDUSTRY_PATH,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": encoding,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _output_frame_record(
    path: Path,
    frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": "utf-8",
    }


def _logic_records() -> dict[str, object]:
    paths = [
        PROJECT_DIR / "signals" / "events.py",
        PROJECT_DIR / "signals" / "definitions" / "single_industry_top1.py",
        Path(__file__),
    ]
    combined = hashlib.sha256()
    files = []
    for path in paths:
        content = path.read_bytes()
        relative = path.relative_to(PROJECT_DIR).as_posix()
        files.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(content)
    return {"combined_sha256": combined.hexdigest(), "files": files}


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
