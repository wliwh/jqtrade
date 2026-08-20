"""Build the immutable stage-E four-industry Top1 signal bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..signals.definitions.four_industry_top1 import (
    MIN_INDUSTRY_VALID_COUNT,
    SIGNAL_VERSION,
    TARGET_NAMES,
    build_four_industry_top1_signal,
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


def run_pipeline(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    signal_version: str = SIGNAL_VERSION,
) -> dict[str, Path]:
    """Validate the V2 snapshot and write daily, episode and manifest files."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"signal bundle already exists: {output_dir}")
    manifest_path = input_dir / "manifest.json"
    input_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _validate_input_protocol(input_manifest)
    frames, source_records = _load_manifest_frames(input_dir, input_manifest)
    daily, episodes, comparison = build_four_industry_top1_signal(
        frames["data/daily_market_features.csv"],
        version=signal_version,
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
            "Causal four-industry MA20 Top1 historical baseline for stage-C/D "
            "evaluation; not a trading strategy."
        ),
        "definition": {
            "direction": "top",
            "target_names": TARGET_NAMES,
            "rank_window": 20,
            "rank_method": "min descending; ties included",
            "min_industry_valid_count": MIN_INDUSTRY_VALID_COUNT,
            "trigger": (
                "At least one target is Top1 among all rank-eligible point-in-time "
                "SW level-1 industries."
            ),
            "raw_value": "maximum MA20 breadth among the four targets",
            "start_policy": (
                "Each industry starts on its first comparable date; a cohort "
                "starts on the latest member start date."
            ),
            "substitution_policy": "No predecessor or substitute industry.",
            "capped_confirmation_n": 2,
        },
        "comparison": comparison,
        "inputs": {
            "manifest": _input_file_record(manifest_path),
            "data_version": input_manifest["data_version"],
            "source_files": source_records,
        },
        "logic": _logic_records(),
        "outputs": [
            _output_frame_record(outputs["signal_daily"], daily, output_dir),
            _output_frame_record(outputs["signal_episodes"], episodes, output_dir),
        ],
        "counts": {
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
        raise ValueError("four-industry signal requires all_a_p1_inputs_v2")
    query = manifest.get("query", {})
    expected_targets = {
        "bank": "银行",
        "nonferrous": "有色金属",
        "steel": "钢铁",
        "coal": "煤炭",
    }
    if query.get("targets") != expected_targets:
        raise ValueError("input target industries do not match frozen definition")
    if query.get("old_mining_mapped_to_coal") is not False:
        raise ValueError("input must not map old mining to coal")
    if (
        query.get("industry_rank_window") != 20
        or query.get("industry_rank_method") != "min_descending"
        or query.get("min_industry_valid_count") != MIN_INDUSTRY_VALID_COUNT
    ):
        raise ValueError("input industry ranking protocol does not match signal")


def _load_manifest_frames(
    input_dir: Path,
    manifest: dict[str, object],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    frames = {}
    records = []
    for source in manifest.get("files", []):
        relative = str(source["path"])
        path = input_dir / relative
        digest = _sha256_file(path)
        if digest != source.get("sha256"):
            raise ValueError(f"input snapshot hash mismatch: {relative}")
        frame = pd.read_csv(path, encoding=source.get("encoding", "utf-8-sig"))
        if len(frame) != source.get("rows"):
            raise ValueError(f"input snapshot row count mismatch: {relative}")
        if list(frame.columns) != source.get("columns"):
            raise ValueError(f"input snapshot columns mismatch: {relative}")
        frames[relative] = frame
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "rows": len(frame),
                "columns": list(frame.columns),
                "encoding": source.get("encoding", "utf-8-sig"),
            }
        )
    required = {
        "data/daily_market_features.csv",
        "data/industry_breadth.csv",
    }
    missing = required.difference(frames)
    if missing:
        raise ValueError(f"input snapshot is missing files: {sorted(missing)}")
    return frames, records


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
        PROJECT_DIR / "signals" / "definitions" / "four_industry_top1.py",
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
    args = parser.parse_args()
    outputs = run_pipeline(
        args.input_dir,
        args.output_dir,
        signal_version=args.signal_version,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
