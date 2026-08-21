"""Shared file and manifest operations for signal bundle pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]


def require_empty_output_dir(output_dir: Path | str) -> Path:
    """Return a normalized output path, refusing to overwrite a bundle."""

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"signal bundle already exists: {output_dir}")
    return output_dir


def load_verified_frame(
    input_dir: Path,
    manifest: dict[str, object],
    relative_path: str,
    *,
    source_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read one manifest-listed CSV after validating its hash and shape."""

    source = next(
        (
            record
            for record in manifest.get("files", [])
            if str(record.get("path")) == relative_path
        ),
        None,
    )
    if source is None:
        raise ValueError(f"input snapshot is missing file: {relative_path}")

    path = input_dir / relative_path
    digest = sha256_file(path)
    if digest != source.get("sha256"):
        raise ValueError(f"input snapshot hash mismatch: {relative_path}")
    encoding = str(source.get("encoding", "utf-8-sig"))
    frame = pd.read_csv(path, encoding=encoding)
    if len(frame) != source.get("rows"):
        raise ValueError(f"input snapshot row count mismatch: {relative_path}")
    if list(frame.columns) != source.get("columns"):
        raise ValueError(f"input snapshot columns mismatch: {relative_path}")

    record: dict[str, object] = {
        "path": relative_path,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": encoding,
    }
    if source_name is not None:
        record = {"source": source_name, **record}
    return frame, record


def write_signal_frames(
    output_dir: Path,
    daily: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Path]:
    """Write the two standard signal tables and return all bundle paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "signal_daily": output_dir / "signal_daily.csv",
        "signal_episodes": output_dir / "signal_episodes.csv",
        "manifest": output_dir / "manifest.json",
    }
    daily.to_csv(outputs["signal_daily"], index=False)
    episodes.to_csv(outputs["signal_episodes"], index=False)
    return outputs


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_file_record(path: Path) -> dict[str, object]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def output_frame_record(
    path: Path,
    frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "encoding": "utf-8",
    }


def logic_records(paths: list[Path]) -> dict[str, object]:
    """Hash signal logic, including this shared bundle implementation."""

    helper_path = Path(__file__)
    if helper_path not in paths:
        paths = [*paths, helper_path]

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
