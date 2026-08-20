"""Build one immutable stage-D evaluation bundle for causal signal events."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..adapters.tdx import INDEX_SPECS, read_tdx_daily
from ..evaluation.post_event import (
    DEFAULT_HORIZONS,
    EVENT_FLAGS,
    build_forward_event_outcomes,
    summarize_forward_event_outcomes,
)
from ..evaluation.region_matching import match_signal_regions
from ..evaluation.region_metrics import (
    add_diagnostic_region_slices,
    summarize_region_slices,
)
from ..evaluation.reports import render_forward_report, render_region_report
from ..ground_truth.regions import DEFAULT_REGION_PROTOCOL
from ..signals.events import (
    DAILY_EVENT_COLUMNS,
    EPISODE_COLUMNS,
    REQUIRED_DAILY_COLUMNS,
)


DEFAULT_MIN_EVENT_COUNT = 20
DEFAULT_MIN_BASELINE_COUNT = 30
EVENT_KINDS = tuple(EVENT_FLAGS)


def run_pipeline(
    signal_daily_path: Path | str,
    signal_episodes_path: Path | str,
    ground_truth_dir: Path | str,
    vipdoc: Path | str,
    output_dir: Path | str,
    *,
    evaluation_version: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_event_count: int = DEFAULT_MIN_EVENT_COUNT,
    min_baseline_count: int = DEFAULT_MIN_BASELINE_COUNT,
) -> dict[str, Path]:
    """Evaluate one signal artifact and write two reports plus their manifest."""

    _validate_version(evaluation_version)
    signal_daily_path = Path(signal_daily_path)
    signal_episodes_path = Path(signal_episodes_path)
    ground_truth_dir = Path(ground_truth_dir)
    vipdoc = Path(vipdoc)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation bundle already exists: {output_dir}")

    signal_daily = pd.read_csv(signal_daily_path)
    signal_episodes = pd.read_csv(signal_episodes_path)
    _validate_signal_bundle(signal_daily, signal_episodes)

    region_dir = (
        ground_truth_dir
        / "regions"
        / DEFAULT_REGION_PROTOCOL.label_version
    )
    regions_path = region_dir / "turning_point_regions.csv"
    lobes_path = region_dir / "turning_point_region_lobes.csv"
    ground_truth_manifest_path = region_dir / "manifest.json"
    ground_truth_manifest = _load_and_verify_ground_truth(
        ground_truth_manifest_path,
        regions_path,
        lobes_path,
    )
    regions = pd.read_csv(regions_path)
    lobes = pd.read_csv(lobes_path)
    ohlc, source_records = _load_ohlc(
        vipdoc,
        ground_truth_manifest["source_files"],
    )
    calendars = ohlc[["index_id", "index_name", "date"]].copy()
    expected_indices = set(regions["index_id"].astype(str))
    actual_indices = set(ohlc["index_id"].astype(str))
    if expected_indices != actual_indices:
        raise ValueError(
            "ground-truth and OHLC indices differ: "
            f"ground_truth={sorted(expected_indices)}, ohlc={sorted(actual_indices)}"
        )

    match_frames = [
        match_signal_regions(
            signal_daily,
            regions,
            lobes,
            calendars,
            event_kind=event_kind,
        )
        for event_kind in EVENT_KINDS
    ]
    region_matches = pd.concat(match_frames, ignore_index=True)
    region_matches = add_diagnostic_region_slices(region_matches, regions)
    region_metrics = summarize_region_slices(region_matches)
    forward_outcomes = build_forward_event_outcomes(
        signal_daily,
        ohlc,
        event_kinds=EVENT_KINDS,
        horizons=horizons,
    )
    forward_metrics = summarize_forward_event_outcomes(
        forward_outcomes,
        signal_daily,
        ohlc,
        min_event_count=min_event_count,
        min_baseline_count=min_baseline_count,
    )

    region_report = render_region_report(
        region_metrics,
        region_matches,
        evaluation_version=evaluation_version,
        label_version=DEFAULT_REGION_PROTOCOL.label_version,
    )
    forward_report = render_forward_report(
        forward_metrics,
        forward_outcomes,
        evaluation_version=evaluation_version,
        min_event_count=min_event_count,
        min_baseline_count=min_baseline_count,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "region_matches": output_dir / "region_matches.csv",
        "region_metrics": output_dir / "region_metrics.csv",
        "region_report": output_dir / "region_report.md",
        "forward_event_outcomes": output_dir / "forward_event_outcomes.csv",
        "forward_metrics": output_dir / "forward_metrics.csv",
        "forward_report": output_dir / "forward_report.md",
        "manifest": output_dir / "manifest.json",
    }
    frames = {
        "region_matches": region_matches,
        "region_metrics": region_metrics,
        "forward_event_outcomes": forward_outcomes,
        "forward_metrics": forward_metrics,
    }
    for name, frame in frames.items():
        frame.to_csv(outputs[name], index=False)
    outputs["region_report"].write_text(region_report, encoding="utf-8")
    outputs["forward_report"].write_text(forward_report, encoding="utf-8")

    manifest = {
        "evaluation_version": evaluation_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Offline evaluation of causal signal events against post-hoc regions "
            "and future OHLC paths; not a trading backtest."
        ),
        "composite_score": None,
        "protocol": {
            "region": DEFAULT_REGION_PROTOCOL.to_dict(),
            "event_kinds": list(EVENT_KINDS),
            "forward_horizons": list(horizons),
            "outcomes": {
                "terminal_return": "close[t+h] / close[t] - 1",
                "max_up": "max(high[t+1:t+h]) / close[t] - 1",
                "max_down": "min(low[t+1:t+h]) / close[t] - 1",
            },
            "baseline": "complete non-event dates in the same signal/index coverage",
            "inference": {
                "model": "OLS outcome ~ intercept + event indicator",
                "covariance": "Newey-West",
                "hac_lag": "equal to horizon",
                "min_event_count": min_event_count,
                "min_baseline_count": min_baseline_count,
                "local_fdr_family": (
                    "signal_id/direction/version/event_kind across indices, "
                    "horizons and outcomes"
                ),
                "global_fdr_family": "all eligible tests in this bundle",
            },
            "aggregate_region_unit": "index-region or index-episode pair",
            "false_alarm_slice_assignment": (
                "nearest same-direction region in coverage, diagnostic only"
            ),
        },
        "inputs": {
            "signal_daily": _input_frame_record(signal_daily_path, signal_daily),
            "signal_episodes": _input_frame_record(
                signal_episodes_path, signal_episodes
            ),
            "ground_truth_manifest": _input_file_record(
                ground_truth_manifest_path
            ),
            "turning_point_regions": _input_frame_record(regions_path, regions),
            "turning_point_region_lobes": _input_frame_record(lobes_path, lobes),
            "tdx_ohlc": source_records,
        },
        "logic": _logic_records(),
        "outputs": [
            _output_frame_record(outputs[name], frames[name], output_dir)
            for name in frames
        ]
        + [
            _output_file_record(outputs["region_report"], output_dir),
            _output_file_record(outputs["forward_report"], output_dir),
        ],
        "counts": {
            "signal_daily_rows": len(signal_daily),
            "signal_episodes": len(signal_episodes),
            "region_match_rows": len(region_matches),
            "region_metric_rows": len(region_metrics),
            "forward_event_outcome_rows": len(forward_outcomes),
            "forward_metric_rows": len(forward_metrics),
            "eligible_forward_tests": int(
                forward_metrics["inference_eligible"].sum()
            ),
        },
    }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def _validate_signal_bundle(
    signal_daily: pd.DataFrame,
    signal_episodes: pd.DataFrame,
) -> None:
    required_daily = set(REQUIRED_DAILY_COLUMNS) | set(DAILY_EVENT_COLUMNS)
    missing_daily = required_daily.difference(signal_daily.columns)
    if missing_daily:
        raise ValueError(f"signal_daily is missing columns: {sorted(missing_daily)}")
    missing_episodes = set(EPISODE_COLUMNS).difference(signal_episodes.columns)
    if missing_episodes:
        raise ValueError(
            f"signal_episodes is missing columns: {sorted(missing_episodes)}"
        )
    if signal_episodes["episode_id"].duplicated().any():
        raise ValueError("signal_episodes contains duplicate episode_id values")
    daily = signal_daily.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    episodes = signal_episodes.copy()
    episodes["onset_date"] = pd.to_datetime(
        episodes["onset_date"], errors="coerce"
    )
    episodes["capped_confirmation_date"] = pd.to_datetime(
        episodes["capped_confirmation_date"], errors="coerce"
    )
    if daily["date"].isna().any() or episodes["onset_date"].isna().any():
        raise ValueError("signal bundle contains invalid dates")
    onsets = daily[_bool_series(daily["event_onset"])][
        ["episode_id", "signal_id", "direction", "version", "date"]
    ].rename(columns={"date": "onset_date"})
    episode_identity = episodes[
        ["episode_id", "signal_id", "direction", "version", "onset_date"]
    ]
    normalized_onsets = _normalized_records(onsets, "onset_date")
    normalized_episodes = _normalized_records(episode_identity, "onset_date")
    if normalized_onsets != normalized_episodes:
        raise ValueError("signal_daily onset events do not match signal_episodes")
    confirmations = daily[_bool_series(daily["event_capped_confirmation"])][
        ["episode_id", "signal_id", "direction", "version", "date"]
    ].rename(columns={"date": "capped_confirmation_date"})
    expected_confirmations = episodes.loc[
        episodes["capped_confirmation_date"].notna(),
        [
            "episode_id",
            "signal_id",
            "direction",
            "version",
            "capped_confirmation_date",
        ],
    ]
    if _normalized_records(
        confirmations, "capped_confirmation_date"
    ) != _normalized_records(expected_confirmations, "capped_confirmation_date"):
        raise ValueError(
            "signal_daily capped confirmations do not match signal_episodes"
        )
    expected_n = DEFAULT_REGION_PROTOCOL.capped_confirmation_n
    daily_n = pd.to_numeric(daily["capped_confirmation_n"], errors="coerce")
    episode_n = pd.to_numeric(
        episodes["capped_confirmation_n"], errors="coerce"
    )
    if not daily_n.eq(expected_n).all() or not episode_n.eq(expected_n).all():
        raise ValueError(
            f"signal capped_confirmation_n must equal frozen value {expected_n}"
        )


def _normalized_records(
    frame: pd.DataFrame,
    date_column: str,
) -> set[tuple[str, ...]]:
    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column]).dt.strftime(
        "%Y-%m-%d"
    )
    return {
        tuple(str(value) for value in row)
        for row in result.itertuples(index=False, name=None)
    }


def _bool_series(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("signal event flags must not contain missing values")
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError("signal event flags must contain only booleans")
    return normalized.map(mapping).astype(bool)


def _load_and_verify_ground_truth(
    manifest_path: Path,
    regions_path: Path,
    lobes_path: Path,
) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("label_version") != DEFAULT_REGION_PROTOCOL.label_version:
        raise ValueError("ground-truth label_version does not match evaluation protocol")
    if manifest.get("protocol") != DEFAULT_REGION_PROTOCOL.to_dict():
        raise ValueError("ground-truth region protocol does not match evaluation protocol")
    output_records = {
        Path(str(record["path"])).name: record
        for record in manifest.get("outputs", [])
    }
    for path in (regions_path, lobes_path):
        record = output_records.get(path.name)
        if record is None or record.get("sha256") != _sha256_file(path):
            raise ValueError(f"ground-truth artifact hash mismatch: {path}")
    if not manifest.get("source_files"):
        raise ValueError("ground-truth manifest has no source_files")
    return manifest


def _load_ohlc(
    vipdoc: Path,
    expected_sources: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    expected_by_index = {
        str(record["index_id"]): record for record in expected_sources
    }
    frames = []
    records = []
    for index_id, index_name, _jq_code, relative_path, float_prices in INDEX_SPECS:
        expected = expected_by_index.get(index_id)
        if expected is None or str(expected.get("path")) != relative_path:
            raise ValueError(f"ground-truth manifest source mismatch: {index_id}")
        path = vipdoc / relative_path
        digest = _sha256_file(path)
        if digest != expected.get("sha256"):
            raise ValueError(f"TDX source hash mismatch: {relative_path}")
        daily = read_tdx_daily(path, float_prices=float_prices).reset_index()
        frame = daily[["date", "high", "low", "close"]].copy()
        frame.insert(0, "index_name", index_name)
        frame.insert(0, "index_id", index_id)
        frames.append(frame)
        records.append(
            {
                "index_id": index_id,
                "index_name": index_name,
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": digest,
                "rows": len(frame),
                "start_date": frame["date"].min().strftime("%Y-%m-%d"),
                "end_date": frame["date"].max().strftime("%Y-%m-%d"),
            }
        )
    if set(expected_by_index) != {record["index_id"] for record in records}:
        raise ValueError("ground-truth manifest contains unexpected TDX sources")
    return pd.concat(frames, ignore_index=True), records


def _validate_version(value: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "/" in value
        or "\\" in value
    ):
        raise ValueError("evaluation_version must be a non-empty path-safe string")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_frame_record(path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        **_input_file_record(path),
        "rows": len(frame),
        "columns": list(frame.columns),
    }


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
        **_output_file_record(path, output_dir),
        "rows": len(frame),
        "columns": list(frame.columns),
    }


def _output_file_record(path: Path, output_dir: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(output_dir).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "encoding": "utf-8",
    }


def _logic_records() -> dict[str, object]:
    project_dir = Path(__file__).resolve().parents[1]
    paths = [
        project_dir / "evaluation" / "region_matching.py",
        project_dir / "evaluation" / "region_metrics.py",
        project_dir / "evaluation" / "post_event.py",
        project_dir / "evaluation" / "reports.py",
        project_dir / "ground_truth" / "regions.py",
        project_dir / "adapters" / "tdx.py",
        Path(__file__),
    ]
    combined = hashlib.sha256()
    files = []
    for path in paths:
        content = path.read_bytes()
        relative = path.relative_to(project_dir).as_posix()
        files.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(content)
    return {"combined_sha256": combined.hexdigest(), "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal-daily", type=Path, required=True)
    parser.add_argument("--signal-episodes", type=Path, required=True)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument(
        "--vipdoc",
        type=Path,
        default=Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-version", required=True)
    args = parser.parse_args()
    outputs = run_pipeline(
        args.signal_daily,
        args.signal_episodes,
        args.ground_truth_dir,
        args.vipdoc,
        args.output_dir,
        evaluation_version=args.evaluation_version,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
