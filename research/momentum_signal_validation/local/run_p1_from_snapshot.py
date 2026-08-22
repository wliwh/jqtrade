"""Run the frozen P1 calculations locally from a validated JQ input snapshot."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from research.momentum_signal_validation import p1_jq_signal_validation as p1
from research.momentum_signal_validation.local.validate_p1_index_archive import (
    validate_p1_index_directory,
)


PRICE_FILES = {
    "broad": "broad_index_close.csv",
    "industry_sw_l1": "sw_l1_index_close.csv",
    "style": "style_index_close.csv",
}


def _manifest_record_by_basename(manifest):
    records = {}
    for record in manifest["files"]:
        basename = Path(record["path"]).name
        if basename in records:
            raise ValueError("duplicate payload basename in manifest: %s" % basename)
        records[basename] = record
    return records


def _read_snapshot_csv(snapshot_dir, records, basename):
    if basename not in records:
        raise ValueError("manifest is missing required payload: %s" % basename)
    path = snapshot_dir / "data" / basename
    return pd.read_csv(path, encoding=records[basename].get("encoding", "utf-8"))


def load_p1_close_panels(snapshot_dir):
    """Validate an extracted snapshot and return ordered wide close panels."""

    validation = validate_p1_index_directory(snapshot_dir)
    snapshot_dir = Path(validation["path"])
    manifest = validation["manifest"]
    records = _manifest_record_by_basename(manifest)
    catalog = _read_snapshot_csv(
        snapshot_dir, records, "universe_catalog.csv"
    ).copy()
    trade_days = _read_snapshot_csv(snapshot_dir, records, "trade_days.csv").copy()
    required_catalog_columns = {"universe_group", "code", "name", "version"}
    if not required_catalog_columns.issubset(catalog.columns):
        raise ValueError("universe catalog is missing required columns")
    if list(trade_days.columns) != ["date"]:
        raise ValueError("trade_days.csv must contain only the date column")
    catalog["code"] = catalog["code"].astype(str)
    trade_index = pd.DatetimeIndex(pd.to_datetime(trade_days["date"])).normalize()
    if trade_index.has_duplicates or not trade_index.is_monotonic_increasing:
        raise ValueError("trade calendar must be unique and increasing")

    expected_groups = tuple(p1.DEFAULT_CONFIG["universe_order"])
    if set(catalog["universe_group"]) != set(expected_groups):
        raise ValueError("snapshot catalog does not contain the frozen P1 groups")
    panels = {}
    members_by_group = {}
    for group_name in expected_groups:
        group_catalog = catalog[catalog["universe_group"].eq(group_name)]
        if group_catalog["code"].duplicated().any():
            raise ValueError("duplicate catalog code in %s" % group_name)
        codes = list(group_catalog["code"])
        members_by_group[group_name] = dict(
            zip(group_catalog["code"], group_catalog["name"].astype(str))
        )
        raw = _read_snapshot_csv(snapshot_dir, records, PRICE_FILES[group_name])
        if not {"date", "code", "close"}.issubset(raw.columns):
            raise ValueError("%s close file is missing required columns" % group_name)
        raw["date"] = pd.to_datetime(raw["date"]).dt.normalize()
        raw["code"] = raw["code"].astype(str)
        raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
        if raw.duplicated(["date", "code"]).any():
            raise ValueError("duplicate date/code rows in %s" % group_name)
        unexpected_codes = set(raw["code"]) - set(codes)
        if unexpected_codes:
            raise ValueError(
                "unexpected codes in %s: %s"
                % (group_name, sorted(unexpected_codes))
            )
        outside_dates = set(raw["date"]) - set(trade_index)
        if outside_dates:
            raise ValueError("close dates outside trade calendar in %s" % group_name)
        panel = raw.pivot(index="date", columns="code", values="close")
        panels[group_name] = panel.reindex(index=trade_index, columns=codes)
    return {
        "validation": validation,
        "manifest": manifest,
        "catalog": catalog,
        "trade_days": trade_index,
        "panels": panels,
        "members_by_group": members_by_group,
    }


def build_snapshot_research_config(loaded, overrides=None):
    """Bind local P1 settings to the members recorded in the snapshot."""

    manifest = loaded["manifest"]
    request = manifest["request"]
    groups = p1._copy_universes(p1.DEFAULT_CONFIG["universes"])
    for group_name in p1.DEFAULT_CONFIG["universe_order"]:
        groups[group_name]["members"] = dict(loaded["members_by_group"][group_name])
        group_catalog = loaded["catalog"][
            loaded["catalog"]["universe_group"].eq(group_name)
        ]
        versions = group_catalog["version"].dropna().astype(str).unique()
        if len(versions) != 1:
            raise ValueError("catalog must contain one version for %s" % group_name)
        groups[group_name]["version"] = versions[0]
    config_overrides = {
        "research_start": request["research_start"],
        "research_end": request["research_end"],
        "universes": groups,
    }
    if overrides:
        for key, value in overrides.items():
            if key == "universes":
                raise ValueError("snapshot universe members cannot be overridden")
            config_overrides[key] = value
    config = p1.build_config(config_overrides)

    prehistory = loaded["trade_days"][
        loaded["trade_days"] < pd.Timestamp(config["research_start"])
    ]
    if len(prehistory) < max(config["lookbacks"]) - 1:
        raise ValueError("snapshot does not contain enough prehistory for lookbacks")
    return config


def run_p1_from_snapshot(snapshot_dir, config_overrides=None, verbose=True):
    """Validate, load and evaluate one immutable JQ index-input snapshot."""

    loaded = load_p1_close_panels(snapshot_dir)
    config = build_snapshot_research_config(loaded, config_overrides)
    results = p1.run_p1(loaded["panels"], config=config, verbose=verbose)
    return results, loaded, config


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_local_results(results, loaded, output_dir):
    """Atomically write versioned local P1 tables and a provenance manifest."""

    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(str(output_dir))
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=".%s.part-" % output_dir.name, dir=str(output_dir.parent))
    )
    try:
        file_records = []
        for table_name in sorted(results):
            frame = results[table_name]
            if not isinstance(frame, pd.DataFrame):
                continue
            filename = "momentum_signal_p1__%s.csv" % table_name
            path = temporary_dir / filename
            frame.to_csv(path, index=False)
            file_records.append(
                {
                    "path": filename,
                    "rows": int(len(frame)),
                    "columns": list(frame.columns),
                    "bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
            )
        manifest = {
            "schema_version": 1,
            "research_id": p1.DEFAULT_CONFIG["research_id"],
            "run_kind": "local_calculation_from_snapshot",
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_path": (
                "research/momentum_signal_validation/local/run_p1_from_snapshot.py"
            ),
            "python_version": sys.version.replace("\n", " "),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "input_dataset_id": loaded["validation"]["dataset_id"],
            "input_manifest_sha256": loaded["validation"]["manifest_sha256"],
            "files": file_records,
        }
        manifest_path = temporary_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(str(temporary_dir), str(output_dir))
    finally:
        if temporary_dir.exists():
            shutil.rmtree(str(temporary_dir))
    return output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run frozen P1 locally from an extracted JQ input snapshot"
    )
    parser.add_argument("snapshot", help="validated extracted snapshot directory")
    parser.add_argument("--output", required=True, help="new versioned output directory")
    parser.add_argument("--quiet", action="store_true", help="suppress P1 progress")
    args = parser.parse_args(argv)
    results, loaded, _ = run_p1_from_snapshot(
        args.snapshot,
        verbose=not args.quiet,
    )
    output_dir = write_local_results(results, loaded, args.output)
    print("local P1 output: %s" % output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
