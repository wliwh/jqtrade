import ast
import builtins as python_builtins
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import types

import pandas as pd
import pytest

from research.momentum_signal_validation.adapters.jq import export_p1_index_inputs as exporter
from research.momentum_signal_validation.local.validate_p1_index_archive import (
    validate_p1_index_archive,
    validate_p1_index_directory,
)
from research.momentum_signal_validation.local.run_p1_from_snapshot import (
    load_p1_close_panels,
    run_p1_from_snapshot,
    write_local_results,
)


def _fake_jq_inputs():
    trade_days = list(pd.bdate_range("2020-01-02", "2020-01-08").date)
    sw_codes = ["801%03d" % number for number in range(10, 30)]
    sw_catalog = pd.DataFrame(
        {
            "name": ["行业%02dI" % number for number in range(len(sw_codes))],
            "start_date": ["2004-02-09"] * len(sw_codes),
        },
        index=sw_codes,
    )

    def fake_get_price(**kwargs):
        assert kwargs["panel"] is False
        assert kwargs["fields"] == ["close"]
        return pd.DataFrame(
            [
                {
                    "time": date,
                    "code": code,
                    "close": 100.0 + code_number + date_number / 10.0,
                }
                for code_number, code in enumerate(kwargs["security"])
                for date_number, date in enumerate(trade_days)
            ]
        )

    def fake_get_industries(name):
        assert name == "sw_l1"
        return sw_catalog

    def fake_sw1_prices(codes, start_date, end_date):
        assert codes == sorted(sw_codes)
        return pd.DataFrame(
            [
                {
                    "date": date,
                    "code": code,
                    "close": 200.0 + code_number + date_number / 10.0,
                }
                for code_number, code in enumerate(codes)
                for date_number, date in enumerate(trade_days)
            ]
        )

    def fake_get_trade_days(start_date, end_date):
        return trade_days

    return (
        fake_get_price,
        fake_get_industries,
        fake_sw1_prices,
        fake_get_trade_days,
        trade_days,
        sw_codes,
    )


def _short_config():
    return {
        "fetch_start": "2020-01-02",
        "research_start": "2020-01-06",
        "research_end": "2020-01-08",
        "output_filename": "test_momentum_index_inputs.tar.gz",
    }


def test_jq_export_writes_one_self_describing_tar_gz(tmp_path):
    (
        fake_get_price,
        fake_get_industries,
        fake_sw1_prices,
        fake_get_trade_days,
        trade_days,
        sw_codes,
    ) = _fake_jq_inputs()

    result = exporter.export_p1_index_inputs(
        output_dir=str(tmp_path),
        config=_short_config(),
        get_price_func=fake_get_price,
        get_industries_func=fake_get_industries,
        sw1_price_fetcher=fake_sw1_prices,
        get_trade_days_func=fake_get_trade_days,
        verbose=False,
    )

    output_files = list(tmp_path.iterdir())
    assert output_files == [Path(result["path"])]
    assert output_files[0].name.endswith(".tar.gz")
    assert result["members"] == 7
    assert not list(tmp_path.glob("*.part"))

    with tarfile.open(result["path"], mode="r:gz") as archive:
        names = set(archive.getnames())
        root = exporter.DEFAULT_CONFIG["dataset_id"]
        assert names == {
            "%s/manifest.json" % root,
            "%s/data/broad_index_close.csv" % root,
            "%s/data/style_index_close.csv" % root,
            "%s/data/sw_l1_index_close.csv" % root,
            "%s/data/universe_catalog.csv" % root,
            "%s/data/universe_coverage.csv" % root,
            "%s/data/trade_days.csv" % root,
        }
        manifest = json.loads(
            archive.extractfile("%s/manifest.json" % root).read().decode("utf-8")
        )

    assert manifest["request"]["fetch_start"] == "2020-01-02"
    assert manifest["request"]["research_end"] == "2020-01-08"
    assert manifest["archive_format"] == "tar+gzip"
    assert manifest["available_assets_by_group"] == {
        "broad": len(exporter.BROAD_INDEX_UNIVERSE),
        "industry_sw_l1": len(sw_codes),
        "style": len(exporter.STYLE_INDEX_UNIVERSE),
    }
    assert len(trade_days) == 5


def test_local_validator_checks_archive_members_and_hashes(tmp_path):
    apis = _fake_jq_inputs()
    result = exporter.export_p1_index_inputs(
        output_dir=str(tmp_path),
        config=_short_config(),
        get_price_func=apis[0],
        get_industries_func=apis[1],
        sw1_price_fetcher=apis[2],
        get_trade_days_func=apis[3],
        verbose=False,
    )

    checked = validate_p1_index_archive(result["path"])

    assert checked["dataset_id"] == exporter.DEFAULT_CONFIG["dataset_id"]
    assert checked["archive_sha256"] == result["sha256"]
    assert checked["members"] == result["members"]
    assert checked["payload_files"] == 6


def test_local_validator_checks_an_extracted_snapshot_directory(tmp_path):
    apis = _fake_jq_inputs()
    result = exporter.export_p1_index_inputs(
        output_dir=str(tmp_path),
        config=_short_config(),
        get_price_func=apis[0],
        get_industries_func=apis[1],
        sw1_price_fetcher=apis[2],
        get_trade_days_func=apis[3],
        verbose=False,
    )
    with tarfile.open(result["path"], mode="r:gz") as archive:
        archive.extractall(str(tmp_path))

    snapshot_dir = tmp_path / exporter.DEFAULT_CONFIG["dataset_id"]
    checked = validate_p1_index_directory(snapshot_dir)

    assert checked["dataset_id"] == exporter.DEFAULT_CONFIG["dataset_id"]
    assert checked["members"] == result["members"]
    assert checked["payload_files"] == 6


def test_extracted_snapshot_validator_rejects_modified_payload(tmp_path):
    apis = _fake_jq_inputs()
    result = exporter.export_p1_index_inputs(
        output_dir=str(tmp_path),
        config=_short_config(),
        get_price_func=apis[0],
        get_industries_func=apis[1],
        sw1_price_fetcher=apis[2],
        get_trade_days_func=apis[3],
        verbose=False,
    )
    with tarfile.open(result["path"], mode="r:gz") as archive:
        archive.extractall(str(tmp_path))
    snapshot_dir = tmp_path / exporter.DEFAULT_CONFIG["dataset_id"]
    target = snapshot_dir / "data" / "trade_days.csv"
    target.write_bytes(target.read_bytes() + b"2020-01-09\n")

    with pytest.raises(ValueError, match="byte count mismatch"):
        validate_p1_index_directory(snapshot_dir)


def test_local_loader_builds_ordered_panels_from_extracted_snapshot(tmp_path):
    apis = _fake_jq_inputs()
    result = exporter.export_p1_index_inputs(
        output_dir=str(tmp_path),
        config=_short_config(),
        get_price_func=apis[0],
        get_industries_func=apis[1],
        sw1_price_fetcher=apis[2],
        get_trade_days_func=apis[3],
        verbose=False,
    )
    with tarfile.open(result["path"], mode="r:gz") as archive:
        archive.extractall(str(tmp_path))
    snapshot_dir = tmp_path / exporter.DEFAULT_CONFIG["dataset_id"]

    loaded = load_p1_close_panels(snapshot_dir)

    assert loaded["panels"]["broad"].shape == (
        5,
        len(exporter.BROAD_INDEX_UNIVERSE),
    )
    assert loaded["panels"]["industry_sw_l1"].shape == (5, 20)
    assert loaded["panels"]["style"].shape == (
        5,
        len(exporter.STYLE_INDEX_UNIVERSE),
    )
    assert list(loaded["panels"]["broad"].columns) == sorted(
        exporter.BROAD_INDEX_UNIVERSE
    )


def test_local_runner_writes_research_results_with_input_manifest_provenance(tmp_path):
    apis = _fake_jq_inputs()
    result = exporter.export_p1_index_inputs(
        output_dir=str(tmp_path),
        config=_short_config(),
        get_price_func=apis[0],
        get_industries_func=apis[1],
        sw1_price_fetcher=apis[2],
        get_trade_days_func=apis[3],
        verbose=False,
    )
    with tarfile.open(result["path"], mode="r:gz") as archive:
        archive.extractall(str(tmp_path))
    snapshot_dir = tmp_path / exporter.DEFAULT_CONFIG["dataset_id"]
    overrides = {
        "periods": {
            "development": ("2020-01-06", "2020-01-06"),
            "validation": ("2020-01-07", "2020-01-07"),
            "locked_oos": ("2020-01-08", "2020-01-08"),
        },
        "lookbacks": (2,),
        "horizons": (1,),
        "primary_lookback": 2,
        "primary_horizon": 1,
    }

    results, loaded, _ = run_p1_from_snapshot(
        snapshot_dir,
        config_overrides=overrides,
        verbose=False,
    )

    output_dir = tmp_path / "local_results_v1"
    written = write_local_results(results, loaded, output_dir)
    manifest = json.loads((written / "manifest.json").read_text(encoding="utf-8"))
    assert written == output_dir.resolve()
    assert manifest["run_kind"] == "local_calculation_from_snapshot"
    assert manifest["input_dataset_id"] == exporter.DEFAULT_CONFIG["dataset_id"]
    assert manifest["input_manifest_sha256"] == loaded["validation"][
        "manifest_sha256"
    ]
    assert len(manifest["files"]) == len(
        [frame for frame in results.values() if isinstance(frame, pd.DataFrame)]
    )


def test_local_validator_rejects_unsafe_tar_member(tmp_path):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(str(archive_path), mode="w:gz") as archive:
        content = b"bad"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    with pytest.raises(ValueError, match="unsafe member path"):
        validate_p1_index_archive(archive_path)


def test_export_refuses_to_overwrite_an_existing_snapshot(tmp_path):
    apis = _fake_jq_inputs()
    kwargs = {
        "output_dir": str(tmp_path),
        "config": _short_config(),
        "get_price_func": apis[0],
        "get_industries_func": apis[1],
        "sw1_price_fetcher": apis[2],
        "get_trade_days_func": apis[3],
        "verbose": False,
    }
    exporter.export_p1_index_inputs(**kwargs)

    with pytest.raises(FileExistsError, match="already exists"):
        exporter.export_p1_index_inputs(**kwargs)


def test_jq_exporter_stays_python36_and_old_pandas_compatible():
    source_path = Path(exporter.__file__)
    source = source_path.read_text(encoding="utf-8")

    ast.parse(source, filename=str(source_path), feature_version=(3, 6))
    assert "from __future__ import annotations" not in source
    assert "pd.NA" not in source
    assert ".to_numpy(" not in source
    assert "groupby(dropna=" not in source
    assert "to_csv(lineterminator=" not in source


def test_jq_exporter_restores_wildcard_polluted_builtins(monkeypatch):
    source_path = Path(exporter.__file__)
    fake_jqdata = types.ModuleType("jqdata")
    fake_jqdata.__all__ = ["any", "min", "max", "sum"]
    fake_jqdata.any = lambda values: True
    fake_jqdata.min = lambda values: -999
    fake_jqdata.max = lambda values: 999
    fake_jqdata.sum = lambda values: -1
    monkeypatch.setitem(sys.modules, "jqdata", fake_jqdata)

    spec = importlib.util.spec_from_file_location(
        "p1_exporter_jq_polluted", str(source_path)
    )
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)

    assert imported.JQDATA_AVAILABLE is True
    assert imported.any is python_builtins.any
    assert imported.min is python_builtins.min
    assert imported.max is python_builtins.max
    assert imported.sum is python_builtins.sum
