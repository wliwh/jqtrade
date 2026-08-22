"""Export the P1 index close-price inputs from JQ Research as one tar.gz.

Copy this self-contained file into one JQ Research cell, execute the cell, and
then run ``ARCHIVE_INFO = export_p1_index_inputs()``.  The only persistent
output is one gzip-compressed tar archive.  It contains raw close-price rows,
the frozen universe catalog, the requested trade calendar, coverage diagnostics
and a checksum manifest for local validation.

This is an investment-research exporter, not a JQ backtest or live strategy.
All dates and data semantics are explicit so the downloaded snapshot can be
analysed locally without silently querying a newer point in time.
"""

try:
    from jqdata import *

    JQDATA_AVAILABLE = True
except ImportError:
    JQDATA_AVAILABLE = False

import builtins as _python_builtins
import datetime
import hashlib
import io
import json
import marshal
import os
import sys
import tarfile
import time

import numpy as np
import pandas as pd


_PYTHON_BUILTIN_NAMES = (
    "Exception",
    "FileExistsError",
    "ImportError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "dict",
    "enumerate",
    "float",
    "getattr",
    "globals",
    "hasattr",
    "int",
    "isinstance",
    "len",
    "list",
    "max",
    "min",
    "next",
    "open",
    "print",
    "range",
    "repr",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


def restore_python_builtins(namespace=None):
    """Undo Python built-in names shadowed by ``from jqdata import *``."""

    if namespace is None:
        namespace = _python_builtins.globals()
    for name in _PYTHON_BUILTIN_NAMES:
        namespace[name] = _python_builtins.getattr(_python_builtins, name)
    return namespace


restore_python_builtins()


def _resolve_jq_api(name):
    """Resolve APIs injected into either module globals or Python builtins."""

    namespace = _python_builtins.globals()
    if name in namespace and namespace[name] is not None:
        return namespace[name]
    return _python_builtins.getattr(_python_builtins, name, None)


def _jq_api_source(name):
    namespace = _python_builtins.globals()
    if name in namespace and namespace[name] is not None:
        return "module_globals"
    if _python_builtins.getattr(_python_builtins, name, None) is not None:
        return "python_builtins"
    return "unavailable"


BROAD_INDEX_UNIVERSE = {
    "000001.XSHG": "上证指数",
    "399001.XSHE": "深证成指",
    "000016.XSHG": "上证50",
    "000300.XSHG": "沪深300",
    "000905.XSHG": "中证500",
    "000852.XSHG": "中证1000",
    "000985.XSHG": "中证全指",
    "399006.XSHE": "创业板指",
    "399303.XSHE": "国证2000",
    "000688.XSHG": "科创50",
}

STYLE_INDEX_UNIVERSE = {
    "000918.XSHG": "300成长",
    "000919.XSHG": "300价值",
    "000920.XSHG": "300R成长",
    "000921.XSHG": "300R价值",
    "000922.XSHG": "中证红利",
    "000015.XSHG": "上证红利",
    "399372.XSHE": "大盘成长",
    "399373.XSHE": "大盘价值",
    "399374.XSHE": "中盘成长",
    "399375.XSHE": "中盘价值",
    "399376.XSHE": "小盘成长",
    "399377.XSHE": "小盘价值",
}

DEFAULT_GROUPS = {
    "broad": {
        "source": "get_price",
        "version": "broad_index_v1",
        "members": BROAD_INDEX_UNIVERSE,
        "min_assets": 6,
    },
    "industry_sw_l1": {
        "source": "finance.SW1_DAILY_PRICE",
        "catalog_source": "get_industries:sw_l1:all_historical",
        "version": "all_historical_sw_l1_v1",
        "members": {},
        "min_assets": 20,
    },
    "style": {
        "source": "get_price",
        "version": "style_index_v1",
        "members": STYLE_INDEX_UNIVERSE,
        "min_assets": 6,
    },
}

DEFAULT_CONFIG = {
    "dataset_id": "momentum_index_p1_inputs_v1",
    "schema_version": 1,
    "fetch_start": "2015-06-05",
    "research_start": "2016-01-01",
    "research_end": "2026-08-20",
    "frequency": "daily",
    "fields": ("close",),
    "price_fq": "pre",
    "skip_paused": False,
    "fill_paused": False,
    "group_order": ("broad", "industry_sw_l1", "style"),
    "groups": DEFAULT_GROUPS,
    "output_filename": "momentum_index_p1_inputs_v1_20150605_20260820.tar.gz",
}


def _copy_groups(groups):
    copied = {}
    for group_name, spec in groups.items():
        copied[group_name] = dict(spec)
        copied[group_name]["members"] = dict(spec.get("members", {}))
    return copied


def build_export_config(overrides=None):
    """Build and validate an independent export configuration."""

    config = dict(DEFAULT_CONFIG)
    config["groups"] = _copy_groups(DEFAULT_CONFIG["groups"])
    if overrides:
        for key, value in overrides.items():
            if key == "groups":
                config[key] = _copy_groups(value)
            else:
                config[key] = value
    _validate_export_config(config)
    return config


def _validate_export_config(config):
    if pd.Timestamp(config["fetch_start"]) > pd.Timestamp(config["research_start"]):
        raise ValueError("fetch_start must not be after research_start")
    if pd.Timestamp(config["research_start"]) > pd.Timestamp(config["research_end"]):
        raise ValueError("research_start must not be after research_end")
    if tuple(config["fields"]) != ("close",):
        raise ValueError("P1 input contract currently exports close only")
    if set(config["group_order"]) != set(config["groups"]):
        raise ValueError("group_order must contain every group exactly once")
    if not str(config["output_filename"]).endswith(".tar.gz"):
        raise ValueError("output_filename must end with .tar.gz")
    dataset_id = str(config["dataset_id"])
    if not dataset_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in dataset_id
    ):
        raise ValueError("dataset_id must be a safe ASCII archive directory name")
    for group_name in config["group_order"]:
        spec = config["groups"][group_name]
        if int(spec.get("min_assets", 0)) < 1:
            raise ValueError("min_assets must be positive for %s" % group_name)
        if group_name != "industry_sw_l1" and not spec.get("members"):
            raise ValueError("members must not be empty for %s" % group_name)


def _build_progress_reporter(total_steps, enabled=True):
    total_steps = max(int(total_steps), 1)
    completed = [0]
    started_at = time.time()

    def report(message, advance=True):
        if not enabled:
            return
        if advance:
            completed[0] = min(completed[0] + 1, total_steps)
        elapsed = max(int(time.time() - started_at), 0)
        percentage = int(100.0 * completed[0] / float(total_steps))
        print(
            "[JQ export %3d%% | %02d:%02d:%02d] %s"
            % (
                percentage,
                elapsed // 3600,
                (elapsed % 3600) // 60,
                elapsed % 60,
                message,
            )
        )

    return report


def _wide_close_to_long(wide, securities):
    wide = wide.copy()
    wide.index = pd.to_datetime(wide.index).normalize()
    parts = []
    for code in securities:
        if code not in wide.columns:
            continue
        part = pd.DataFrame(
            {
                "date": wide.index,
                "code": code,
                "close": wide[code].values,
            }
        )
        parts.append(part)
    if not parts:
        raise ValueError("get_price result contains none of the requested codes")
    return pd.concat(parts, ignore_index=True)


def normalize_index_close_rows(raw, securities):
    """Normalize JQ ``get_price(panel=False)`` shapes into date/code/close."""

    if raw is None or len(raw) == 0:
        raise ValueError("get_price returned no rows")
    frame = raw.copy()
    securities = list(securities)

    if isinstance(frame.columns, pd.MultiIndex):
        level_zero = frame.columns.get_level_values(0)
        if "close" not in level_zero:
            raise ValueError("cannot locate close in MultiIndex columns")
        result = _wide_close_to_long(frame["close"], securities)
    else:
        flat = frame.reset_index()
        column_names = set(flat.columns)
        date_column = next(
            (
                name
                for name in ("time", "date", "datetime", "index")
                if name in column_names
            ),
            None,
        )
        if {"code", "close"}.issubset(column_names) and date_column is not None:
            result = flat[[date_column, "code", "close"]].rename(
                columns={date_column: "date"}
            )
        elif "close" in frame.columns and len(securities) == 1:
            result = pd.DataFrame(
                {
                    "date": frame.index,
                    "code": securities[0],
                    "close": frame["close"].values,
                }
            )
        elif securities and set(securities).intersection(frame.columns):
            result = _wide_close_to_long(frame, securities)
        else:
            raise ValueError(
                "unsupported get_price result; expected time/code/close long data"
            )

    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["code"] = result["code"].astype(str)
    unexpected = set(result["code"]) - set(securities)
    if unexpected:
        raise ValueError("get_price returned unexpected codes: %s" % sorted(unexpected))
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result[["date", "code", "close"]].sort_values(
        ["date", "code"]
    ).reset_index(drop=True)
    if result.duplicated(["date", "code"]).any():
        raise ValueError("get_price returned duplicate date/code rows")
    return result


def _fetch_index_group(members, config, get_price_func):
    kwargs = {
        "security": list(members),
        "start_date": config["fetch_start"],
        "end_date": config["research_end"],
        "frequency": config["frequency"],
        "fields": list(config["fields"]),
        "skip_paused": bool(config["skip_paused"]),
        "fq": config["price_fq"],
        "panel": False,
        "fill_paused": bool(config["fill_paused"]),
    }
    try:
        raw = get_price_func(**kwargs)
    except TypeError:
        kwargs.pop("fill_paused")
        raw = get_price_func(**kwargs)
    return normalize_index_close_rows(raw, list(members))


def _historical_sw1_members(get_industries_func):
    catalog = get_industries_func(name="sw_l1")
    if catalog is None or len(catalog) == 0:
        raise ValueError("get_industries('sw_l1') returned no historical industries")
    members = {}
    start_dates = {}
    for code, row in catalog.iterrows():
        code = str(code)
        if not code.startswith("801"):
            continue
        members[code] = str(row.get("name", code))
        start_date = row.get("start_date", None)
        start_dates[code] = "" if pd.isnull(start_date) else str(start_date)[:10]
    if not members:
        raise ValueError("historical sw_l1 catalog contains no 801xxx codes")
    return members, start_dates


def _fetch_sw1_prices_from_jq(codes, start_date, end_date):
    """Fetch one SW L1 code per query to remain below the 5000-row limit."""

    frames = []
    table = finance.SW1_DAILY_PRICE
    for position, code in enumerate(codes, start=1):
        request = query(table.date, table.code, table.close).filter(
            table.code == code,
            table.date >= start_date,
            table.date <= end_date,
        ).order_by(table.date)
        frame = finance.run_query(request)
        if len(frame) >= 5000:
            raise ValueError(
                "SW1 query reached the 5000-row boundary for %s; split by date" % code
            )
        frames.append(frame)
        if position % 5 == 0 or position == len(codes):
            print("  SW L1 close: %d/%d" % (position, len(codes)))
    if not frames:
        raise ValueError("no SW1 price query was executed")
    return pd.concat(frames, ignore_index=True)


def normalize_sw1_close_rows(raw, securities):
    if raw is None or len(raw) == 0:
        raise ValueError("SW1_DAILY_PRICE returned no rows")
    frame = raw.copy()
    required = {"date", "code", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("SW1_DAILY_PRICE is missing columns: %s" % sorted(missing))
    result = frame[["date", "code", "close"]].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["code"] = result["code"].astype(str)
    unexpected = set(result["code"]) - set(securities)
    if unexpected:
        raise ValueError("SW1_DAILY_PRICE returned unexpected codes: %s" % sorted(unexpected))
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result = result.sort_values(["date", "code"]).reset_index(drop=True)
    if result.duplicated(["date", "code"]).any():
        raise ValueError("SW1_DAILY_PRICE returned duplicate date/code rows")
    return result


def _trade_days_frame(values):
    dates = pd.to_datetime(list(values)).normalize()
    if len(dates) == 0:
        raise ValueError("get_trade_days returned no dates")
    frame = pd.DataFrame({"date": dates})
    if frame["date"].duplicated().any():
        raise ValueError("get_trade_days returned duplicate dates")
    return frame.sort_values("date").reset_index(drop=True)


def _build_catalog(config, sw1_members, sw1_start_dates):
    rows = []
    for group_name in config["group_order"]:
        spec = config["groups"][group_name]
        members = sw1_members if group_name == "industry_sw_l1" else spec["members"]
        for code in sorted(members):
            rows.append(
                {
                    "universe_group": group_name,
                    "code": code,
                    "name": members[code],
                    "source": spec["source"],
                    "version": spec["version"],
                    "catalog_start_date": sw1_start_dates.get(code, ""),
                    "min_assets": int(spec["min_assets"]),
                }
            )
    return pd.DataFrame(rows)


def _build_coverage(config, prices_by_group, catalog):
    rows = []
    for catalog_row in catalog.to_dict("records"):
        group_name = catalog_row["universe_group"]
        code = catalog_row["code"]
        prices = prices_by_group[group_name]
        selected = prices[prices["code"].eq(code)]
        valid = selected[selected["close"].notnull()]
        rows.append(
            {
                "universe_group": group_name,
                "code": code,
                "name": catalog_row["name"],
                "rows": int(len(selected)),
                "non_null_close": int(len(valid)),
                "first_valid_date": (
                    "" if valid.empty else valid["date"].min().strftime("%Y-%m-%d")
                ),
                "last_valid_date": (
                    "" if valid.empty else valid["date"].max().strftime("%Y-%m-%d")
                ),
                "status": "available" if len(valid) else "no_valid_close",
            }
        )
    coverage = pd.DataFrame(rows)
    for group_name in config["group_order"]:
        observed = int(
            coverage[
                coverage["universe_group"].eq(group_name)
                & coverage["status"].eq("available")
            ].shape[0]
        )
        minimum = int(config["groups"][group_name]["min_assets"])
        if observed < minimum:
            raise ValueError(
                "%s has only %d assets with valid close, below min_assets=%d"
                % (group_name, observed, minimum)
            )
    return coverage


def fetch_input_tables(
    config=None,
    get_price_func=None,
    get_industries_func=None,
    sw1_price_fetcher=None,
    get_trade_days_func=None,
    progress_func=None,
):
    """Fetch and validate all raw input tables before any archive is written."""

    config = build_export_config(config)
    get_price_func = get_price_func or _resolve_jq_api("get_price")
    get_industries_func = get_industries_func or _resolve_jq_api("get_industries")
    get_trade_days_func = get_trade_days_func or _resolve_jq_api("get_trade_days")
    sw1_price_fetcher = sw1_price_fetcher or _fetch_sw1_prices_from_jq
    if get_price_func is None:
        raise RuntimeError("get_price is unavailable; run this exporter in JQ Research")
    if get_industries_func is None:
        raise RuntimeError(
            "get_industries is unavailable; run this exporter in JQ Research"
        )
    if get_trade_days_func is None:
        raise RuntimeError(
            "get_trade_days is unavailable; run this exporter in JQ Research"
        )

    broad_members = config["groups"]["broad"]["members"]
    style_members = config["groups"]["style"]["members"]
    prices_by_group = {
        "broad": _fetch_index_group(broad_members, config, get_price_func),
    }
    if progress_func is not None:
        progress_func("broad index close rows ready: %d" % len(prices_by_group["broad"]))

    prices_by_group["style"] = _fetch_index_group(
        style_members, config, get_price_func
    )
    if progress_func is not None:
        progress_func("style index close rows ready: %d" % len(prices_by_group["style"]))

    sw1_members, sw1_start_dates = _historical_sw1_members(get_industries_func)
    if progress_func is not None:
        progress_func("historical SW L1 catalog ready: %d codes" % len(sw1_members))
    sw1_raw = sw1_price_fetcher(
        list(sorted(sw1_members)), config["fetch_start"], config["research_end"]
    )
    prices_by_group["industry_sw_l1"] = normalize_sw1_close_rows(
        sw1_raw, list(sw1_members)
    )
    if progress_func is not None:
        progress_func(
            "SW L1 index close rows ready: %d"
            % len(prices_by_group["industry_sw_l1"])
        )

    trade_days = _trade_days_frame(
        get_trade_days_func(
            start_date=config["fetch_start"], end_date=config["research_end"]
        )
    )
    if progress_func is not None:
        progress_func("trade calendar ready: %d dates" % len(trade_days))

    expected_dates = set(trade_days["date"])
    for group_name, prices in prices_by_group.items():
        outside = set(prices["date"]) - expected_dates
        if outside:
            raise ValueError(
                "%s close data contains dates outside the JQ trade calendar"
                % group_name
            )
    catalog = _build_catalog(config, sw1_members, sw1_start_dates)
    coverage = _build_coverage(config, prices_by_group, catalog)
    if progress_func is not None:
        progress_func("catalog and coverage checks complete")
    return {
        "broad_index_close": prices_by_group["broad"],
        "style_index_close": prices_by_group["style"],
        "sw_l1_index_close": prices_by_group["industry_sw_l1"],
        "universe_catalog": catalog,
        "universe_coverage": coverage,
        "trade_days": trade_days,
    }


def _csv_bytes(frame):
    serializable = frame.copy()
    for column in serializable.columns:
        if np.issubdtype(serializable[column].dtype, np.datetime64):
            serializable[column] = serializable[column].dt.strftime("%Y-%m-%d")
    return serializable.to_csv(
        index=False,
        float_format="%.12g",
    ).encode("utf-8-sig")


def _file_record(path, content, frame, date_column=None, key_columns=None):
    record = {
        "path": path,
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "bytes": int(len(content)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "encoding": "utf-8-sig",
    }
    if date_column is not None and len(frame):
        dates = pd.to_datetime(frame[date_column])
        record["date_column"] = date_column
        record["min_date"] = dates.min().strftime("%Y-%m-%d")
        record["max_date"] = dates.max().strftime("%Y-%m-%d")
    if key_columns is not None:
        record["key_columns"] = list(key_columns)
        record["duplicate_keys"] = int(frame.duplicated(list(key_columns)).sum())
    return record


def build_exporter_logic_hash():
    """Hash loaded exporter functions and frozen universe/config constants."""

    digest = hashlib.sha256()
    namespace = _python_builtins.globals()
    for name in sorted(namespace):
        value = namespace[name]
        if (
            callable(value)
            and hasattr(value, "__code__")
            and getattr(value, "__module__", None) == __name__
        ):
            digest.update(name.encode("utf-8"))
            digest.update(marshal.dumps(value.__code__))
    for value in (
        BROAD_INDEX_UNIVERSE,
        STYLE_INDEX_UNIVERSE,
        DEFAULT_GROUPS,
        DEFAULT_CONFIG,
    ):
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _tar_add_bytes(archive, path, content, modification_time):
    info = tarfile.TarInfo(path)
    info.size = len(content)
    info.mode = 0o644
    info.mtime = modification_time
    archive.addfile(info, io.BytesIO(content))


def build_input_archive(tables, config=None):
    """Return one validated tar.gz payload and its self-describing manifest."""

    config = build_export_config(config)
    archive_root = config["dataset_id"]
    table_specs = (
        ("broad_index_close", "data/broad_index_close.csv", "date", ("date", "code")),
        ("style_index_close", "data/style_index_close.csv", "date", ("date", "code")),
        ("sw_l1_index_close", "data/sw_l1_index_close.csv", "date", ("date", "code")),
        ("universe_catalog", "data/universe_catalog.csv", None, ("universe_group", "code")),
        ("universe_coverage", "data/universe_coverage.csv", None, ("universe_group", "code")),
        ("trade_days", "data/trade_days.csv", "date", ("date",)),
    )
    payloads = {}
    records = []
    for table_name, relative_path, date_column, key_columns in table_specs:
        frame = tables[table_name]
        path = "%s/%s" % (archive_root, relative_path)
        content = _csv_bytes(frame)
        payloads[path] = content
        record = _file_record(path, content, frame, date_column, key_columns)
        if record.get("duplicate_keys", 0):
            raise ValueError("duplicate keys in %s" % table_name)
        records.append(record)

    coverage = tables["universe_coverage"]
    available_by_group = {}
    for group_name in config["group_order"]:
        available_by_group[group_name] = int(
            coverage[
                coverage["universe_group"].eq(group_name)
                & coverage["status"].eq("available")
            ].shape[0]
        )
    manifest = {
        "schema_version": int(config["schema_version"]),
        "dataset_id": config["dataset_id"],
        "archive_root": archive_root,
        "archive_format": "tar+gzip",
        "generated_at_jq": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_platform": "JQ Research",
        "source_path": (
            "research/momentum_signal_validation/adapters/jq/"
            "export_p1_index_inputs.py"
        ),
        "exporter_logic_hash": build_exporter_logic_hash(),
        "runtime": {
            "python_version": sys.version.replace("\n", " "),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "jqdata_imported": bool(JQDATA_AVAILABLE),
            "get_price_source": _jq_api_source("get_price"),
            "get_industries_source": _jq_api_source("get_industries"),
            "get_trade_days_source": _jq_api_source("get_trade_days"),
        },
        "request": {
            "fetch_start": config["fetch_start"],
            "research_start": config["research_start"],
            "research_end": config["research_end"],
            "frequency": config["frequency"],
            "fields": list(config["fields"]),
            "price_fq": config["price_fq"],
            "skip_paused": bool(config["skip_paused"]),
            "fill_paused": bool(config["fill_paused"]),
            "sw1_query_batching": "one industry code per query",
            "sw1_query_row_boundary": 5000,
        },
        "universe_versions": {
            group_name: config["groups"][group_name]["version"]
            for group_name in config["group_order"]
        },
        "available_assets_by_group": available_by_group,
        "files": records,
    }
    manifest_path = "%s/manifest.json" % archive_root
    manifest_content = json.dumps(
        manifest, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8")

    buffer = io.BytesIO()
    modification_time = int(time.time())
    with tarfile.open(
        fileobj=buffer,
        mode="w:gz",
        format=tarfile.PAX_FORMAT,
    ) as archive:
        _tar_add_bytes(archive, manifest_path, manifest_content, modification_time)
        for path in sorted(payloads):
            _tar_add_bytes(archive, path, payloads[path], modification_time)
    archive_content = buffer.getvalue()

    expected_paths = set([manifest_path] + list(payloads))
    with tarfile.open(fileobj=io.BytesIO(archive_content), mode="r:gz") as archive:
        members = archive.getmembers()
        actual_paths = set(member.name for member in members)
        if actual_paths != expected_paths:
            raise ValueError("tar member list does not match the manifest contract")
        if any(not member.isfile() for member in members):
            raise ValueError("tar archive contains a non-regular member")
        for record in records:
            extracted = archive.extractfile(record["path"])
            content = extracted.read()
            if hashlib.sha256(content).hexdigest() != record["sha256"]:
                raise ValueError("tar payload checksum mismatch: %s" % record["path"])
    return archive_content, manifest


def export_p1_index_inputs(
    output_dir=".",
    config=None,
    get_price_func=None,
    get_industries_func=None,
    sw1_price_fetcher=None,
    get_trade_days_func=None,
    verbose=True,
):
    """Fetch JQ inputs and atomically write one downloadable tar.gz file."""

    config = build_export_config(config)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.abspath(os.path.join(output_dir, config["output_filename"]))
    if os.path.exists(output_path):
        raise FileExistsError(
            "%s already exists; rename it or advance the dataset version" % output_path
        )
    progress = _build_progress_reporter(8, enabled=verbose)
    progress("configuration validated", advance=False)
    tables = fetch_input_tables(
        config,
        get_price_func=get_price_func,
        get_industries_func=get_industries_func,
        sw1_price_fetcher=sw1_price_fetcher,
        get_trade_days_func=get_trade_days_func,
        progress_func=progress,
    )
    content, manifest = build_input_archive(tables, config)
    progress("tar.gz payload built and internally verified")

    # Repeat the check in case another notebook process created it while fetching.
    if os.path.exists(output_path):
        raise FileExistsError(
            "%s already exists; rename it or advance the dataset version" % output_path
        )
    temporary_path = output_path + ".part"
    try:
        with open(temporary_path, "wb") as output_file:
            output_file.write(content)
        with tarfile.open(temporary_path, mode="r:gz") as archive:
            if len(archive.getmembers()) != 1 + len(manifest["files"]):
                raise ValueError("written tar.gz member count is incorrect")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    progress("single archive written: %s" % output_path)

    result = {
        "path": output_path,
        "bytes": int(len(content)),
        "sha256": hashlib.sha256(content).hexdigest(),
        "members": int(1 + len(manifest["files"])),
        "dataset_id": config["dataset_id"],
        "manifest": manifest,
    }
    if verbose:
        print("download this one file: %s" % output_path)
        print("archive sha256: %s" % result["sha256"])
        print("archive size: %.2f MiB" % (result["bytes"] / 1048576.0))
    return result


if __name__ == "__main__":
    export_p1_index_inputs()
