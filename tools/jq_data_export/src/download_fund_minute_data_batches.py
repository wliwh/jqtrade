"""分批下载 JQ 平台的 ETF/LOF 历史 1 分钟数据。

脚本每次运行只生成一个 ZIP。ZIP 存在时脚本会暂停；人工下载并删除
该 ZIP 后，再次运行即可从状态文件记录的位置继续。

本文件可独立放入 JQ 环境运行，不依赖项目中的其他 Python 文件。
"""

import datetime as dt
import hashlib
import json
import os
import posixpath
import shutil
import zipfile

from jqdata import get_all_securities, get_price


# -------------------- 在这里修改下载参数 --------------------
# None 表示从每只基金的上市日开始；底层脚本仍会限制到 2005-01-01 以后。
START_DATE = None

# 必须填写固定日期。2026-04-27 是当前 TDX 1m 起点，因此默认只抓此前历史。
END_DATE = "2026-04-26"

# JQ 运行目录下的相对目录；不要在人工清理时删除其中的 state.json。
OUTPUT_DIR = "fund_minute_batches"

# 累计 CSV 原始大小达到或刚超过该值后打包并暂停。
BATCH_TARGET_MB = 512

# 单只基金仍按自然日分片请求 JQ，最后合并成一个 CSV 放入 ZIP。
CHUNK_DAYS = 60

# 官方文档建议先取全部场内基金，再按返回的 type 过滤。
FUND_TYPES = ("etf", "lof")

# 历史入库必须固定为未复权。不要改为 pre/post。
FQ = None

STATE_VERSION = 1
STATE_FILENAME = "state.json"
MANIFEST_FILENAME = "manifest.json"
JQ_MINUTE_DATA_START = dt.date(2005, 1, 1)


def _now_text():
    return dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def _parse_date(value, name, allow_none=False):
    if value is None and allow_none:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        value = value.strip()
        for date_format in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                return dt.datetime.strptime(value, date_format).date()
            except ValueError:
                pass
    raise ValueError("%s 必须是固定的 YYYY-MM-DD 日期" % name)


def _date_text(value):
    if value is None:
        return None
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    return str(value)[:10]


def _normalize_relative_dir(path):
    path = (path or "").strip().replace("\\", "/")
    normalized = posixpath.normpath(path)
    if normalized in ("", "."):
        raise ValueError("OUTPUT_DIR 不能为空，状态文件和压缩包需要独立目录")
    if posixpath.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
        raise ValueError("OUTPUT_DIR 必须是 JQ 运行目录下的相对目录")
    return normalized.rstrip("/")


def _human_size(size_bytes):
    value = float(size_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return "%.2f %s" % (value, unit)
        value /= 1024.0


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path, value):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary_path = path + ".tmp"
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _settings(
    start_date,
    end_date,
    output_dir,
    batch_target_mb,
    chunk_days,
    fund_types,
):
    parsed_start = _parse_date(start_date, "START_DATE", allow_none=True)
    parsed_end = _parse_date(end_date, "END_DATE")
    if parsed_end >= dt.date.today():
        raise ValueError("END_DATE 必须早于今天，不能随运行日期自动推进")
    if parsed_start is not None and parsed_start > parsed_end:
        raise ValueError("START_DATE 不能晚于 END_DATE")
    if not isinstance(chunk_days, int) or isinstance(chunk_days, bool) or chunk_days <= 0:
        raise ValueError("CHUNK_DAYS 必须是正整数")
    if not isinstance(batch_target_mb, (int, float)) or isinstance(batch_target_mb, bool):
        raise ValueError("BATCH_TARGET_MB 必须是正数")
    if batch_target_mb <= 0:
        raise ValueError("BATCH_TARGET_MB 必须是正数")

    normalized_types = sorted(set(str(item).strip().lower() for item in fund_types))
    if not normalized_types or not set(normalized_types).issubset({"etf", "lof"}):
        raise ValueError("FUND_TYPES 只能包含 etf 和 lof")

    return {
        "start_date": _date_text(parsed_start),
        "end_date": _date_text(parsed_end),
        "output_dir": _normalize_relative_dir(output_dir),
        "batch_target_bytes": int(float(batch_target_mb) * 1024 * 1024),
        "chunk_days": chunk_days,
        "fund_types": normalized_types,
        "frequency": "1m",
        "fq": None,
        "skip_paused": False,
    }


def _build_universe(fund_types):
    funds = get_all_securities(types=["fund"], date=None)
    if funds is None or funds.empty:
        raise ValueError("get_all_securities(['fund']) 未返回基金列表")

    required_columns = {"display_name", "name", "start_date", "end_date", "type"}
    missing_columns = sorted(required_columns.difference(funds.columns))
    if missing_columns:
        raise ValueError("基金列表缺少字段：%s" % ", ".join(missing_columns))

    selected = funds[funds["type"].isin(fund_types)].sort_index()
    universe = []
    for security, row in selected.iterrows():
        universe.append(
            {
                "security": str(security).strip().upper(),
                "display_name": str(row["display_name"]),
                "name": str(row["name"]),
                "fund_type": str(row["type"]).strip().lower(),
                "start_date": _date_text(row["start_date"]),
                "end_date": _date_text(row["end_date"]),
            }
        )

    if not universe:
        raise ValueError("基金列表中没有找到类型：%s" % ", ".join(fund_types))
    return universe


def _iter_date_chunks(start_date, end_date, chunk_days):
    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + dt.timedelta(days=chunk_days - 1),
            end_date,
        )
        yield chunk_start, chunk_end
        chunk_start = chunk_end + dt.timedelta(days=1)


def _download_one_chunk(security, chunk_start, chunk_end):
    data = get_price(
        security=security,
        start_date=chunk_start.strftime("%Y-%m-%d 00:00:00"),
        end_date=chunk_end.strftime("%Y-%m-%d 23:59:59"),
        frequency="1m",
        fields=["open", "close", "high", "low", "volume", "money", "paused"],
        skip_paused=False,
        fq=None,
    )
    if data is None or data.empty:
        return None

    data = data.copy()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data.index.name = "datetime"
    data.insert(0, "security", security)
    return data


def _merge_csv_files(chunk_paths, output_path):
    temporary_path = output_path + ".tmp"
    expected_header = None
    try:
        with open(temporary_path, "w", encoding="utf-8", newline="") as output_file:
            for chunk_path in chunk_paths:
                with open(chunk_path, "r", encoding="utf-8", newline="") as chunk_file:
                    header = chunk_file.readline()
                    if not header:
                        continue
                    if expected_header is None:
                        expected_header = header
                        output_file.write(header)
                    elif header != expected_header:
                        raise ValueError("分片字段不一致，无法合并：%s" % chunk_path)
                    for line in chunk_file:
                        output_file.write(line)

        if expected_header is None:
            raise ValueError("所有日期片均为空，没有可合并的数据")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _download_security_minute_data(
    security,
    effective_start,
    effective_end,
    output_dir,
    chunk_days,
):
    """独立完成单只基金的未复权 1 分钟分片下载和流式合并。"""
    safe_security = security.replace(".", "_")
    range_tag = "%s_%s" % (
        effective_start.strftime("%Y%m%d"),
        effective_end.strftime("%Y%m%d"),
    )
    filename = "%s_1m_%s.csv" % (safe_security, range_tag)
    output_path = posixpath.join(output_dir, filename)
    chunk_dir = posixpath.join(output_dir, "chunks", safe_security)
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_paths = []
    total_rows = 0
    chunks = list(_iter_date_chunks(effective_start, effective_end, chunk_days))
    for number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            "  下载分片 %d/%d：%s 至 %s"
            % (number, len(chunks), chunk_start, chunk_end)
        )
        try:
            data = _download_one_chunk(security, chunk_start, chunk_end)
        except Exception as error:
            raise RuntimeError(
                "分片 %d/%d 下载失败（%s 至 %s）"
                % (number, len(chunks), chunk_start, chunk_end)
            ) from error
        if data is None:
            print("    该分片没有行情数据，跳过")
            continue

        chunk_filename = "%s_1m_part_%04d_%s_%s.csv" % (
            safe_security,
            number,
            chunk_start.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        )
        chunk_path = posixpath.join(chunk_dir, chunk_filename)
        with open(chunk_path, "w", encoding="utf-8", newline="") as file:
            file.write(data.to_csv())
        chunk_paths.append(chunk_path)
        total_rows += len(data)

    if not chunk_paths:
        raise ValueError("全部日期片均未获取到行情数据")

    _merge_csv_files(chunk_paths, output_path)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    return output_path, total_rows


def _new_batch(number):
    return {
        "number": number,
        "raw_bytes": 0,
        "files": [],
    }


def _new_state(settings, universe):
    return {
        "version": STATE_VERSION,
        "created_at": _now_text(),
        "updated_at": _now_text(),
        "status": "running",
        "settings": settings,
        "universe_source": "get_all_securities(['fund'], date=None), filter type",
        "universe": universe,
        "next_index": 0,
        "current_batch": _new_batch(1),
        "pending_archive": None,
        "released_archives": [],
        "skipped": [],
        "failures": {},
        "all_items_processed": False,
    }


def _save_state(state_path, state):
    state["updated_at"] = _now_text()
    _atomic_write_json(state_path, state)


def _load_or_create_state(state_path, settings):
    if os.path.exists(state_path):
        state = _read_json(state_path)
        if state.get("version") != STATE_VERSION:
            raise ValueError("状态文件版本不兼容，请保留旧文件并更换 OUTPUT_DIR")
        if state.get("settings") != settings:
            raise ValueError(
                "当前参数与状态文件不一致。为避免混合口径，请恢复原参数，"
                "或更换 OUTPUT_DIR 开始新任务"
            )
        return state

    universe = _build_universe(settings["fund_types"])
    state = _new_state(settings, universe)
    _save_state(state_path, state)
    return state


def _effective_download_range(item, settings):
    requested_start = _parse_date(settings["start_date"], "start_date", allow_none=True)
    requested_end = _parse_date(settings["end_date"], "end_date")
    listing_date = _parse_date(item["start_date"], "基金上市日期")
    security_end = _parse_date(item["end_date"], "基金结束日期")
    effective_start = max(listing_date, JQ_MINUTE_DATA_START)
    if requested_start is not None:
        effective_start = max(effective_start, requested_start)
    effective_end = min(requested_end, security_end)
    if effective_start > effective_end:
        return None
    return effective_start, effective_end


def _batch_work_dir(output_dir, batch_number):
    return posixpath.join(output_dir, "work", "batch_%04d" % batch_number)


def _archive_name(batch):
    first_position = batch["files"][0]["universe_position"] + 1
    last_position = batch["files"][-1]["universe_position"] + 1
    return "jq_etf_lof_1m_batch_%04d_%06d-%06d.zip" % (
        batch["number"],
        first_position,
        last_position,
    )


def _archive_manifest(state, batch):
    files = []
    for item in batch["files"]:
        files.append(
            {
                "security": item["security"],
                "display_name": item["display_name"],
                "fund_type": item["fund_type"],
                "rows": item["rows"],
                "bytes": item["bytes"],
                "sha256": item["sha256"],
                "archive_member": "data/%s" % os.path.basename(item["path"]),
                "listing_date": item["listing_date"],
                "security_end_date": item["security_end_date"],
            }
        )
    return {
        "manifest_version": 1,
        "created_at": _now_text(),
        "source": "JoinQuant",
        "universe_source": state["universe_source"],
        "batch_number": batch["number"],
        "query": {
            "frequency": "1m",
            "start_date": state["settings"]["start_date"],
            "end_date": state["settings"]["end_date"],
            "fq": None,
            "skip_paused": False,
            "fields": ["open", "close", "high", "low", "volume", "money", "paused"],
        },
        "raw_bytes": batch["raw_bytes"],
        "file_count": len(files),
        "files": files,
    }


def _create_archive(state, state_path, final_batch=False):
    batch = state["current_batch"]
    if not batch["files"]:
        raise ValueError("当前批次没有文件，无法创建压缩包")

    output_dir = state["settings"]["output_dir"]
    archive_path = posixpath.join(output_dir, _archive_name(batch))
    temporary_path = archive_path + ".tmp"
    manifest = _archive_manifest(state, batch)
    os.makedirs(output_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(
            temporary_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            archive.writestr(
                MANIFEST_FILENAME,
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            )
            for item in batch["files"]:
                archive.write(item["path"], "data/%s" % os.path.basename(item["path"]))

        with zipfile.ZipFile(temporary_path, mode="r") as archive:
            broken_member = archive.testzip()
            if broken_member is not None:
                raise ValueError("ZIP 校验失败：%s" % broken_member)

        os.replace(temporary_path, archive_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    archive_info = {
        "batch_number": batch["number"],
        "path": archive_path,
        "bytes": os.path.getsize(archive_path),
        "sha256": _sha256(archive_path),
        "file_count": len(batch["files"]),
        "raw_bytes": batch["raw_bytes"],
        "work_dir": _batch_work_dir(output_dir, batch["number"]),
        "created_at": _now_text(),
        "final_batch": bool(final_batch),
    }
    state["pending_archive"] = archive_info
    state["status"] = "awaiting_archive_cleanup"
    state["all_items_processed"] = bool(final_batch)
    state["current_batch"] = _new_batch(batch["number"] + 1)
    _save_state(state_path, state)

    # ZIP 已通过 CRC 检查并且状态已落盘，工作 CSV 可安全清理以节省空间。
    shutil.rmtree(archive_info["work_dir"], ignore_errors=True)
    print("已生成压缩包：%s" % archive_path)
    print(
        "  %d 个文件，CSV 合计 %s，ZIP %s"
        % (
            archive_info["file_count"],
            _human_size(archive_info["raw_bytes"]),
            _human_size(archive_info["bytes"]),
        )
    )
    print("请人工下载并核对压缩包，然后只删除该 ZIP；保留 %s" % state_path)
    print("删除 ZIP 后再次运行本脚本即可继续。")
    return archive_info


def _handle_pending_archive(state, state_path):
    pending = state.get("pending_archive")
    if pending is None:
        return True

    shutil.rmtree(pending.get("work_dir", ""), ignore_errors=True)
    archive_path = pending["path"]
    if os.path.exists(archive_path):
        print("等待人工处理压缩包：%s" % archive_path)
        print("下载并核对后删除该 ZIP，再次运行脚本。不要删除 %s" % state_path)
        return False

    released = dict(pending)
    released.pop("work_dir", None)
    released["released_at"] = _now_text()
    state["released_archives"].append(released)
    state["pending_archive"] = None
    if state.get("all_items_processed"):
        state["status"] = "completed"
    else:
        state["status"] = "running"
    _save_state(state_path, state)
    return True


def run_batch_download(
    start_date=START_DATE,
    end_date=END_DATE,
    output_dir=OUTPUT_DIR,
    batch_target_mb=BATCH_TARGET_MB,
    chunk_days=CHUNK_DAYS,
    fund_types=FUND_TYPES,
):
    """推进一个下载批次；生成 ZIP 后返回，等待人工下载和删除。"""
    settings = _settings(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        batch_target_mb=batch_target_mb,
        chunk_days=chunk_days,
        fund_types=fund_types,
    )
    os.makedirs(settings["output_dir"], exist_ok=True)
    state_path = posixpath.join(settings["output_dir"], STATE_FILENAME)
    state = _load_or_create_state(state_path, settings)

    if not _handle_pending_archive(state, state_path):
        return "awaiting_archive_cleanup"
    if state.get("status") == "completed":
        print("全部完成：%d 个标的已处理" % len(state["universe"]))
        print("状态文件：%s" % state_path)
        return "completed"

    batch = state["current_batch"]
    if batch["files"] and batch["raw_bytes"] >= settings["batch_target_bytes"]:
        _create_archive(state, state_path, final_batch=False)
        return "archive_created"

    universe = state["universe"]
    while state["next_index"] < len(universe):
        position = state["next_index"]
        item = universe[position]
        security = item["security"]

        effective_range = _effective_download_range(item, settings)
        if effective_range is None:
            state["skipped"].append(
                {
                    "security": security,
                    "reason": "requested_range_outside_security_lifetime",
                    "recorded_at": _now_text(),
                }
            )
            state["next_index"] += 1
            _save_state(state_path, state)
            print("跳过 %s：请求范围与上市区间不重叠" % security)
            continue

        batch = state["current_batch"]
        work_dir = _batch_work_dir(settings["output_dir"], batch["number"])
        print(
            "处理 %d/%d：%s %s (%s)"
            % (
                position + 1,
                len(universe),
                security,
                item["display_name"],
                item["fund_type"],
            )
        )
        try:
            saved_path, row_count = _download_security_minute_data(
                security=security,
                effective_start=effective_range[0],
                effective_end=effective_range[1],
                output_dir=work_dir,
                chunk_days=settings["chunk_days"],
            )
        except Exception as error:
            previous = state["failures"].get(security, {})
            state["failures"][security] = {
                "attempts": int(previous.get("attempts", 0)) + 1,
                "last_error": "%s: %s" % (type(error).__name__, error),
                "last_failed_at": _now_text(),
                "universe_position": position,
            }
            state["status"] = "failed"
            _save_state(state_path, state)
            raise RuntimeError(
                "批量下载在 %s 失败；状态已保存，修复原因后重新运行会重试该标的"
                % security
            ) from error

        file_size = os.path.getsize(saved_path)
        batch["files"].append(
            {
                "security": security,
                "display_name": item["display_name"],
                "fund_type": item["fund_type"],
                "listing_date": item["start_date"],
                "security_end_date": item["end_date"],
                "universe_position": position,
                "path": saved_path,
                "rows": int(row_count),
                "bytes": file_size,
                "sha256": _sha256(saved_path),
            }
        )
        batch["raw_bytes"] += file_size
        state["next_index"] += 1
        state["status"] = "running"
        state["failures"].pop(security, None)
        _save_state(state_path, state)
        print(
            "  当前批次：%d 个文件，%s / %s"
            % (
                len(batch["files"]),
                _human_size(batch["raw_bytes"]),
                _human_size(settings["batch_target_bytes"]),
            )
        )

        if batch["raw_bytes"] >= settings["batch_target_bytes"]:
            _create_archive(state, state_path, final_batch=False)
            return "archive_created"

    state["all_items_processed"] = True
    batch = state["current_batch"]
    if batch["files"]:
        _create_archive(state, state_path, final_batch=True)
        return "archive_created"

    state["status"] = "completed"
    _save_state(state_path, state)
    print("全部完成：%d 个标的已处理" % len(universe))
    return "completed"


if __name__ == "__main__":
    run_batch_download()
