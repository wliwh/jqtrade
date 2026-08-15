"""分片下载 JoinQuant 单只股票的长时间分钟线，并合并为一个 CSV。"""

import datetime as dt
import os
import posixpath

from jqdata import get_price, get_security_info


# -------------------- 在这里修改下载参数 --------------------
SECURITY = "159915.XSHE"
START_DATE = "2020-01-01"      # 可早于上市日；程序会自动调整
END_DATE = "2020-07-01"        # 填 None 时下载到昨天
OUTPUT_DIR = "minute_data"     # 运行目录下的相对目录；当前目录填 ""
CHUNK_DAYS = 60                # 每片包含的自然日数
KEEP_CHUNK_FILES = True        # 合并后是否保留分片 CSV
FQ = None                      # "pre" 前复权、"post" 后复权、None 不复权

# 聚宽 A 股分钟行情的可用历史通常从 2005 年开始。
JQ_MINUTE_DATA_START = dt.date(2005, 1, 1)

FIELDS = [
    "open",
    "close",
    "high",
    "low",
    "volume",
    "money",
    "paused",
]


def write_file(path, content, append=False):
    """将文本或二进制内容写入本地文件，并自动创建父目录。"""
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    is_binary = isinstance(content, (bytes, bytearray))
    mode = ("ab" if append else "wb") if is_binary else ("a" if append else "w")
    open_kwargs = {} if is_binary else {"encoding": "utf-8", "newline": ""}

    with open(path, mode, **open_kwargs) as file:
        file.write(content)


def _as_date(value, name):
    """把 YYYY-MM-DD、date 或 datetime 统一转换为 date。"""
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
    raise ValueError("%s 必须是 YYYY-MM-DD、date 或 datetime" % name)


def _normalize_output_dir(output_dir):
    """检查并规范运行环境中的相对目录。"""
    output_dir = (output_dir or "").strip().replace("\\", "/")
    normalized = posixpath.normpath(output_dir)
    if normalized in ("", "."):
        return ""
    if posixpath.isabs(normalized) or normalized == ".." or normalized.startswith("../"):
        raise ValueError("OUTPUT_DIR 必须是运行环境中的相对目录")
    return normalized.rstrip("/")


def resolve_download_range(security, start_date, end_date):
    """结合用户设置、上市/退市日和数据边界，确定实际下载日期。"""
    security_info = get_security_info(security)
    if security_info is None:
        raise ValueError("找不到证券信息，请检查代码：%s" % security)

    listing_date = _as_date(security_info.start_date, "上市日期")
    latest_complete_date = dt.date.today() - dt.timedelta(days=1)

    requested_start = listing_date if start_date is None else _as_date(start_date, "START_DATE")
    requested_end = latest_complete_date if end_date is None else _as_date(end_date, "END_DATE")
    if requested_start > requested_end:
        raise ValueError("START_DATE 不能晚于 END_DATE")

    effective_start = max(requested_start, listing_date, JQ_MINUTE_DATA_START)
    effective_end = min(requested_end, latest_complete_date)

    # 已退市证券的 end_date 是最后有效日期；在市证券通常是很远的未来日期。
    security_end = getattr(security_info, "end_date", None)
    if security_end is not None:
        effective_end = min(effective_end, _as_date(security_end, "证券结束日期"))

    if effective_start > effective_end:
        raise ValueError(
            "设置的日期范围内没有可下载数据；证券上市日为 %s" % listing_date
        )

    return effective_start, effective_end, listing_date


def iter_date_chunks(start_date, end_date, chunk_days):
    """把闭区间 [start_date, end_date] 切成互不重叠的日期片。"""
    if not isinstance(chunk_days, int) or isinstance(chunk_days, bool) or chunk_days <= 0:
        raise ValueError("CHUNK_DAYS 必须是正整数")

    chunk_start = start_date
    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + dt.timedelta(days=chunk_days - 1),
            end_date,
        )
        yield chunk_start, chunk_end
        chunk_start = chunk_end + dt.timedelta(days=1)


def _download_one_chunk(security, chunk_start, chunk_end, fq):
    """下载一个日期片，并清理片内重复时间戳。"""
    data = get_price(
        security=security,
        start_date=chunk_start.strftime("%Y-%m-%d 00:00:00"),
        end_date=chunk_end.strftime("%Y-%m-%d 23:59:59"),
        frequency="1m",
        fields=FIELDS,
        skip_paused=False,
        fq=fq,
    )

    if data is None or data.empty:
        return None

    data = data.copy()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data.index.name = "datetime"
    data.insert(0, "security", security)
    return data


def _merge_csv_files(chunk_paths, output_path):
    """按顺序流式合并 CSV，避免把全部分钟数据同时载入内存。"""
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

        # 只有完整合并成功后才替换目标文件。
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def download_stock_minute_data(
    security,
    start_date=None,
    end_date=None,
    output_dir="minute_data",
    chunk_days=60,
    fq="pre",
    keep_chunk_files=True,
):
    """分片下载长时间分钟线，全部成功后合并并返回文件路径和行数。"""
    if not isinstance(security, str) or not security.strip():
        raise ValueError("security 不能为空，例如：000001.XSHE")

    security = security.strip().upper()
    output_dir = _normalize_output_dir(output_dir)
    effective_start, effective_end, listing_date = resolve_download_range(
        security,
        start_date,
        end_date,
    )

    if start_date is not None and _as_date(start_date, "START_DATE") < effective_start:
        print(
            "起始日已从 %s 调整为 %s（上市日：%s，分钟数据边界：%s）"
            % (start_date, effective_start, listing_date, JQ_MINUTE_DATA_START)
        )

    safe_security = security.replace(".", "_")
    range_tag = "%s_%s" % (
        effective_start.strftime("%Y%m%d"),
        effective_end.strftime("%Y%m%d"),
    )
    filename = "%s_1m_%s.csv" % (safe_security, range_tag)
    output_path = posixpath.join(output_dir, filename) if output_dir else filename
    chunk_dir = posixpath.join(output_dir, "chunks", safe_security) if output_dir else posixpath.join("chunks", safe_security)
    os.makedirs(chunk_dir, exist_ok=True)

    chunks = list(iter_date_chunks(effective_start, effective_end, chunk_days))
    chunk_paths = []
    total_rows = 0

    for number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(
            "下载分片 %d/%d：%s 至 %s"
            % (number, len(chunks), chunk_start, chunk_end)
        )
        try:
            data = _download_one_chunk(security, chunk_start, chunk_end, fq)
        except Exception as error:
            raise RuntimeError(
                "分片 %d/%d 下载失败（%s 至 %s）"
                % (number, len(chunks), chunk_start, chunk_end)
            ) from error

        if data is None:
            print("  该分片没有行情数据，跳过")
            continue

        chunk_filename = "%s_1m_part_%04d_%s_%s.csv" % (
            safe_security,
            number,
            chunk_start.strftime("%Y%m%d"),
            chunk_end.strftime("%Y%m%d"),
        )
        chunk_path = posixpath.join(chunk_dir, chunk_filename)
        write_file(chunk_path, data.to_csv(), append=False)
        chunk_paths.append(chunk_path)
        total_rows += len(data)
        print("  已保存 %d 行：%s" % (len(data), chunk_path))

    if not chunk_paths:
        raise ValueError("全部日期片均未获取到行情数据")

    _merge_csv_files(chunk_paths, output_path)

    if not keep_chunk_files:
        for chunk_path in chunk_paths:
            os.remove(chunk_path)

    return output_path, total_rows


if __name__ == "__main__":
    saved_path, row_count = download_stock_minute_data(
        security=SECURITY,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=OUTPUT_DIR,
        chunk_days=CHUNK_DAYS,
        fq=FQ,
        keep_chunk_files=KEEP_CHUNK_FILES,
    )
    print("全部完成：共 %d 行分钟数据，合并文件：%s" % (row_count, saved_path))
