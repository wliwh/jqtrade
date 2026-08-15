import datetime as dt
import hashlib
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
from jqdata import *

# ---------------------------------------------------------------------------
# Configuration: freeze these values before evaluating turning-point results.
# ---------------------------------------------------------------------------

DATA_VERSION = "all_a_breadth_v1"
START_DATE = "2012-01-01"
END_DATE = "2026-08-14"
UNIVERSE_INDEX = "000985.XSHG"
INDUSTRY_LEVEL = "sw_l1"
MA_WINDOWS = (20, 60, 120)
MIN_INDUSTRY_VALID_COUNT = 5
SECURITY_BATCH_SIZE = 500
PRICE_FQ = "pre"

TARGET_INDUSTRIES = {
    "bank": "银行",
    "nonferrous": "有色金属",
    "steel": "钢铁",
    "coal": "煤炭",
}

OUTPUT_FILENAME = "%s_%s_%s.zip" % (
    DATA_VERSION,
    START_DATE.replace("-", ""),
    END_DATE.replace("-", ""),
)


def _batched(values, batch_size):
    values = list(values)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _date_index(values):
    return pd.DatetimeIndex(pd.to_datetime(list(values))).normalize()


def _normalize_long_prices(data):
    """Normalize the multi-security ``get_price(panel=False)`` result."""

    if data is None or data.empty:
        return pd.DataFrame(columns=["time", "code", "close", "paused"])

    result = data.copy()
    if "time" not in result.columns:
        index_name = result.index.name or "index"
        result = result.reset_index().rename(columns={index_name: "time"})
    required = {"time", "code", "close", "paused"}
    missing = required.difference(result.columns)
    if missing:
        raise ValueError("get_price result is missing columns: %s" % sorted(missing))
    result["time"] = pd.to_datetime(result["time"]).dt.normalize()
    return result[["time", "code", "close", "paused"]]


def _query_price_matrices(securities, start_date, end_date, trade_days):
    close_parts = []
    paused_parts = []
    for batch in _batched(sorted(securities), SECURITY_BATCH_SIZE):
        data = get_price(
            batch,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=["close", "paused"],
            skip_paused=False,
            fq=PRICE_FQ,
            panel=False,
            fill_paused=True,
        )
        data = _normalize_long_prices(data)
        if data.empty:
            continue
        close_parts.append(data.pivot(index="time", columns="code", values="close"))
        paused_parts.append(data.pivot(index="time", columns="code", values="paused"))

    if not close_parts:
        raise RuntimeError("JQ get_price returned no data")

    index = _date_index(trade_days)
    columns = sorted(securities)
    close = pd.concat(close_parts, axis=1).reindex(index=index, columns=columns)
    paused = pd.concat(paused_parts, axis=1).reindex(index=index, columns=columns)
    return close.astype(float), paused.astype(float)


def _query_st_matrix(securities, start_date, end_date, trade_days):
    parts = []
    for batch in _batched(sorted(securities), SECURITY_BATCH_SIZE):
        data = get_extras(
            "is_st",
            batch,
            start_date=start_date,
            end_date=end_date,
            df=True,
        )
        if data is None or data.empty:
            continue
        part = data.copy()
        part.index = pd.to_datetime(part.index).normalize()
        parts.append(part)

    if not parts:
        raise RuntimeError("JQ get_extras('is_st') returned no data")

    return pd.concat(parts, axis=1).reindex(
        index=_date_index(trade_days),
        columns=sorted(securities),
    )


def _industry_map(securities, date):
    raw = get_industry(sorted(securities), date=date)
    result = {}
    for security in securities:
        level = (raw.get(security) or {}).get(INDUSTRY_LEVEL)
        if not level:
            result[security] = (None, None)
            continue
        result[security] = (
            level.get("industry_code"),
            level.get("industry_name"),
        )
    return result


def _normalized_industry_name(value):
    if not isinstance(value, str):
        return None
    value = value.strip().replace(" ", "")
    if value.endswith("I") or value.endswith("Ⅰ"):
        value = value[:-1]
    return value


def _target_id(industry_name):
    normalized = _normalized_industry_name(industry_name)
    for target_id, target_name in TARGET_INDUSTRIES.items():
        if normalized == target_name:
            return target_id
    return None


def _finite(series):
    values = pd.to_numeric(series, errors="coerce").values
    return pd.Series(
        np.isfinite(np.asarray(values, dtype=float)),
        index=series.index,
    )


def _safe_ratio(numerator, denominator):
    if not denominator:
        return np.nan
    return float(numerator) / float(denominator)


def aggregate_day(
    date,
    universe,
    close,
    paused,
    is_st,
    moving_averages,
    industries,
    min_industry_valid_count=MIN_INDUSTRY_VALID_COUNT,
):
    """Aggregate one date without calling JQ APIs.

    This pure function is intentionally public so the denominator, exclusion,
    ranking, and tie rules can be tested locally.
    """

    if min_industry_valid_count <= 0:
        raise ValueError("min_industry_valid_count must be positive")
    missing_windows = set(MA_WINDOWS).difference(moving_averages)
    if missing_windows:
        raise ValueError("missing moving averages: %s" % sorted(missing_windows))

    universe = sorted(set(universe))
    frame = pd.DataFrame(index=pd.Index(universe, name="security"))
    frame["close"] = pd.to_numeric(close.reindex(universe), errors="coerce")
    frame["paused"] = pd.to_numeric(paused.reindex(universe), errors="coerce")
    frame["is_st"] = is_st.reindex(universe)
    frame["industry_code"] = [industries.get(code, (None, None))[0] for code in universe]
    frame["industry_name"] = [industries.get(code, (None, None))[1] for code in universe]

    close_known = _finite(frame["close"])
    paused_known = frame["paused"].notna()
    st_known = frame["is_st"].notna()
    is_paused = paused_known & frame["paused"].ne(0)
    st_flag = st_known & frame["is_st"].fillna(False).astype(bool)
    base_valid = close_known & paused_known & ~is_paused & st_known & ~st_flag
    industry_known = frame["industry_code"].notna() & frame["industry_name"].notna()

    date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
    summary = {
        "date": date_text,
        "universe_size": len(frame),
        "close_missing_count": int((~close_known).sum()),
        "paused_count": int(is_paused.sum()),
        "paused_status_missing_count": int((~paused_known).sum()),
        "st_count": int(st_flag.sum()),
        "st_status_missing_count": int((~st_known).sum()),
        "base_valid_count": int(base_valid.sum()),
        "base_valid_missing_industry_count": int((base_valid & ~industry_known).sum()),
    }

    valid_by_window = {}
    above_by_window = {}
    for window in MA_WINDOWS:
        ma = pd.to_numeric(moving_averages[window].reindex(universe), errors="coerce")
        frame["ma%d" % window] = ma
        valid = base_valid & _finite(ma)
        above = valid & frame["close"].gt(ma)
        valid_by_window[window] = valid
        above_by_window[window] = above
        valid_count = int(valid.sum())
        above_count = int(above.sum())
        summary["insufficient_history_count_ma%d" % window] = int(
            (base_valid & ~_finite(ma)).sum()
        )
        summary["valid_count_ma%d" % window] = valid_count
        summary["above_count_ma%d" % window] = above_count
        summary["breadth_ma%d" % window] = _safe_ratio(above_count, valid_count)

    industry_records = []
    grouped = frame.loc[industry_known].groupby(
        ["industry_code", "industry_name"],
        sort=True,
    )
    for (industry_code, industry_name), group in grouped:
        codes = group.index
        record = {
            "date": date_text,
            "industry_code": industry_code,
            "industry_name": industry_name,
            "universe_count": len(codes),
            "base_valid_count": int(base_valid.reindex(codes).sum()),
        }
        for window in MA_WINDOWS:
            valid_count = int(valid_by_window[window].reindex(codes).sum())
            above_count = int(above_by_window[window].reindex(codes).sum())
            record["valid_count_ma%d" % window] = valid_count
            record["above_count_ma%d" % window] = above_count
            record["breadth_ma%d" % window] = _safe_ratio(above_count, valid_count)
        industry_records.append(record)

    industries_frame = pd.DataFrame(industry_records)
    if industries_frame.empty:
        industries_frame = pd.DataFrame(
            columns=["date", "industry_code", "industry_name"]
        )
        eligible_rank = pd.Series(dtype=bool)
    else:
        eligible_rank = (
            industries_frame["valid_count_ma20"].ge(min_industry_valid_count)
            & industries_frame["breadth_ma20"].notna()
        )
    industries_frame["rank_eligible_ma20"] = eligible_rank
    industries_frame["rank_ma20"] = np.nan
    if eligible_rank.any():
        industries_frame.loc[eligible_rank, "rank_ma20"] = industries_frame.loc[
            eligible_rank, "breadth_ma20"
        ].rank(method="min", ascending=False)
    industries_frame["is_top1_ma20"] = industries_frame["rank_ma20"].eq(1)
    industries_frame["target_id"] = industries_frame.get(
        "industry_name", pd.Series(dtype=object)
    ).map(_target_id)
    industries_frame["is_target_industry"] = industries_frame["target_id"].notna()

    top1 = industries_frame.loc[industries_frame["is_top1_ma20"]].sort_values(
        ["industry_code", "industry_name"]
    )
    summary["ranked_industry_count_ma20"] = int(eligible_rank.sum())
    summary["top1_tie_count_ma20"] = len(top1)
    summary["top1_industry_codes_ma20"] = "|".join(
        top1.get("industry_code", pd.Series(dtype=str)).astype(str)
    )
    summary["top1_industry_names_ma20"] = "|".join(
        top1.get("industry_name", pd.Series(dtype=str)).astype(str)
    )

    triggered_ids = []
    present_count = 0
    for target_id, target_name in TARGET_INDUSTRIES.items():
        matches = industries_frame.loc[industries_frame["target_id"].eq(target_id)].copy()
        prefix = "target_%s_" % target_id
        summary[prefix + "expected_name"] = target_name
        summary[prefix + "mapping_count"] = len(matches)
        summary[prefix + "industry_code"] = ""
        summary[prefix + "industry_name"] = ""
        summary[prefix + "valid_count_ma20"] = 0
        summary[prefix + "breadth_ma20"] = np.nan
        summary[prefix + "rank_ma20"] = np.nan
        summary[prefix + "is_top1_ma20"] = False
        if matches.empty:
            continue
        present_count += 1
        selected = matches.sort_values(
            ["valid_count_ma20", "industry_code"],
            ascending=[False, True],
        ).iloc[0]
        summary[prefix + "industry_code"] = selected["industry_code"]
        summary[prefix + "industry_name"] = selected["industry_name"]
        summary[prefix + "valid_count_ma20"] = int(selected["valid_count_ma20"])
        summary[prefix + "breadth_ma20"] = selected["breadth_ma20"]
        summary[prefix + "rank_ma20"] = selected["rank_ma20"]
        summary[prefix + "is_top1_ma20"] = bool(selected["is_top1_ma20"])
        if bool(selected["is_top1_ma20"]):
            triggered_ids.append(target_id)

    summary["four_industry_present_count"] = present_count
    summary["four_industry_top1_triggered"] = bool(triggered_ids)
    summary["four_industry_top1_ids"] = "|".join(sorted(triggered_ids))
    return summary, industries_frame


def _point_in_time_universes(trade_days):
    result = {}
    for position, date in enumerate(trade_days, start=1):
        securities = sorted(set(get_index_stocks(UNIVERSE_INDEX, date=date)))
        if not securities:
            raise RuntimeError(
                "%s returned no constituents on %s; verify the JQ index code"
                % (UNIVERSE_INDEX, date)
            )
        result[pd.Timestamp(date).normalize()] = securities
        if position % 50 == 0 or position == len(trade_days):
            print("  constituents: %d/%d" % (position, len(trade_days)))
    return result


def _process_trade_day_chunk(trade_days):
    trade_days = list(trade_days)
    if not trade_days:
        return pd.DataFrame(), pd.DataFrame()

    output_index = _date_index(trade_days)
    universes = _point_in_time_universes(trade_days)
    securities = sorted({code for codes in universes.values() for code in codes})
    warmup = list(get_trade_days(end_date=trade_days[0], count=max(MA_WINDOWS)))
    if not warmup:
        raise RuntimeError("JQ returned no warm-up trade days")
    price_days = list(get_trade_days(start_date=warmup[0], end_date=trade_days[-1]))

    close, paused = _query_price_matrices(
        securities,
        price_days[0],
        price_days[-1],
        price_days,
    )
    st = _query_st_matrix(
        securities,
        trade_days[0],
        trade_days[-1],
        trade_days,
    )
    moving_averages = {
        window: close.rolling(window=window, min_periods=window).mean()
        for window in MA_WINDOWS
    }

    daily_records = []
    industry_frames = []
    for position, date in enumerate(output_index, start=1):
        universe = universes[date]
        summary, industries_frame = aggregate_day(
            date=date,
            universe=universe,
            close=close.loc[date],
            paused=paused.loc[date],
            is_st=st.loc[date],
            moving_averages={
                window: moving_averages[window].loc[date]
                for window in MA_WINDOWS
            },
            industries=_industry_map(universe, date.date()),
        )
        daily_records.append(summary)
        industry_frames.append(industries_frame)
        if position % 20 == 0 or position == len(output_index):
            print("  aggregation: %d/%d" % (position, len(output_index)))

    return pd.DataFrame(daily_records), pd.concat(industry_frames, ignore_index=True)


def _csv_bytes(frame):
    return frame.to_csv(
        index=False,
        float_format="%.10g",
    ).encode("utf-8-sig")


def _file_record(path, content, frame):
    return {
        "path": path,
        "rows": len(frame),
        "columns": list(frame.columns),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "encoding": "utf-8-sig",
    }


def build_archive(daily, industry, start_date, end_date):
    """Return validated ZIP bytes for the two processed tables."""

    daily_content = _csv_bytes(daily)
    industry_content = _csv_bytes(industry)
    files = [
        _file_record("data/daily_summary.csv", daily_content, daily),
        _file_record("data/industry_breadth.csv", industry_content, industry),
    ]
    manifest = {
        "data_version": DATA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "availability": "Each row uses date-t close and is available after date-t close.",
        "query": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "frequency": "daily",
            "universe": {
                "method": "point_in_time_index_constituents",
                "index": UNIVERSE_INDEX,
            },
            "price_fields": ["close", "paused"],
            "price_fq": PRICE_FQ,
            "fill_paused": True,
            "st_source": "get_extras:is_st",
            "industry_source": "get_industry:%s" % INDUSTRY_LEVEL,
            "ma_windows": list(MA_WINDOWS),
            "industry_rank_window": 20,
            "industry_rank_method": "min_descending",
            "min_industry_valid_count": MIN_INDUSTRY_VALID_COUNT,
            "targets": TARGET_INDUSTRIES,
            "old_mining_mapped_to_coal": False,
        },
        "denominators": {
            "market": (
                "Index constituent with finite close, known non-paused status, "
                "known non-ST status, and enough adjusted-close history for the MA window."
            ),
            "industry": (
                "Market-valid stock with a point-in-time SW level-1 assignment; "
                "at least five MA20-valid stocks are required to enter Top1 ranking."
            ),
        },
        "export_level": {
            "daily_summary": "one row per trade date",
            "industry_breadth": "one row per trade date and observed SW level-1 industry",
            "stock_level_rows_exported": False,
        },
        "files": files,
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        archive.writestr("data/daily_summary.csv", daily_content)
        archive.writestr("data/industry_breadth.csv", industry_content)

    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
        broken_member = archive.testzip()
        if broken_member is not None:
            raise ValueError("ZIP validation failed: %s" % broken_member)
    return payload


def _validate_outputs(daily, industry, expected_dates):
    expected = pd.Index(pd.to_datetime(expected_dates).strftime("%Y-%m-%d"))
    actual = pd.Index(daily["date"].astype(str))
    if not actual.is_unique or not actual.equals(expected):
        raise ValueError("daily_summary dates do not match requested trade days")
    if industry.duplicated(["date", "industry_code", "industry_name"]).any():
        raise ValueError("duplicate date-industry rows")
    for window in MA_WINDOWS:
        for frame, column in (
            (daily, "breadth_ma%d" % window),
            (industry, "breadth_ma%d" % window),
        ):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.between(0.0, 1.0).all():
                raise ValueError("breadth outside [0, 1]: %s" % column)


def run_export(
    start_date=START_DATE,
    end_date=END_DATE,
    output_path=OUTPUT_FILENAME,
):
    """Run annual chunks and write one final, self-describing ZIP."""

    trade_days = list(get_trade_days(start_date=start_date, end_date=end_date))
    if not trade_days:
        raise RuntimeError("JQ returned no trade days for the requested range")
    if os.path.exists(output_path):
        raise FileExistsError(
            "%s already exists; remove or rename it before rerunning" % output_path
        )

    daily_parts = []
    industry_parts = []
    years = sorted({day.year for day in trade_days})
    for year in years:
        chunk = [day for day in trade_days if day.year == year]
        print("processing %d: %s to %s" % (year, chunk[0], chunk[-1]))
        daily, industry = _process_trade_day_chunk(chunk)
        daily_parts.append(daily)
        industry_parts.append(industry)

    daily = pd.concat(daily_parts, ignore_index=True)
    industry = pd.concat(industry_parts, ignore_index=True)
    _validate_outputs(daily, industry, trade_days)
    payload = build_archive(daily, industry, trade_days[0], trade_days[-1])

    temporary_path = output_path + ".part"
    try:
        with open(temporary_path, "wb") as file:
            file.write(payload)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print("archive: %s" % output_path)
    print("daily rows: %d" % len(daily))
    print("industry rows: %d" % len(industry))
    return output_path


if __name__ == "__main__":
    run_export()
