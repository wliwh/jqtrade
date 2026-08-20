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
# This file must remain directly copyable into the JQ research environment.
# ---------------------------------------------------------------------------

DATA_VERSION = "all_a_p1_inputs_v2"
START_DATE = "2012-01-01"
END_DATE = "2026-08-14"
UNIVERSE_INDEX = "000985.XSHG"
INDUSTRY_LEVEL = "sw_l1"
MA_WINDOWS = (20, 60, 120)
HIGH_LOW_WINDOWS = (60, 120, 250)
TURNOVER_EXTREME_THRESHOLDS_PCT = (5.0, 10.0, 20.0)
MIN_INDUSTRY_VALID_COUNT = 5
SECURITY_BATCH_SIZE = 500
VALUATION_MAX_ROWS_PER_QUERY = 4500
PRICE_FQ = "pre"
MA_COMPARISON_RELATIVE_TOLERANCE = 1e-12
MA_COMPARISON_ABSOLUTE_TOLERANCE = 1e-12
LIMIT_PRICE_RELATIVE_TOLERANCE = 1e-6
EXTREME_PRICE_RELATIVE_TOLERANCE = 1e-10

PRICE_FIELDS = (
    "close",
    "high",
    "low",
    "high_limit",
    "low_limit",
    "paused",
)
VALUATION_FIELDS = (
    "turnover_ratio",
    "circulating_market_cap",
)

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

    columns = ["time", "code"] + list(PRICE_FIELDS)
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)

    result = data.copy()
    if "time" not in result.columns:
        index_name = result.index.name or "index"
        result = result.reset_index().rename(columns={index_name: "time"})
    required = set(columns)
    missing = required.difference(result.columns)
    if missing:
        raise ValueError("get_price result is missing columns: %s" % sorted(missing))
    result["time"] = pd.to_datetime(result["time"]).dt.normalize()
    if result.duplicated(["time", "code"]).any():
        raise ValueError("get_price returned duplicate time-code rows")
    return result[columns]


def _query_price_matrices(securities, start_date, end_date, trade_days):
    parts = {field: [] for field in PRICE_FIELDS}
    for batch in _batched(sorted(securities), SECURITY_BATCH_SIZE):
        data = get_price(
            batch,
            start_date=start_date,
            end_date=end_date,
            frequency="daily",
            fields=list(PRICE_FIELDS),
            skip_paused=False,
            fq=PRICE_FQ,
            panel=False,
            fill_paused=True,
        )
        data = _normalize_long_prices(data)
        if data.empty:
            continue
        for field in PRICE_FIELDS:
            parts[field].append(
                data.pivot(index="time", columns="code", values=field)
            )

    if not parts["close"]:
        raise RuntimeError("JQ get_price returned no data")

    index = _date_index(trade_days)
    columns = sorted(securities)
    matrices = {}
    for field in PRICE_FIELDS:
        matrices[field] = pd.concat(parts[field], axis=1).reindex(
            index=index,
            columns=columns,
        ).astype(float)
    return matrices


def _normalize_valuations(data):
    columns = ["day", "code"] + list(VALUATION_FIELDS)
    if data is None or data.empty:
        return pd.DataFrame(columns=columns)

    result = data.copy()
    required = set(columns)
    missing = required.difference(result.columns)
    if missing:
        raise ValueError("get_valuation result is missing columns: %s" % sorted(missing))
    result["day"] = pd.to_datetime(result["day"]).dt.normalize()
    if result.duplicated(["day", "code"]).any():
        raise ValueError("get_valuation returned duplicate day-code rows")
    return result[columns]


def _query_valuation_matrices(securities, start_date, end_date, trade_days):
    trade_days = list(trade_days)
    if not trade_days:
        raise ValueError("trade_days must not be empty")
    safe_batch_size = max(
        1,
        VALUATION_MAX_ROWS_PER_QUERY // len(trade_days),
    )
    safe_batch_size = min(SECURITY_BATCH_SIZE, safe_batch_size)

    parts = {field: [] for field in VALUATION_FIELDS}
    for batch in _batched(sorted(securities), safe_batch_size):
        data = get_valuation(
            batch,
            start_date=start_date,
            end_date=end_date,
            fields=list(VALUATION_FIELDS),
        )
        data = _normalize_valuations(data)
        if len(data) > VALUATION_MAX_ROWS_PER_QUERY:
            raise RuntimeError(
                "get_valuation exceeded the safe row limit: %d" % len(data)
            )
        if data.empty:
            continue
        for field in VALUATION_FIELDS:
            parts[field].append(
                data.pivot(index="day", columns="code", values=field)
            )

    if not parts["turnover_ratio"]:
        raise RuntimeError("JQ get_valuation returned no turnover data")

    index = _date_index(trade_days)
    columns = sorted(securities)
    matrices = {}
    for field in VALUATION_FIELDS:
        matrices[field] = pd.concat(parts[field], axis=1).reindex(
            index=index,
            columns=columns,
        ).astype(float)
    return matrices


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


def _price_feature_matrices(close, high, low):
    """Build causal rolling feature masks from current and earlier rows only."""

    result = {
        "ma_complete": {},
        "above_ma": {},
        "high_low_complete": {},
        "new_high": {},
        "new_low": {},
    }
    for window in MA_WINDOWS:
        moving_average = close.rolling(window=window, min_periods=window).mean()
        result["ma_complete"][window] = moving_average.notnull()
        comparison_tolerance = (
            moving_average.abs() * MA_COMPARISON_RELATIVE_TOLERANCE
            + MA_COMPARISON_ABSOLUTE_TOLERANCE
        )
        result["above_ma"][window] = close.sub(moving_average).gt(
            comparison_tolerance
        )

    for window in HIGH_LOW_WINDOWS:
        rolling_high = high.rolling(window=window, min_periods=window).max()
        rolling_low = low.rolling(window=window, min_periods=window).min()
        complete = rolling_high.notnull() & rolling_low.notnull()
        result["high_low_complete"][window] = complete
        result["new_high"][window] = complete & high.ge(
            rolling_high * (1.0 - EXTREME_PRICE_RELATIVE_TOLERANCE)
        )
        result["new_low"][window] = complete & low.le(
            rolling_low * (1.0 + EXTREME_PRICE_RELATIVE_TOLERANCE)
        )
    return result


def _series_bool(series, index):
    return series.reindex(index).fillna(False).astype(bool)


def _add_turnover_summary(summary, base_valid, turnover_ratio, circulating_cap):
    turnover = pd.to_numeric(
        turnover_ratio.reindex(base_valid.index),
        errors="coerce",
    )
    cap = pd.to_numeric(
        circulating_cap.reindex(base_valid.index),
        errors="coerce",
    )
    valid = base_valid & _finite(turnover) & turnover.ge(0.0)
    values = np.asarray(turnover.loc[valid].values, dtype=float)

    summary["turnover_valid_count"] = int(valid.sum())
    for name in ("mean", "p25", "p50", "p75", "p90", "p95"):
        summary["turnover_ratio_pct_%s" % name] = np.nan
    if len(values):
        summary["turnover_ratio_pct_mean"] = float(np.mean(values))
        for name, percentile in (
            ("p25", 25),
            ("p50", 50),
            ("p75", 75),
            ("p90", 90),
            ("p95", 95),
        ):
            summary["turnover_ratio_pct_%s" % name] = float(
                np.percentile(values, percentile)
            )

    weighted_valid = valid & _finite(cap) & cap.gt(0.0)
    summary["turnover_cap_weight_valid_count"] = int(weighted_valid.sum())
    summary["turnover_ratio_pct_cap_weighted_mean"] = np.nan
    if weighted_valid.any():
        weights = np.asarray(cap.loc[weighted_valid].values, dtype=float)
        weighted_values = np.asarray(
            turnover.loc[weighted_valid].values,
            dtype=float,
        )
        if float(weights.sum()) > 0.0:
            summary["turnover_ratio_pct_cap_weighted_mean"] = float(
                np.average(weighted_values, weights=weights)
            )

    for threshold in TURNOVER_EXTREME_THRESHOLDS_PCT:
        label = ("%g" % threshold).replace(".", "p")
        count = int((valid & turnover.ge(threshold)).sum())
        summary["turnover_ge_%spct_count" % label] = count
        summary["turnover_ge_%spct_ratio" % label] = _safe_ratio(
            count,
            int(valid.sum()),
        )


def aggregate_day(
    date,
    universe,
    prices,
    is_st,
    price_features,
    valuations,
    industries,
    min_industry_valid_count=MIN_INDUSTRY_VALID_COUNT,
):
    """Aggregate all P1 point-in-time inputs for one trade date."""

    index = pd.Index(universe)
    frame = pd.DataFrame(index=index)
    for field in PRICE_FIELDS:
        frame[field] = pd.to_numeric(
            prices[field].reindex(index),
            errors="coerce",
        )
    st = is_st.reindex(index)
    frame["is_st"] = st
    frame["industry_code"] = [industries.get(code, (None, None))[0] for code in index]
    frame["industry_name"] = [industries.get(code, (None, None))[1] for code in index]

    close_finite = _finite(frame["close"])
    paused_known = frame["paused"].notnull()
    st_known = frame["is_st"].notnull()
    paused = frame["paused"].fillna(1).astype(bool)
    st_value = frame["is_st"].fillna(True).astype(bool)
    base_valid = close_finite & paused_known & ~paused & st_known & ~st_value
    industry_known = (
        frame["industry_code"].notnull()
        & frame["industry_name"].notnull()
    )

    date_text = pd.Timestamp(date).strftime("%Y-%m-%d")
    summary = {
        "date": date_text,
        "universe_size": len(index),
        "close_missing_count": int((~close_finite).sum()),
        "high_missing_count": int((~_finite(frame["high"])).sum()),
        "low_missing_count": int((~_finite(frame["low"])).sum()),
        "paused_count": int((paused_known & paused).sum()),
        "paused_status_missing_count": int((~paused_known).sum()),
        "st_count": int((st_known & st_value).sum()),
        "st_status_missing_count": int((~st_known).sum()),
        "base_valid_count": int(base_valid.sum()),
        "base_valid_missing_industry_count": int(
            (base_valid & ~industry_known).sum()
        ),
    }

    valid_by_window = {}
    above_by_window = {}
    for window in MA_WINDOWS:
        complete = _series_bool(price_features["ma_complete"][window], index)
        above = _series_bool(price_features["above_ma"][window], index)
        valid = base_valid & complete
        above = valid & above
        valid_by_window[window] = valid
        above_by_window[window] = above
        valid_count = int(valid.sum())
        above_count = int(above.sum())
        summary["insufficient_history_count_ma%d" % window] = int(
            (base_valid & ~complete).sum()
        )
        summary["valid_count_ma%d" % window] = valid_count
        summary["above_count_ma%d" % window] = above_count
        summary["breadth_ma%d" % window] = _safe_ratio(
            above_count,
            valid_count,
        )

    high_finite = _finite(frame["high"])
    low_finite = _finite(frame["low"])
    for window in HIGH_LOW_WINDOWS:
        complete = _series_bool(
            price_features["high_low_complete"][window],
            index,
        )
        valid = base_valid & high_finite & low_finite & complete
        new_high = valid & _series_bool(
            price_features["new_high"][window],
            index,
        )
        new_low = valid & _series_bool(
            price_features["new_low"][window],
            index,
        )
        valid_count = int(valid.sum())
        high_count = int(new_high.sum())
        low_count = int(new_low.sum())
        summary["insufficient_history_count_high_low_%d" % window] = int(
            (base_valid & ~complete).sum()
        )
        summary["valid_count_high_low_%d" % window] = valid_count
        summary["new_high_count_%d" % window] = high_count
        summary["new_low_count_%d" % window] = low_count
        summary["new_high_ratio_%d" % window] = _safe_ratio(
            high_count,
            valid_count,
        )
        summary["new_low_ratio_%d" % window] = _safe_ratio(
            low_count,
            valid_count,
        )
        summary["new_high_low_net_count_%d" % window] = high_count - low_count
        summary["new_high_low_net_ratio_%d" % window] = _safe_ratio(
            high_count - low_count,
            valid_count,
        )

    limit_fields_finite = (
        close_finite
        & high_finite
        & low_finite
        & _finite(frame["high_limit"])
        & _finite(frame["low_limit"])
        & frame["high_limit"].gt(0.0)
        & frame["low_limit"].gt(0.0)
    )
    limit_valid = base_valid & limit_fields_finite
    limit_up_hit = limit_valid & frame["high"].ge(
        frame["high_limit"] * (1.0 - LIMIT_PRICE_RELATIVE_TOLERANCE)
    )
    limit_down_hit = limit_valid & frame["low"].le(
        frame["low_limit"] * (1.0 + LIMIT_PRICE_RELATIVE_TOLERANCE)
    )
    limit_up_close = limit_valid & frame["close"].ge(
        frame["high_limit"] * (1.0 - LIMIT_PRICE_RELATIVE_TOLERANCE)
    )
    limit_down_close = limit_valid & frame["close"].le(
        frame["low_limit"] * (1.0 + LIMIT_PRICE_RELATIVE_TOLERANCE)
    )
    limit_valid_count = int(limit_valid.sum())
    summary["limit_price_missing_count"] = int(
        (base_valid & ~limit_fields_finite).sum()
    )
    summary["valid_count_limit"] = limit_valid_count
    for name, mask in (
        ("limit_up_hit", limit_up_hit),
        ("limit_down_hit", limit_down_hit),
        ("limit_up_close", limit_up_close),
        ("limit_down_close", limit_down_close),
    ):
        count = int(mask.sum())
        summary[name + "_count"] = count
        summary[name + "_ratio"] = _safe_ratio(count, limit_valid_count)
    summary["limit_hit_net_count"] = int(limit_up_hit.sum() - limit_down_hit.sum())
    summary["limit_hit_net_ratio"] = _safe_ratio(
        summary["limit_hit_net_count"],
        limit_valid_count,
    )
    summary["limit_close_net_count"] = int(
        limit_up_close.sum() - limit_down_close.sum()
    )
    summary["limit_close_net_ratio"] = _safe_ratio(
        summary["limit_close_net_count"],
        limit_valid_count,
    )

    _add_turnover_summary(
        summary,
        base_valid,
        valuations["turnover_ratio"],
        valuations["circulating_market_cap"],
    )

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
            record["breadth_ma%d" % window] = _safe_ratio(
                above_count,
                valid_count,
            )
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
            & industries_frame["breadth_ma20"].notnull()
        )
    industries_frame["rank_eligible_ma20"] = eligible_rank
    industries_frame["rank_ma20"] = np.nan
    if eligible_rank.any():
        industries_frame.loc[eligible_rank, "rank_ma20"] = industries_frame.loc[
            eligible_rank,
            "breadth_ma20",
        ].rank(method="min", ascending=False)
    industries_frame["is_top1_ma20"] = industries_frame["rank_ma20"].eq(1)
    industries_frame["target_id"] = industries_frame.get(
        "industry_name",
        pd.Series(dtype=object),
    ).map(_target_id)
    industries_frame["is_target_industry"] = industries_frame["target_id"].notnull()

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
        matches = industries_frame.loc[
            industries_frame["target_id"].eq(target_id)
        ].copy()
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
    warmup_count = max(max(MA_WINDOWS), max(HIGH_LOW_WINDOWS))
    warmup = list(get_trade_days(end_date=trade_days[0], count=warmup_count))
    if not warmup:
        raise RuntimeError("JQ returned no warm-up trade days")
    price_days = list(get_trade_days(start_date=warmup[0], end_date=trade_days[-1]))

    prices = _query_price_matrices(
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
    valuations = _query_valuation_matrices(
        securities,
        trade_days[0],
        trade_days[-1],
        trade_days,
    )
    price_features = _price_feature_matrices(
        prices["close"],
        prices["high"],
        prices["low"],
    )

    daily_records = []
    industry_frames = []
    for position, date in enumerate(output_index, start=1):
        universe = universes[date]
        summary, industries_frame = aggregate_day(
            date=date,
            universe=universe,
            prices={field: prices[field].loc[date] for field in PRICE_FIELDS},
            is_st=st.loc[date],
            price_features={
                group: {
                    window: matrix.loc[date]
                    for window, matrix in values.items()
                }
                for group, values in price_features.items()
            },
            valuations={
                field: valuations[field].loc[date]
                for field in VALUATION_FIELDS
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
    """Return validated ZIP bytes for the processed P1 input tables."""

    daily_content = _csv_bytes(daily)
    industry_content = _csv_bytes(industry)
    files = [
        _file_record("data/daily_market_features.csv", daily_content, daily),
        _file_record("data/industry_breadth.csv", industry_content, industry),
    ]
    manifest = {
        "data_version": DATA_VERSION,
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "availability": (
            "Each row uses date-t close/valuation data and is available only "
            "after date-t close."
        ),
        "query": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "frequency": "daily",
            "universe": {
                "method": "point_in_time_index_constituents",
                "index": UNIVERSE_INDEX,
            },
            "price_fields": list(PRICE_FIELDS),
            "price_fq": PRICE_FQ,
            "fill_paused": True,
            "st_source": "get_extras:is_st",
            "industry_source": "get_industry:%s" % INDUSTRY_LEVEL,
            "valuation_source": "get_valuation",
            "valuation_fields": list(VALUATION_FIELDS),
            "ma_windows": list(MA_WINDOWS),
            "high_low_windows": list(HIGH_LOW_WINDOWS),
            "turnover_extreme_thresholds_pct": list(
                TURNOVER_EXTREME_THRESHOLDS_PCT
            ),
            "ma_comparison_relative_tolerance": MA_COMPARISON_RELATIVE_TOLERANCE,
            "ma_comparison_absolute_tolerance": MA_COMPARISON_ABSOLUTE_TOLERANCE,
            "industry_rank_window": 20,
            "industry_rank_method": "min_descending",
            "min_industry_valid_count": MIN_INDUSTRY_VALID_COUNT,
            "targets": TARGET_INDUSTRIES,
            "old_mining_mapped_to_coal": False,
        },
        "feature_definitions": {
            "ma_breadth": (
                "pre-adjusted close minus the inclusive MA is greater than "
                "absolute_tolerance + relative_tolerance * abs(MA); numerical "
                "equality is not above"
            ),
            "new_high_low_breadth": (
                "pre-adjusted intraday high at the inclusive rolling high, "
                "or intraday low at the inclusive rolling low"
            ),
            "limit_hit": (
                "high reaches high_limit or low reaches low_limit; all compared "
                "on the same pre-adjusted price scale"
            ),
            "limit_close": (
                "close reaches high_limit or low_limit on the same adjusted scale"
            ),
            "turnover": (
                "JQ turnover_ratio in percent, with cross-sectional quantiles, "
                "fixed diagnostic tails and circulating-market-cap weighted mean"
            ),
        },
        "denominators": {
            "base": (
                "Point-in-time index constituent with finite close, known "
                "non-paused status and known non-ST status."
            ),
            "ma": "Base-valid stock with a complete adjusted-close MA window.",
            "high_low": (
                "Base-valid stock with finite high/low and complete rolling windows."
            ),
            "limit": (
                "Base-valid stock with finite positive high_limit/low_limit and "
                "finite close/high/low."
            ),
            "turnover": (
                "Base-valid stock with a finite non-negative JQ turnover_ratio."
            ),
            "industry": (
                "MA-valid stock with a point-in-time SW level-1 assignment; "
                "at least five MA20-valid stocks enter Top1 ranking."
            ),
        },
        "export_level": {
            "daily_market_features": "one row per trade date",
            "industry_breadth": (
                "one row per trade date and observed SW level-1 industry"
            ),
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
        archive.writestr("data/daily_market_features.csv", daily_content)
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
        raise ValueError("daily_market_features dates do not match requested trade days")
    if industry.duplicated(["date", "industry_code", "industry_name"]).any():
        raise ValueError("duplicate date-industry rows")

    ratio_columns = []
    for window in MA_WINDOWS:
        ratio_columns.append("breadth_ma%d" % window)
    for window in HIGH_LOW_WINDOWS:
        ratio_columns.extend(
            [
                "new_high_ratio_%d" % window,
                "new_low_ratio_%d" % window,
            ]
        )
    ratio_columns.extend(
        [
            "limit_up_hit_ratio",
            "limit_down_hit_ratio",
            "limit_up_close_ratio",
            "limit_down_close_ratio",
        ]
    )
    for threshold in TURNOVER_EXTREME_THRESHOLDS_PCT:
        label = ("%g" % threshold).replace(".", "p")
        ratio_columns.append("turnover_ge_%spct_ratio" % label)
    for column in ratio_columns:
        values = pd.to_numeric(daily[column], errors="coerce").dropna()
        if not values.between(0.0, 1.0).all():
            raise ValueError("ratio outside [0, 1]: %s" % column)

    net_ratio_columns = ["limit_hit_net_ratio", "limit_close_net_ratio"]
    net_ratio_columns.extend(
        ["new_high_low_net_ratio_%d" % window for window in HIGH_LOW_WINDOWS]
    )
    for column in net_ratio_columns:
        values = pd.to_numeric(daily[column], errors="coerce").dropna()
        if not values.between(-1.0, 1.0).all():
            raise ValueError("net ratio outside [-1, 1]: %s" % column)

    for window in MA_WINDOWS:
        values = pd.to_numeric(
            industry["breadth_ma%d" % window],
            errors="coerce",
        ).dropna()
        if not values.between(0.0, 1.0).all():
            raise ValueError("industry breadth outside [0, 1]")


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
