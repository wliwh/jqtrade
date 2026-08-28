"""Build the offline viewer for source search terms with local comparison indices."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs

from ..adapters.tdx import read_tdx_daily


@dataclass(frozen=True)
class KeywordSpec:
    keyword: str
    index_id: str


@dataclass(frozen=True)
class ViewerIndexSpec:
    index_id: str
    index_name: str
    tdx_path: str
    float_prices: bool = False


# Every locally validated index price source offered by the viewer.
VIEWER_INDEX_SPECS = (
    ViewerIndexSpec("sse_composite", "上证指数", "sh/lday/sh999999.day"),
    ViewerIndexSpec("sse50", "上证50", "sh/lday/sh000016.day"),
    ViewerIndexSpec("csi300", "沪深300", "sh/lday/sh000300.day"),
    ViewerIndexSpec("csi500", "中证500", "sh/lday/sh000905.day"),
    ViewerIndexSpec("csi1000", "中证1000", "sh/lday/sh000852.day"),
    ViewerIndexSpec("cni2000", "国证2000", "sz/lday/sz399303.day"),
    ViewerIndexSpec("chinext", "创业板指", "sz/lday/sz399006.day"),
    ViewerIndexSpec("star50", "科创50", "sh/lday/sh000688.day"),
    ViewerIndexSpec("szse_component", "深证成指", "sz/lday/sz399001.day"),
    ViewerIndexSpec("microcap", "微盘股", "sh/lday/sh880823.day"),
    ViewerIndexSpec("all_a", "全A", "ds/lday/62#000985.day", True),
)


# Keep source keywords that have a meaningful local comparison index, in source order.
# Exact-name matches use their local counterpart; broad A-share terms use SSE.
KEYWORD_SPECS = (
    KeywordSpec("上证指数", "sse_composite"),
    KeywordSpec("股票", "sse_composite"),
    KeywordSpec("a股", "sse_composite"),
    KeywordSpec("基金", "sse_composite"),
    KeywordSpec("股市", "sse_composite"),
    KeywordSpec("上证", "sse_composite"),
    KeywordSpec("熊市", "sse_composite"),
    KeywordSpec("沪深300", "csi300"),
    KeywordSpec("上证50", "sse50"),
    KeywordSpec("牛市", "sse_composite"),
    KeywordSpec("中证500", "csi500"),
    KeywordSpec("创业板指", "chinext"),
    KeywordSpec("科创50", "star50"),
)

EXPECTED_COLUMNS = ("date", "keyword", "type", "count")
ROLLING_WINDOW = 252
ROLLING_MIN_PERIODS = 60
VIEWER_VERSION = "bsearch_index_exploration_v1_6_20110104_20260814"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bsearch_csv(path: Path | str) -> pd.DataFrame:
    """Read and validate the raw daily keyword heat CSV without modifying it."""

    path = Path(path)
    frame = pd.read_csv(path)
    if len(frame.columns) != 4:
        raise ValueError("bsearch CSV must contain exactly four columns")

    frame = frame.copy()
    frame.columns = list(EXPECTED_COLUMNS)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["keyword"] = frame["keyword"].astype(str)
    frame["type"] = frame["type"].astype(str)
    frame["count"] = pd.to_numeric(frame["count"], errors="raise")

    if frame.empty:
        raise ValueError("bsearch CSV is empty")
    if frame[list(EXPECTED_COLUMNS)].isna().any().any():
        raise ValueError("bsearch CSV contains missing values")
    if (frame["count"] < 0).any():
        raise ValueError("bsearch count must be non-negative")
    if set(frame["type"].unique()) != {"all"}:
        raise ValueError("bsearch viewer requires type=all")
    if frame.duplicated(["date", "keyword", "type"]).any():
        raise ValueError("bsearch CSV contains duplicate date-keyword rows")
    if not frame["date"].between("2000-01-01", "2100-01-01").all():
        raise ValueError("bsearch CSV contains implausible dates")

    available = set(frame["keyword"].unique())
    expected = {spec.keyword for spec in KEYWORD_SPECS}
    missing = sorted(expected - available)
    if missing:
        raise ValueError(f"bsearch CSV is missing eligible source keywords: {missing}")
    return frame.sort_values(["keyword", "date"]).reset_index(drop=True)


def point_in_time_heat_z(
    count: pd.Series,
    *,
    window: int = ROLLING_WINDOW,
    min_periods: int = ROLLING_MIN_PERIODS,
) -> pd.Series:
    """Standardize log heat against prior trading days only."""

    log_heat = np.log1p(count.astype(float))
    history = log_heat.shift(1).rolling(window, min_periods=min_periods)
    scale = history.std(ddof=0).replace(0.0, np.nan)
    return (log_heat - history.mean()) / scale


def align_keyword_to_prices(
    bsearch: pd.DataFrame,
    prices: pd.DataFrame,
    spec: KeywordSpec,
) -> pd.DataFrame:
    """Inner-join one keyword to its index calendar, dropping non-trading days."""

    price_frame = prices.reset_index()
    if "date" not in price_frame.columns or "close" not in price_frame.columns:
        raise ValueError("price frame must provide date and close")

    heat = bsearch.loc[bsearch["keyword"] == spec.keyword, ["date", "count"]]
    aligned = heat.merge(price_frame[["date", "close"]], on="date", how="inner", validate="one_to_one")
    aligned = aligned.sort_values("date").reset_index(drop=True)
    if aligned.empty:
        raise ValueError(f"no trading-day overlap for keyword: {spec.keyword}")
    if aligned["date"].duplicated().any():
        raise ValueError(f"duplicate aligned trading date for keyword: {spec.keyword}")

    aligned["heat_z252"] = point_in_time_heat_z(aligned["count"])
    return aligned


def _json_numbers(series: pd.Series, *, digits: int | None = None) -> list[float | int | None]:
    values: list[float | int | None] = []
    for value in series:
        if pd.isna(value):
            values.append(None)
        elif digits is None:
            values.append(int(value) if float(value).is_integer() else float(value))
        else:
            values.append(round(float(value), digits))
    return values


def _missing_weekdays(dates: pd.Series) -> list[str]:
    observed = pd.DatetimeIndex(dates)
    weekdays = pd.date_range(observed.min(), observed.max(), freq="B")
    missing = weekdays.difference(observed)
    return missing.strftime("%Y-%m-%d").tolist()


def load_regions(path: Path | str) -> dict[str, list[dict[str, str]]]:
    """Load frozen post-hoc region envelopes for visual evaluation only."""

    frame = pd.read_csv(path)
    required = {"index_id", "event_type", "eligible", "region_start", "region_end"}
    if not required.issubset(frame.columns):
        raise ValueError("top/bottom region CSV is missing required columns")

    eligible = frame["eligible"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().map({"true": True, "false": False})
    frame = frame.loc[eligible.fillna(False)].copy()
    frame["region_start"] = pd.to_datetime(frame["region_start"], errors="raise")
    frame["region_end"] = pd.to_datetime(frame["region_end"], errors="raise")
    frame = frame.sort_values(["index_id", "region_start", "event_type"])
    return {
        str(index_id): [
            {
                "event_type": str(row.event_type),
                "start": row.region_start.strftime("%Y-%m-%d"),
                "end": row.region_end.strftime("%Y-%m-%d"),
            }
            for row in group.itertuples()
        ]
        for index_id, group in frame.groupby("index_id", sort=False)
    }


def build_viewer_payload(
    bsearch: pd.DataFrame,
    prices_by_index: dict[str, pd.DataFrame],
    regions_by_index: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Create the compact JSON payload embedded in the standalone HTML."""

    index_specs = {spec.index_id: spec for spec in VIEWER_INDEX_SPECS}
    search_start = bsearch["date"].min()
    search_end = bsearch["date"].max()
    indices: dict[str, Any] = {}
    for index_id, spec in index_specs.items():
        try:
            prices = prices_by_index[index_id]
        except KeyError as exc:
            raise ValueError(f"missing price frame for index: {index_id}") from exc
        price_frame = prices.reset_index()
        ohlc_columns = ["open", "high", "low", "close"]
        if "date" not in price_frame.columns or not set(ohlc_columns).issubset(price_frame.columns):
            raise ValueError("price frame must provide date and OHLC")
        price_frame = price_frame.loc[
            price_frame["date"].between(search_start, search_end), ["date", *ohlc_columns]
        ].sort_values("date").reset_index(drop=True)
        if price_frame.empty:
            raise ValueError(f"no search-date overlap for index: {index_id}")
        indices[index_id] = {
            "index_id": index_id,
            "index_name": spec.index_name,
            "date": price_frame["date"].dt.strftime("%Y-%m-%d").tolist(),
            **{
                column: _json_numbers(price_frame[column], digits=4)
                for column in ohlc_columns
            },
            "start_date": price_frame["date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": price_frame["date"].iloc[-1].strftime("%Y-%m-%d"),
            "missing_weekdays": _missing_weekdays(price_frame["date"]),
        }

    series: dict[str, Any] = {}
    try:
        trading_calendar = prices_by_index["sse_composite"]
    except KeyError as exc:
        raise ValueError("missing price frame for index: sse_composite") from exc
    for spec in KEYWORD_SPECS:
        aligned = align_keyword_to_prices(bsearch, trading_calendar, spec)
        recommended_index = index_specs[spec.index_id]
        source_rows = int((bsearch["keyword"] == spec.keyword).sum())
        series[spec.keyword] = {
            "keyword": spec.keyword,
            "recommended_index_id": spec.index_id,
            "recommended_index_name": recommended_index.index_name,
            "date": aligned["date"].dt.strftime("%Y-%m-%d").tolist(),
            "count": _json_numbers(aligned["count"]),
            "heat_z252": _json_numbers(aligned["heat_z252"], digits=4),
            "skipped_rows": source_rows - len(aligned),
            "start_date": aligned["date"].iloc[0].strftime("%Y-%m-%d"),
            "end_date": aligned["date"].iloc[-1].strftime("%Y-%m-%d"),
            "missing_weekdays": _missing_weekdays(aligned["date"]),
        }

    return {
        "index_ids": list(indices),
        "indices": indices,
        "keywords": [spec.keyword for spec in KEYWORD_SPECS],
        "series": series,
        "regions_by_index": regions_by_index,
    }


def write_offline_viewer(
    *,
    bsearch_path: Path | str,
    vipdoc: Path | str,
    region_path: Path | str,
    output_path: Path | str,
) -> dict[str, Any]:
    """Write one self-contained HTML viewer and return its provenance metadata."""

    bsearch_path = Path(bsearch_path)
    vipdoc = Path(vipdoc)
    region_path = Path(region_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite viewer: {output_path}")

    bsearch = read_bsearch_csv(bsearch_path)
    unique_price_specs = {spec.index_id: spec for spec in VIEWER_INDEX_SPECS}
    prices_by_index = {
        index_id: read_tdx_daily(vipdoc / spec.tdx_path, float_prices=spec.float_prices)
        for index_id, spec in unique_price_specs.items()
    }
    regions_by_index = load_regions(region_path)
    payload = build_viewer_payload(bsearch, prices_by_index, regions_by_index)

    template_path = Path(__file__).with_name("templates") / "bsearch_index_viewer.html"
    template = template_path.read_text(encoding="utf-8")
    metadata = {
        "viewer_version": VIEWER_VERSION,
        "purpose": "Exploratory offline visualization; not a frozen signal or trading rule.",
        "bsearch_path": str(bsearch_path),
        "bsearch_sha256": sha256_file(bsearch_path),
        "region_path": str(region_path),
        "region_sha256": sha256_file(region_path),
        "price_sources": [
            {
                "index_id": index_id,
                "path": spec.tdx_path,
                "sha256": sha256_file(vipdoc / spec.tdx_path),
            }
            for index_id, spec in unique_price_specs.items()
        ],
        "keyword_specs": [spec.__dict__ for spec in KEYWORD_SPECS],
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace("/*__PLOTLY_JS__*/", get_plotlyjs())
    html = html.replace("/*__BSEARCH_PAYLOAD__*/", payload_json)
    html = html.replace("/*__BSEARCH_METADATA__*/", metadata_json)
    if "/*__" in html:
        raise ValueError("viewer template contains unresolved placeholders")

    output_path.parent.mkdir(parents=True, exist_ok=False)
    output_path.write_text(html, encoding="utf-8")
    metadata["output_path"] = str(output_path)
    metadata["output_bytes"] = output_path.stat().st_size
    metadata["output_sha256"] = sha256_file(output_path)
    return metadata
