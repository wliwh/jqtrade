"""Read and validate the local TDX index OHLC sources."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_SPECS = (
    ("sse_composite", "上证指数", "000001.XSHG", "sh/lday/sh999999.day", False),
    ("csi300", "沪深300", "000300.XSHG", "sh/lday/sh000300.day", False),
    ("csi500", "中证500", "000905.XSHG", "sh/lday/sh000905.day", False),
    ("csi1000", "中证1000", "000852.XSHG", "sh/lday/sh000852.day", False),
    ("cni2000", "国证2000", "399303.XSHE", "sz/lday/sz399303.day", False),
    ("microcap", "微盘股", "TDX.880823", "sh/lday/sh880823.day", False),
    ("all_a", "全A", "000985.XSHG", "ds/lday/62#000985.day", True),
)

# Fixed before signal evaluation. These are intentionally simple volatility
# buckets rather than rolling or outcome-optimized parameters.
INDEX_THRESHOLD_MULTIPLIERS = {
    "sse_composite": 0.8,
    "csi300": 0.9,
    "csi500": 1.1,
    "csi1000": 1.2,
    "cni2000": 1.2,
    "microcap": 1.3,
    "all_a": 1.0,
}

STANDARD_RECORD = struct.Struct("<IIIIIfII")
FLOAT_RECORD = struct.Struct("<IfffffII")
DAILY_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "amount",
    "volume",
    "reserved",
]


def threshold_for_index(index_id: str, base_threshold: float) -> float:
    """Return one fixed index-specific threshold from a common base scale."""

    try:
        multiplier = INDEX_THRESHOLD_MULTIPLIERS[index_id]
    except KeyError as exc:
        raise KeyError(f"missing threshold multiplier for index: {index_id}") from exc

    threshold = round(float(base_threshold) * multiplier, 10)
    if not 0.0 < threshold < 1.0:
        raise ValueError("adjusted threshold must be between 0 and 1")
    return threshold


def read_tdx_daily(path: Path | str, *, float_prices: bool = False) -> pd.DataFrame:
    """Read and minimally validate one 32-byte TDX daily file."""

    path = Path(path)
    data = path.read_bytes()
    record = FLOAT_RECORD if float_prices else STANDARD_RECORD
    if not data or len(data) % record.size:
        raise ValueError(f"invalid TDX daily file size: {path}")

    frame = pd.DataFrame(record.iter_unpack(data), columns=DAILY_COLUMNS)
    if not float_prices:
        price_columns = ["open", "high", "low", "close"]
        frame[price_columns] = frame[price_columns].astype(float).div(100.0)

    try:
        dates = pd.to_datetime(frame.pop("date").astype(str), format="%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"invalid date in TDX daily file: {path}") from exc
    frame.index = dates
    frame.index.name = "date"

    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"dates are not increasing: {path}")

    duplicated = frame.index.duplicated(keep=False)
    if duplicated.any():
        for date, group in frame.loc[duplicated].groupby(level=0, sort=False):
            if len(group.drop_duplicates()) != 1:
                raise ValueError(f"conflicting rows for {date.date()}: {path}")
        frame = frame.loc[~frame.index.duplicated(keep="last")]

    prices = frame[["open", "high", "low", "close"]]
    valid_ohlc = (
        np.isfinite(prices.to_numpy()).all()
        and (prices > 0).all().all()
        and (frame["high"] >= frame[["open", "close"]].max(axis=1)).all()
        and (frame["low"] <= frame[["open", "close"]].min(axis=1)).all()
        and (frame["high"] >= frame["low"]).all()
    )
    if not valid_ohlc:
        raise ValueError(f"invalid OHLC data: {path}")

    return frame
