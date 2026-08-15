"""Small batch pipeline for index turning-point research data."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import pandas as pd

from .labels import directional_change_labels


THRESHOLDS = (0.05, 0.10, 0.20)
HORIZONS = (5, 10, 20, 60)

INDEX_SPECS = (
    ("sse_composite", "上证指数", "000001.XSHG", "sh/lday/sh999999.day", False),
    ("csi300", "沪深300", "000300.XSHG", "sh/lday/sh000300.day", False),
    ("csi500", "中证500", "000905.XSHG", "sh/lday/sh000905.day", False),
    ("csi1000", "中证1000", "000852.XSHG", "sh/lday/sh000852.day", False),
    ("cni2000", "国证2000", "399303.XSHE", "sz/lday/sz399303.day", False),
    ("microcap", "微盘股", "TDX.880823", "sh/lday/sh880823.day", False),
    ("all_a", "全A", "000985.XSHG", "ds/lday/62#000985.day", True),
)

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


def forward_outcomes(
    close: pd.Series,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Calculate full-window future downside, upside and terminal return."""

    result = pd.DataFrame({"close": close.astype(float)})
    for horizon in horizons:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizons must contain positive integers")

        future = pd.concat(
            [close.shift(-step) for step in range(1, horizon + 1)],
            axis=1,
        )
        complete = future.notna().all(axis=1)
        returns = future.divide(close, axis=0).sub(1.0)
        result[f"future_max_down_{horizon}d"] = returns.min(axis=1).where(complete)
        result[f"future_max_up_{horizon}d"] = returns.max(axis=1).where(complete)
        result[f"future_return_{horizon}d"] = (
            close.shift(-horizon).divide(close).sub(1.0).where(complete)
        )

    return result


def run_pipeline(vipdoc: Path | str, output_dir: Path | str) -> dict[str, Path]:
    """Read seven indices and write manifest, labels and forward outcomes."""

    vipdoc = Path(vipdoc)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests = []
    all_labels = []
    all_outcomes = []

    for index_id, index_name, jq_code, relative_path, float_prices in INDEX_SPECS:
        path = vipdoc / relative_path
        raw_rows = path.stat().st_size // STANDARD_RECORD.size
        daily = read_tdx_daily(path, float_prices=float_prices)
        close = daily["close"]

        manifests.append(
            {
                "index_id": index_id,
                "index_name": index_name,
                "jq_code": jq_code,
                "source_file": relative_path,
                "raw_rows": raw_rows,
                "rows": len(daily),
                "duplicates_removed": raw_rows - len(daily),
                "start_date": daily.index.min(),
                "end_date": daily.index.max(),
                "last_close": close.iloc[-1],
            }
        )

        for threshold in THRESHOLDS:
            labels = directional_change_labels(close, threshold)
            labels.insert(0, "index_name", index_name)
            labels.insert(0, "index_id", index_id)
            all_labels.append(labels)

        outcomes = forward_outcomes(close).reset_index()
        outcomes.insert(0, "index_name", index_name)
        outcomes.insert(0, "index_id", index_id)
        all_outcomes.append(outcomes)

    outputs = {
        "manifest": output_dir / "data_manifest.csv",
        "labels": output_dir / "turning_point_labels.csv",
        "outcomes": output_dir / "forward_outcomes.csv",
    }
    pd.DataFrame(manifests).to_csv(outputs["manifest"], index=False)
    pd.concat(all_labels, ignore_index=True).to_csv(outputs["labels"], index=False)
    pd.concat(all_outcomes, ignore_index=True).to_csv(outputs["outcomes"], index=False)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vipdoc",
        type=Path,
        default=Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc",
        help="通达信 vipdoc 目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
        help="CSV 输出目录",
    )
    args = parser.parse_args()

    outputs = run_pipeline(args.vipdoc, args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
