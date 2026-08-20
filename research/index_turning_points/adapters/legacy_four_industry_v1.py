"""Normalize the archived four-industry Top1 V1 daily signal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SIGNAL_PATH = (
    PROJECT_DIR
    / "archive"
    / "four_industry_width_v1"
    / "results"
    / "four_industry_top1"
    / "signal_daily_phases.csv"
)
TARGET_COLUMNS = (
    "target_bank",
    "target_coal",
    "target_nonferrous",
    "target_steel",
)
REQUIRED_COLUMNS = (
    "date",
    "breadth_ma20",
    "triggered",
    "onset",
    "continuation",
    "episode_id",
    "episode_day",
    "four_industry_top1_ids",
    "top1_tie_count_ma20",
) + TARGET_COLUMNS


def _as_bool(series: pd.Series, column: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype("string").str.strip().str.lower()
    invalid = normalized.notna() & ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        values = sorted(normalized.loc[invalid].dropna().unique().tolist())
        raise ValueError("%s contains invalid booleans: %s" % (column, values))
    return normalized.isin({"true", "1"})


def load_four_industry_v1_signal(path: Path | str) -> pd.DataFrame:
    """Load and validate the archived point-in-time daily signal."""

    path = Path(path)
    signal = pd.read_csv(path)
    missing = sorted(set(REQUIRED_COLUMNS).difference(signal.columns))
    if missing:
        raise ValueError("four-industry signal is missing columns: %s" % missing)

    signal = signal.loc[:, REQUIRED_COLUMNS].copy()
    signal["date"] = pd.to_datetime(signal["date"], errors="raise")
    if signal["date"].duplicated().any():
        duplicates = signal.loc[signal["date"].duplicated(), "date"]
        raise ValueError(
            "four-industry signal has duplicate dates: %s"
            % duplicates.dt.strftime("%Y-%m-%d").tolist()
        )
    signal = signal.sort_values("date").reset_index(drop=True)

    for column in ("triggered", "onset", "continuation") + TARGET_COLUMNS:
        signal[column] = _as_bool(signal[column], column)

    target_active = signal.loc[:, TARGET_COLUMNS].any(axis=1)
    if not signal["triggered"].equals(target_active):
        raise ValueError("triggered does not match the four industry target columns")

    expected_onset = signal["triggered"] & ~signal["triggered"].shift(
        fill_value=False
    )
    expected_continuation = signal["triggered"] & ~expected_onset
    if not signal["onset"].equals(expected_onset):
        raise ValueError("onset is inconsistent with triggered")
    if not signal["continuation"].equals(expected_continuation):
        raise ValueError("continuation is inconsistent with triggered")
    if signal.loc[signal["triggered"], "episode_id"].isna().any():
        raise ValueError("active signal dates must have episode_id")
    if signal.loc[signal["triggered"], "episode_day"].isna().any():
        raise ValueError("active signal dates must have episode_day")

    return signal
