"""Causal daily event and episode semantics shared by every signal."""

from __future__ import annotations

import numpy as np
import pandas as pd


SERIES_COLUMNS = ("signal_id", "direction", "version")
REQUIRED_DAILY_COLUMNS = (
    "date",
    "signal_id",
    "direction",
    "raw_value",
    "triggered",
    "universe_size",
    "valid_count",
    "version",
)
DAILY_EVENT_COLUMNS = (
    "episode_id",
    "episode_number",
    "episode_day",
    "episode_stage",
    "event_onset",
    "event_continuation",
    "event_exit",
    "event_capped_confirmation",
    "capped_confirmation_reason",
    "capped_confirmation_n",
)
EPISODE_COLUMNS = (
    "episode_id",
    "signal_id",
    "direction",
    "version",
    "episode_number",
    "onset_date",
    "last_active_date",
    "exit_date",
    "active_days",
    "status",
    "capped_confirmation_n",
    "capped_confirmation_date",
    "capped_confirmation_reason",
    "confirmation_status",
    "series_start_date",
    "series_end_date",
)


def build_signal_events(
    daily: pd.DataFrame,
    *,
    capped_confirmation_n: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive causal phases and one episode summary per contiguous active run.

    Daily event columns are point-in-time invariant. A short episode confirms on
    the first observed exit day; an unexited sample-tail episode remains pending
    in the episode summary and never receives a backfilled daily event.
    """

    _validate_capped_confirmation_n(capped_confirmation_n)
    source = _validate_daily(daily)
    daily_frames: list[pd.DataFrame] = []
    episode_records: list[dict[str, object]] = []

    for key, group in source.groupby(list(SERIES_COLUMNS), sort=True):
        signal_id, direction, version = (str(value) for value in key)
        annotated, episodes = _build_one_series(
            group.reset_index(drop=True),
            signal_id=signal_id,
            direction=direction,
            version=version,
            capped_confirmation_n=capped_confirmation_n,
        )
        daily_frames.append(annotated)
        episode_records.extend(episodes)

    event_daily = pd.concat(daily_frames, ignore_index=True)
    event_daily["episode_number"] = event_daily["episode_number"].astype("Int64")
    event_daily["episode_day"] = event_daily["episode_day"].astype("Int64")
    episodes = pd.DataFrame(episode_records, columns=EPISODE_COLUMNS)
    if not episodes.empty:
        episodes["episode_number"] = episodes["episode_number"].astype("Int64")
        episodes["active_days"] = episodes["active_days"].astype("Int64")
        for column in (
            "onset_date",
            "last_active_date",
            "exit_date",
            "capped_confirmation_date",
            "series_start_date",
            "series_end_date",
        ):
            episodes[column] = pd.to_datetime(episodes[column])
    return event_daily, episodes


def _build_one_series(
    group: pd.DataFrame,
    *,
    signal_id: str,
    direction: str,
    version: str,
    capped_confirmation_n: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    annotated = group.copy()
    row_count = len(annotated)
    episode_ids: list[object] = [pd.NA] * row_count
    episode_numbers: list[object] = [pd.NA] * row_count
    episode_days: list[object] = [pd.NA] * row_count
    stages = ["inactive"] * row_count
    onsets = np.zeros(row_count, dtype=bool)
    continuations = np.zeros(row_count, dtype=bool)
    exits = np.zeros(row_count, dtype=bool)
    confirmations = np.zeros(row_count, dtype=bool)
    confirmation_reasons = [""] * row_count

    series_start = pd.Timestamp(annotated["date"].iloc[0])
    series_end = pd.Timestamp(annotated["date"].iloc[-1])
    episode_number = 0
    current: dict[str, object] | None = None
    records: list[dict[str, object]] = []

    for position, row in annotated.iterrows():
        date = pd.Timestamp(row["date"])
        active = bool(row["triggered"])
        if active:
            if current is None:
                episode_number += 1
                episode_id = _episode_id(
                    signal_id,
                    direction,
                    version,
                    episode_number,
                )
                current = {
                    "episode_id": episode_id,
                    "episode_number": episode_number,
                    "onset_date": date,
                    "last_active_date": date,
                    "active_days": 1,
                    "confirmation_date": pd.NaT,
                    "confirmation_reason": "",
                }
                onsets[position] = True
                stages[position] = "onset"
            else:
                current["active_days"] = int(current["active_days"]) + 1
                current["last_active_date"] = date
                continuations[position] = True
                stages[position] = "continuation"

            episode_ids[position] = current["episode_id"]
            episode_numbers[position] = current["episode_number"]
            episode_days[position] = current["active_days"]
            if int(current["active_days"]) == capped_confirmation_n:
                confirmations[position] = True
                confirmation_reasons[position] = "nth_active_day"
                current["confirmation_date"] = date
                current["confirmation_reason"] = "nth_active_day"
            continue

        if current is None:
            continue

        exits[position] = True
        stages[position] = "exit"
        episode_ids[position] = current["episode_id"]
        episode_numbers[position] = current["episode_number"]
        if int(current["active_days"]) < capped_confirmation_n:
            confirmations[position] = True
            confirmation_reasons[position] = "short_episode_exit"
            current["confirmation_date"] = date
            current["confirmation_reason"] = "short_episode_exit"
        records.append(
            _episode_record(
                current,
                signal_id=signal_id,
                direction=direction,
                version=version,
                exit_date=date,
                status="closed",
                capped_confirmation_n=capped_confirmation_n,
                series_start=series_start,
                series_end=series_end,
            )
        )
        current = None

    if current is not None:
        records.append(
            _episode_record(
                current,
                signal_id=signal_id,
                direction=direction,
                version=version,
                exit_date=pd.NaT,
                status="active",
                capped_confirmation_n=capped_confirmation_n,
                series_start=series_start,
                series_end=series_end,
            )
        )

    annotated["episode_id"] = episode_ids
    annotated["episode_number"] = episode_numbers
    annotated["episode_day"] = episode_days
    annotated["episode_stage"] = stages
    annotated["event_onset"] = onsets
    annotated["event_continuation"] = continuations
    annotated["event_exit"] = exits
    annotated["event_capped_confirmation"] = confirmations
    annotated["capped_confirmation_reason"] = confirmation_reasons
    annotated["capped_confirmation_n"] = capped_confirmation_n
    return annotated, records


def _episode_record(
    current: dict[str, object],
    *,
    signal_id: str,
    direction: str,
    version: str,
    exit_date: object,
    status: str,
    capped_confirmation_n: int,
    series_start: pd.Timestamp,
    series_end: pd.Timestamp,
) -> dict[str, object]:
    confirmation_date = current["confirmation_date"]
    confirmed = not pd.isna(confirmation_date)
    return {
        "episode_id": current["episode_id"],
        "signal_id": signal_id,
        "direction": direction,
        "version": version,
        "episode_number": current["episode_number"],
        "onset_date": current["onset_date"],
        "last_active_date": current["last_active_date"],
        "exit_date": exit_date,
        "active_days": current["active_days"],
        "status": status,
        "capped_confirmation_n": capped_confirmation_n,
        "capped_confirmation_date": confirmation_date,
        "capped_confirmation_reason": current["confirmation_reason"],
        "confirmation_status": "confirmed" if confirmed else "pending",
        "series_start_date": series_start,
        "series_end_date": series_end,
    }


def _episode_id(
    signal_id: str,
    direction: str,
    version: str,
    episode_number: int,
) -> str:
    return f"{signal_id}::{direction}::{version}::{episode_number:06d}"


def _validate_capped_confirmation_n(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("capped_confirmation_n must be a positive integer")


def _validate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    missing = set(REQUIRED_DAILY_COLUMNS).difference(daily.columns)
    if missing:
        raise ValueError(f"daily is missing columns: {sorted(missing)}")
    if daily.empty:
        raise ValueError("daily must not be empty")

    result = daily.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("daily contains an invalid date")
    for column in SERIES_COLUMNS:
        values = result[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} must contain non-empty values")
        result[column] = values.astype(str)
    if not result["direction"].isin(["top", "bottom"]).all():
        raise ValueError("direction must be top or bottom")
    result["triggered"] = _strict_bool(result["triggered"], "triggered")

    for column in ("universe_size", "valid_count"):
        values = pd.to_numeric(result[column], errors="coerce")
        if (
            values.isna().any()
            or values.lt(0).any()
            or np.not_equal(values, np.floor(values)).any()
        ):
            raise ValueError(f"{column} must contain non-negative integers")
        result[column] = values.astype(int)
    if result["valid_count"].gt(result["universe_size"]).any():
        raise ValueError("valid_count must not exceed universe_size")

    if result.duplicated([*SERIES_COLUMNS, "date"]).any():
        raise ValueError("daily contains duplicate series dates")
    return result.sort_values([*SERIES_COLUMNS, "date"]).reset_index(drop=True)


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        invalid = sorted(normalized[~normalized.isin(mapping)].unique())
        raise ValueError(f"{name} contains invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)
