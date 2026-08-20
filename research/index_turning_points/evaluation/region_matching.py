"""Deterministic one-to-one matching of causal signal events to regions."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from ..ground_truth.regions import DEFAULT_REGION_PROTOCOL, RegionProtocol


EVENT_FLAGS = {
    "onset": "event_onset",
    "capped_confirmation": "event_capped_confirmation",
    "exit": "event_exit",
}
SERIES_COLUMNS = ("signal_id", "direction", "version")
MATCH_COLUMNS = (
    "record_id",
    "signal_id",
    "direction",
    "version",
    "event_kind",
    "index_id",
    "index_name",
    "match_status",
    "primary_match",
    "episode_id",
    "event_date",
    "region_id",
    "anchor_date",
    "region_start",
    "region_end",
    "region_form",
    "lobe_count",
    "matched_lobe_id",
    "match_level",
    "strict_matched",
    "loose_matched",
    "timing",
    "lead_lag_days",
    "absolute_distance_days",
    "window_days",
    "primary_episode_id_for_region",
    "coverage_start_date",
    "coverage_end_date",
    "prediction_window_complete",
    "confirmation_window_complete",
    "label_version",
)
METRIC_COLUMNS = (
    "signal_id",
    "direction",
    "version",
    "event_kind",
    "index_id",
    "index_name",
    "match_scope",
    "region_count",
    "matched_region_count",
    "missed_region_count",
    "region_recall",
    "episode_count",
    "matched_episode_count",
    "false_alarm_count",
    "episode_precision",
    "duplicate_alarm_count",
    "median_lead_lag_days",
    "q25_lead_lag_days",
    "q75_lead_lag_days",
)


def match_signal_regions(
    signal_daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    calendars: pd.DataFrame,
    *,
    event_kind: str = "onset",
    protocol: RegionProtocol = DEFAULT_REGION_PROTOCOL,
) -> pd.DataFrame:
    """Return auditable primary matches, duplicates, false alarms and misses.

    Matching is repeated independently for every signal series and index. Edges
    are greedily assigned by core lobe, envelope, frozen window, absolute
    anchor distance and stable IDs. This implements the frozen stage-C priority
    without allowing labels to alter signal dates.
    """

    if event_kind not in EVENT_FLAGS:
        raise ValueError(f"unsupported event_kind: {event_kind}")
    if not isinstance(protocol, RegionProtocol):
        raise TypeError("protocol must be a RegionProtocol")
    signal = _validate_signal_daily(signal_daily, EVENT_FLAGS[event_kind])
    region_frame = _validate_regions(regions, protocol)
    lobe_frame = _validate_lobes(lobes, region_frame)
    calendar_frame = _validate_calendars(calendars)
    records: list[dict[str, object]] = []

    for key, series in signal.groupby(list(SERIES_COLUMNS), sort=True):
        signal_id, direction, version = (str(value) for value in key)
        series = series.sort_values("date").reset_index(drop=True)
        series_dates = pd.DatetimeIndex(series["date"])
        series_start = pd.Timestamp(series_dates.min())
        series_end = pd.Timestamp(series_dates.max())
        if not _has_aligned_reference_calendar(series_dates, calendar_frame):
            raise ValueError(f"signal calendar mismatch for {signal_id}")
        position_by_date = {
            pd.Timestamp(date): position for position, date in enumerate(series_dates)
        }
        events = series.loc[
            series[EVENT_FLAGS[event_kind]], ["episode_id", "date"]
        ].rename(columns={"date": "event_date"})

        for index_id, calendar in calendar_frame.groupby("index_id", sort=True):
            calendar = calendar.sort_values("date").reset_index(drop=True)
            index_name = str(calendar["index_name"].iloc[0])
            calendar_dates = pd.DatetimeIndex(calendar["date"])
            evaluation_dates = series_dates[
                (series_dates >= max(series_start, pd.Timestamp(calendar_dates.min())))
                & (series_dates <= min(series_end, pd.Timestamp(calendar_dates.max())))
            ]
            if evaluation_dates.empty:
                continue
            coverage_start = pd.Timestamp(evaluation_dates.min())
            coverage_end = pd.Timestamp(evaluation_dates.max())
            index_events = events[
                events["event_date"].between(coverage_start, coverage_end)
            ].copy()
            index_regions = region_frame[
                region_frame["index_id"].eq(index_id)
                & region_frame["event_type"].eq(direction)
                & region_frame["anchor_date"].between(coverage_start, coverage_end)
            ].copy()
            index_lobes = lobe_frame[
                lobe_frame["region_id"].isin(index_regions["region_id"])
            ]
            records.extend(
                _match_one_index(
                    index_events,
                    index_regions,
                    index_lobes,
                    position_by_date,
                    signal_id=signal_id,
                    direction=direction,
                    version=version,
                    event_kind=event_kind,
                    index_id=str(index_id),
                    index_name=index_name,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                    protocol=protocol,
                )
            )

    result = pd.DataFrame(records, columns=MATCH_COLUMNS)
    if result.empty:
        return result
    for column in (
        "event_date",
        "anchor_date",
        "region_start",
        "region_end",
        "coverage_start_date",
        "coverage_end_date",
    ):
        result[column] = pd.to_datetime(result[column])
    for column in (
        "lobe_count",
        "lead_lag_days",
        "absolute_distance_days",
        "window_days",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    status_order = {
        "matched": 0,
        "duplicate_alarm": 1,
        "false_alarm": 2,
        "missed_region": 3,
    }
    result["_status_order"] = result["match_status"].map(status_order)
    result["_sort_date"] = result["event_date"].fillna(result["anchor_date"])
    result = result.sort_values(
        [
            "signal_id",
            "direction",
            "version",
            "event_kind",
            "index_id",
            "_sort_date",
            "_status_order",
            "record_id",
        ]
    ).drop(columns=["_status_order", "_sort_date"])
    return result.reset_index(drop=True)


def _has_aligned_reference_calendar(
    series_dates: pd.DatetimeIndex,
    calendars: pd.DataFrame,
) -> bool:
    series_start = pd.Timestamp(series_dates.min())
    series_end = pd.Timestamp(series_dates.max())
    for _, calendar in calendars.groupby("index_id", sort=True):
        calendar_dates = pd.DatetimeIndex(calendar.sort_values("date")["date"])
        overlap_start = max(series_start, pd.Timestamp(calendar_dates.min()))
        overlap_end = min(series_end, pd.Timestamp(calendar_dates.max()))
        if overlap_start > overlap_end:
            continue
        expected = calendar_dates[
            (calendar_dates >= overlap_start) & (calendar_dates <= overlap_end)
        ]
        actual = series_dates[
            (series_dates >= overlap_start) & (series_dates <= overlap_end)
        ]
        if actual.equals(expected):
            return True
    return False


def summarize_region_matches(matches: pd.DataFrame) -> pd.DataFrame:
    """Summarize strict, loose and all-window primary matching scopes."""

    if not isinstance(matches, pd.DataFrame):
        raise TypeError("matches must be a pandas DataFrame")
    missing = set(MATCH_COLUMNS).difference(matches.columns)
    if missing:
        raise ValueError(f"matches is missing columns: {sorted(missing)}")
    if matches.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    keys = [*SERIES_COLUMNS, "event_kind", "index_id", "index_name"]
    records: list[dict[str, object]] = []
    for key, group in matches.groupby(keys, sort=True):
        identity = dict(zip(keys, key))
        region_rows = group[group["match_status"].isin(["matched", "missed_region"])]
        episode_rows = group[
            group["match_status"].isin(
                ["matched", "duplicate_alarm", "false_alarm"]
            )
        ]
        region_count = int(region_rows["region_id"].nunique())
        episode_count = int(episode_rows["episode_id"].nunique())
        duplicate_count = int(group["match_status"].eq("duplicate_alarm").sum())
        scopes = {
            "strict": group["match_status"].eq("matched")
            & group["strict_matched"],
            "loose": group["match_status"].eq("matched")
            & group["loose_matched"],
            "window": group["match_status"].eq("matched"),
        }
        for scope, selected in scopes.items():
            selected_rows = group[selected]
            matched_regions = int(selected_rows["region_id"].nunique())
            matched_episodes = int(selected_rows["episode_id"].nunique())
            leads = pd.to_numeric(
                selected_rows["lead_lag_days"], errors="coerce"
            ).dropna()
            records.append(
                {
                    **identity,
                    "match_scope": scope,
                    "region_count": region_count,
                    "matched_region_count": matched_regions,
                    "missed_region_count": region_count - matched_regions,
                    "region_recall": (
                        matched_regions / region_count if region_count else np.nan
                    ),
                    "episode_count": episode_count,
                    "matched_episode_count": matched_episodes,
                    "false_alarm_count": episode_count - matched_episodes,
                    "episode_precision": (
                        matched_episodes / episode_count if episode_count else np.nan
                    ),
                    "duplicate_alarm_count": duplicate_count,
                    "median_lead_lag_days": leads.median(),
                    "q25_lead_lag_days": leads.quantile(0.25),
                    "q75_lead_lag_days": leads.quantile(0.75),
                }
            )
    return pd.DataFrame(records, columns=METRIC_COLUMNS)


def _match_one_index(
    events: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    position_by_date: dict[pd.Timestamp, int],
    *,
    signal_id: str,
    direction: str,
    version: str,
    event_kind: str,
    index_id: str,
    index_name: str,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
    protocol: RegionProtocol,
) -> list[dict[str, object]]:
    lobe_groups = {
        region_id: frame.sort_values("lobe_id")
        for region_id, frame in lobes.groupby("region_id", sort=True)
    }
    event_records = {
        str(row.episode_id): {
            "episode_id": str(row.episode_id),
            "event_date": pd.Timestamp(row.event_date),
        }
        for row in events.itertuples(index=False)
    }
    region_records = {
        str(row.region_id): row._asdict()
        for row in regions.itertuples(index=False)
    }
    candidates: list[dict[str, object]] = []
    candidates_by_episode: dict[str, list[dict[str, object]]] = defaultdict(list)
    for episode_id, event in event_records.items():
        for region_id, region in region_records.items():
            candidate = _candidate(
                event,
                region,
                lobe_groups.get(region_id),
                position_by_date,
                protocol,
            )
            if candidate is None:
                continue
            candidate["episode_id"] = episode_id
            candidate["region_id"] = region_id
            candidates.append(candidate)
            candidates_by_episode[episode_id].append(candidate)

    candidates.sort(key=_candidate_sort_key)
    used_episodes: set[str] = set()
    used_regions: dict[str, str] = {}
    records: list[dict[str, object]] = []
    for candidate in candidates:
        episode_id = str(candidate["episode_id"])
        region_id = str(candidate["region_id"])
        if episode_id in used_episodes or region_id in used_regions:
            continue
        used_episodes.add(episode_id)
        used_regions[region_id] = episode_id
        records.append(
            _match_record(
                "matched",
                event_records[episode_id],
                region_records[region_id],
                candidate,
                signal_id=signal_id,
                direction=direction,
                version=version,
                event_kind=event_kind,
                index_id=index_id,
                index_name=index_name,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                position_by_date=position_by_date,
                protocol=protocol,
                primary_episode_id=episode_id,
            )
        )

    for episode_id, event in event_records.items():
        if episode_id in used_episodes:
            continue
        episode_candidates = sorted(
            candidates_by_episode.get(episode_id, []), key=_candidate_sort_key
        )
        if episode_candidates:
            candidate = episode_candidates[0]
            region_id = str(candidate["region_id"])
            records.append(
                _match_record(
                    "duplicate_alarm",
                    event,
                    region_records[region_id],
                    candidate,
                    signal_id=signal_id,
                    direction=direction,
                    version=version,
                    event_kind=event_kind,
                    index_id=index_id,
                    index_name=index_name,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                    position_by_date=position_by_date,
                    protocol=protocol,
                    primary_episode_id=used_regions.get(region_id, ""),
                )
            )
        else:
            records.append(
                _match_record(
                    "false_alarm",
                    event,
                    None,
                    None,
                    signal_id=signal_id,
                    direction=direction,
                    version=version,
                    event_kind=event_kind,
                    index_id=index_id,
                    index_name=index_name,
                    coverage_start=coverage_start,
                    coverage_end=coverage_end,
                    position_by_date=position_by_date,
                    protocol=protocol,
                    primary_episode_id="",
                )
            )

    for region_id, region in region_records.items():
        if region_id in used_regions:
            continue
        records.append(
            _match_record(
                "missed_region",
                None,
                region,
                None,
                signal_id=signal_id,
                direction=direction,
                version=version,
                event_kind=event_kind,
                index_id=index_id,
                index_name=index_name,
                coverage_start=coverage_start,
                coverage_end=coverage_end,
                position_by_date=position_by_date,
                protocol=protocol,
                primary_episode_id="",
            )
        )
    return records


def _candidate(
    event: dict[str, object],
    region: dict[str, object],
    lobes: pd.DataFrame | None,
    position_by_date: dict[pd.Timestamp, int],
    protocol: RegionProtocol,
) -> dict[str, object] | None:
    event_date = pd.Timestamp(event["event_date"])
    anchor_date = pd.Timestamp(region["anchor_date"])
    if event_date not in position_by_date or anchor_date not in position_by_date:
        raise ValueError("event and anchor dates must exist in the index calendar")
    lead_lag = position_by_date[event_date] - position_by_date[anchor_date]
    timing = "prediction" if lead_lag <= 0 else "confirmation"
    windows = (
        protocol.prediction_windows
        if timing == "prediction"
        else protocol.confirmation_windows
    )
    window_days = next((value for value in windows if abs(lead_lag) <= value), None)

    matched_lobe_id = ""
    level = ""
    priority = -1
    if lobes is not None and not lobes.empty:
        hits = lobes[
            lobes["lobe_start"].le(event_date)
            & lobes["lobe_end"].ge(event_date)
        ]
        if not hits.empty:
            matched_lobe_id = str(hits.sort_values("lobe_id").iloc[0]["lobe_id"])
            level = "core_lobe"
            priority = 0
    if not level and pd.Timestamp(region["region_start"]) <= event_date <= pd.Timestamp(
        region["region_end"]
    ):
        level = "envelope"
        priority = 1
    if not level:
        if window_days is None:
            return None
        level = "prediction_window" if timing == "prediction" else "confirmation_window"
        priority = 2
    return {
        "match_level": level,
        "priority": priority,
        "matched_lobe_id": matched_lobe_id,
        "timing": timing,
        "lead_lag_days": lead_lag,
        "absolute_distance_days": abs(lead_lag),
        "window_days": window_days,
        "event_date": event_date,
        "anchor_date": anchor_date,
    }


def _candidate_sort_key(candidate: dict[str, object]) -> tuple[object, ...]:
    return (
        int(candidate["priority"]),
        int(candidate["absolute_distance_days"]),
        pd.Timestamp(candidate["event_date"]),
        str(candidate.get("episode_id", "")),
        str(candidate.get("region_id", "")),
    )


def _match_record(
    status: str,
    event: dict[str, object] | None,
    region: dict[str, object] | None,
    candidate: dict[str, object] | None,
    *,
    signal_id: str,
    direction: str,
    version: str,
    event_kind: str,
    index_id: str,
    index_name: str,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
    position_by_date: dict[pd.Timestamp, int],
    protocol: RegionProtocol,
    primary_episode_id: str,
) -> dict[str, object]:
    episode_id = "" if event is None else str(event["episode_id"])
    event_date = pd.NaT if event is None else pd.Timestamp(event["event_date"])
    region_id = "" if region is None else str(region["region_id"])
    anchor_date = pd.NaT if region is None else pd.Timestamp(region["anchor_date"])
    primary_match = status == "matched"
    strict_matched = bool(
        primary_match and candidate["match_level"] == "core_lobe"
    ) if candidate else False
    loose_matched = bool(
        primary_match and candidate["match_level"] in ["core_lobe", "envelope"]
    ) if candidate else False
    prediction_complete: object = pd.NA
    confirmation_complete: object = pd.NA
    if region is not None:
        anchor_position = position_by_date[anchor_date]
        prediction_complete = (
            anchor_position - position_by_date[coverage_start]
            >= max(protocol.prediction_windows)
        )
        confirmation_complete = (
            position_by_date[coverage_end] - anchor_position
            >= max(protocol.confirmation_windows)
        )
    identity = episode_id or region_id or "empty"
    record_id = (
        f"{signal_id}::{direction}::{version}::{event_kind}::{index_id}::"
        f"{status}::{identity}"
    )
    return {
        "record_id": record_id,
        "signal_id": signal_id,
        "direction": direction,
        "version": version,
        "event_kind": event_kind,
        "index_id": index_id,
        "index_name": index_name,
        "match_status": status,
        "primary_match": primary_match,
        "episode_id": episode_id,
        "event_date": event_date,
        "region_id": region_id,
        "anchor_date": anchor_date,
        "region_start": pd.NaT if region is None else region["region_start"],
        "region_end": pd.NaT if region is None else region["region_end"],
        "region_form": (
            ""
            if region is None
            else ("multi_lobe" if int(region["lobe_count"]) > 1 else "single_lobe")
        ),
        "lobe_count": pd.NA if region is None else int(region["lobe_count"]),
        "matched_lobe_id": "" if candidate is None else candidate["matched_lobe_id"],
        "match_level": "" if candidate is None else candidate["match_level"],
        "strict_matched": strict_matched,
        "loose_matched": loose_matched,
        "timing": "" if candidate is None else candidate["timing"],
        "lead_lag_days": pd.NA if candidate is None else candidate["lead_lag_days"],
        "absolute_distance_days": (
            pd.NA if candidate is None else candidate["absolute_distance_days"]
        ),
        "window_days": pd.NA if candidate is None else candidate["window_days"],
        "primary_episode_id_for_region": primary_episode_id,
        "coverage_start_date": coverage_start,
        "coverage_end_date": coverage_end,
        "prediction_window_complete": prediction_complete,
        "confirmation_window_complete": confirmation_complete,
        "label_version": "" if region is None else region["label_version"],
    }


def _validate_signal_daily(signal: pd.DataFrame, event_flag: str) -> pd.DataFrame:
    if not isinstance(signal, pd.DataFrame):
        raise TypeError("signal_daily must be a pandas DataFrame")
    required = {*SERIES_COLUMNS, "date", "episode_id", event_flag}
    missing = required.difference(signal.columns)
    if missing:
        raise ValueError(f"signal_daily is missing columns: {sorted(missing)}")
    if signal.empty:
        raise ValueError("signal_daily must not be empty")
    result = signal.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("signal_daily contains an invalid date")
    if not result["direction"].isin(["top", "bottom"]).all():
        raise ValueError("signal direction must be top or bottom")
    if result.duplicated([*SERIES_COLUMNS, "date"]).any():
        raise ValueError("signal_daily contains duplicate series dates")
    result[event_flag] = _strict_bool(result[event_flag], event_flag)
    selected = result[result[event_flag]]
    if selected["episode_id"].isna().any() or selected["episode_id"].astype(str).eq("").any():
        raise ValueError(f"{event_flag} rows must have an episode_id")
    if selected.duplicated([*SERIES_COLUMNS, "episode_id"]).any():
        raise ValueError(f"{event_flag} must occur at most once per episode")
    return result.sort_values([*SERIES_COLUMNS, "date"]).reset_index(drop=True)


def _validate_regions(
    regions: pd.DataFrame,
    protocol: RegionProtocol,
) -> pd.DataFrame:
    if not isinstance(regions, pd.DataFrame):
        raise TypeError("regions must be a pandas DataFrame")
    required = {
        "region_id",
        "index_id",
        "index_name",
        "event_type",
        "status",
        "eligible",
        "region_start",
        "region_end",
        "anchor_date",
        "lobe_count",
        "label_version",
    }
    missing = required.difference(regions.columns)
    if missing:
        raise ValueError(f"regions is missing columns: {sorted(missing)}")
    result = regions.copy()
    for column in ("region_start", "region_end", "anchor_date"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if result[["region_start", "region_end", "anchor_date"]].isna().any().any():
        raise ValueError("regions contain invalid dates")
    result["eligible"] = _strict_bool(result["eligible"], "eligible")
    result = result[result["status"].eq("confirmed") & result["eligible"]].copy()
    if result["region_id"].duplicated().any():
        raise ValueError("region_id must be unique")
    if not result["event_type"].isin(["top", "bottom"]).all():
        raise ValueError("region event_type must be top or bottom")
    if not result["label_version"].eq(protocol.label_version).all():
        raise ValueError("region label_version does not match the protocol")
    if (
        result["region_start"].gt(result["anchor_date"]).any()
        or result["region_end"].lt(result["anchor_date"]).any()
    ):
        raise ValueError("region envelope must contain its anchor")
    return result.reset_index(drop=True)


def _validate_lobes(lobes: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(lobes, pd.DataFrame):
        raise TypeError("lobes must be a pandas DataFrame")
    required = {"region_id", "lobe_id", "lobe_start", "lobe_end"}
    missing = required.difference(lobes.columns)
    if missing:
        raise ValueError(f"lobes is missing columns: {sorted(missing)}")
    result = lobes.copy()
    for column in ("lobe_start", "lobe_end"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if result[["lobe_start", "lobe_end"]].isna().any().any():
        raise ValueError("lobes contain invalid dates")
    if result["lobe_id"].duplicated().any():
        raise ValueError("lobe_id must be unique")
    selected_ids = set(regions["region_id"])
    result = result[result["region_id"].isin(selected_ids)].copy()
    missing_lobes = selected_ids.difference(result["region_id"])
    if missing_lobes:
        raise ValueError(f"eligible regions are missing lobes: {sorted(missing_lobes)}")
    if result["lobe_start"].gt(result["lobe_end"]).any():
        raise ValueError("lobe_start must not be after lobe_end")
    return result.reset_index(drop=True)


def _validate_calendars(calendars: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(calendars, pd.DataFrame):
        raise TypeError("calendars must be a pandas DataFrame")
    required = {"index_id", "index_name", "date"}
    missing = required.difference(calendars.columns)
    if missing:
        raise ValueError(f"calendars is missing columns: {sorted(missing)}")
    result = calendars[["index_id", "index_name", "date"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        raise ValueError("calendars contain an invalid date")
    if result.duplicated(["index_id", "date"]).any():
        raise ValueError("calendar index dates must be unique")
    if result.groupby("index_id")["index_name"].nunique().gt(1).any():
        raise ValueError("calendar index_name must be stable")
    return result.sort_values(["index_id", "date"]).reset_index(drop=True)


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} contains invalid boolean values")
    return normalized.map(mapping).astype(bool)
