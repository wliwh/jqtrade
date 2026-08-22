"""Post-hoc operational labels for causal MA20 candidate episodes."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


OPERATIONAL_LABEL_VERSION = "ma20_episode_operational_window_v1"
OPERATIONAL_WINDOW_TRADE_DAYS = 5


@dataclass(frozen=True)
class _MatchEdge:
    priority: int
    absolute_distance: int
    onset_date: pd.Timestamp
    episode_id: str
    region_id: str
    lead_lag_days: int
    match_level: str

    def sort_key(self) -> tuple[object, ...]:
        return (
            self.priority,
            self.absolute_distance,
            self.onset_date,
            self.episode_id,
            self.region_id,
        )


def build_operational_episode_labels(
    candidates: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    calendar: pd.Series | pd.DatetimeIndex,
    *,
    window_trade_days: int = OPERATIONAL_WINDOW_TRADE_DAYS,
) -> pd.DataFrame:
    """Greedily match candidate onsets to strict lobes or a symmetric window."""

    if (
        isinstance(window_trade_days, bool)
        or not isinstance(window_trade_days, int)
        or window_trade_days <= 0
    ):
        raise ValueError("window_trade_days must be a positive integer")
    source = _validate_candidates(candidates)
    dates = _validate_calendar(calendar)
    region_frame = _validate_regions(regions)
    region_frame = region_frame[
        region_frame["anchor_date"].between(dates.min(), dates.max())
    ].copy()
    if region_frame.empty:
        raise ValueError("regions has no eligible all_a rows inside calendar coverage")
    lobe_frame = _validate_lobes(lobes, region_frame)
    position_by_date = {
        pd.Timestamp(date): position for position, date in enumerate(dates)
    }

    records: dict[str, dict[str, object]] = {}
    for direction, direction_candidates in source.groupby("direction", sort=True):
        direction_regions = region_frame[
            region_frame["event_type"].eq(direction)
        ]
        direction_region_ids = set(direction_regions["region_id"].astype(str))
        direction_lobes = lobe_frame[
            lobe_frame["region_id"].astype(str).isin(direction_region_ids)
        ]
        lobe_groups = {
            str(region_id): group
            for region_id, group in direction_lobes.groupby("region_id", sort=True)
        }
        edges_by_episode: dict[str, list[_MatchEdge]] = {}
        all_edges: list[_MatchEdge] = []
        for candidate in direction_candidates.itertuples(index=False):
            episode_id = str(candidate.candidate_episode_id)
            onset_date = pd.Timestamp(candidate.onset_date)
            if onset_date not in position_by_date:
                raise ValueError(f"candidate onset is outside calendar: {onset_date}")
            episode_edges: list[_MatchEdge] = []
            for region in direction_regions.itertuples(index=False):
                region_id = str(region.region_id)
                anchor_date = pd.Timestamp(region.anchor_date)
                if anchor_date not in position_by_date:
                    raise ValueError(
                        f"region anchor is outside calendar: {anchor_date}"
                    )
                lead_lag = position_by_date[onset_date] - position_by_date[anchor_date]
                strict = _inside_core_lobe(
                    onset_date, lobe_groups.get(region_id)
                )
                if not strict and abs(lead_lag) > window_trade_days:
                    continue
                edge = _MatchEdge(
                    priority=0 if strict else 1,
                    absolute_distance=abs(lead_lag),
                    onset_date=onset_date,
                    episode_id=episode_id,
                    region_id=region_id,
                    lead_lag_days=lead_lag,
                    match_level="core_lobe" if strict else "operational_window",
                )
                episode_edges.append(edge)
                all_edges.append(edge)
            edges_by_episode[episode_id] = sorted(
                episode_edges, key=_MatchEdge.sort_key
            )

        used_episodes: set[str] = set()
        used_regions: set[str] = set()
        for edge in sorted(all_edges, key=_MatchEdge.sort_key):
            if edge.episode_id in used_episodes or edge.region_id in used_regions:
                continue
            used_episodes.add(edge.episode_id)
            used_regions.add(edge.region_id)
            records[edge.episode_id] = _label_record(
                edge,
                status="matched",
                primary_match=True,
                window_trade_days=window_trade_days,
            )

        for candidate in direction_candidates.itertuples(index=False):
            episode_id = str(candidate.candidate_episode_id)
            if episode_id in records:
                continue
            episode_edges = edges_by_episode[episode_id]
            if episode_edges:
                records[episode_id] = _label_record(
                    episode_edges[0],
                    status="duplicate_candidate",
                    primary_match=False,
                    window_trade_days=window_trade_days,
                )
            else:
                records[episode_id] = {
                    "candidate_episode_id": episode_id,
                    "operational_match_status": "false_alarm",
                    "operational_primary_match": False,
                    "target_operational_match": False,
                    "target_operational_strict_match": False,
                    "operational_region_id": "",
                    "operational_match_level": "",
                    "operational_lead_lag_days": pd.NA,
                    "operational_label_version": OPERATIONAL_LABEL_VERSION,
                    "operational_window_trade_days": window_trade_days,
                }

    result = pd.DataFrame([records[value] for value in source["candidate_episode_id"]])
    result["operational_lead_lag_days"] = pd.to_numeric(
        result["operational_lead_lag_days"], errors="coerce"
    ).astype("Int64")
    return result


def _label_record(
    edge: _MatchEdge,
    *,
    status: str,
    primary_match: bool,
    window_trade_days: int,
) -> dict[str, object]:
    return {
        "candidate_episode_id": edge.episode_id,
        "operational_match_status": status,
        "operational_primary_match": primary_match,
        "target_operational_match": primary_match,
        "target_operational_strict_match": (
            primary_match and edge.match_level == "core_lobe"
        ),
        "operational_region_id": edge.region_id,
        "operational_match_level": edge.match_level,
        "operational_lead_lag_days": edge.lead_lag_days,
        "operational_label_version": OPERATIONAL_LABEL_VERSION,
        "operational_window_trade_days": window_trade_days,
    }


def _inside_core_lobe(
    date: pd.Timestamp,
    lobes: pd.DataFrame | None,
) -> bool:
    if lobes is None or lobes.empty:
        return False
    return bool(
        (lobes["lobe_start"].le(date) & lobes["lobe_end"].ge(date)).any()
    )


def _validate_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_episode_id", "direction", "onset_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"candidates is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("candidates must not be empty")
    result = frame.loc[:, ["candidate_episode_id", "direction", "onset_date"]].copy()
    result["candidate_episode_id"] = result["candidate_episode_id"].astype(str)
    result["onset_date"] = pd.to_datetime(result["onset_date"], errors="coerce")
    if result["candidate_episode_id"].str.strip().eq("").any():
        raise ValueError("candidate_episode_id must be non-empty")
    if result["candidate_episode_id"].duplicated().any():
        raise ValueError("candidate_episode_id must be unique")
    if result["onset_date"].isna().any():
        raise ValueError("candidate onset_date must be valid")
    if not result["direction"].isin(["top", "bottom"]).all():
        raise ValueError("candidate direction must be top or bottom")
    return result.sort_values(["onset_date", "candidate_episode_id"]).reset_index(
        drop=True
    )


def _validate_regions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"region_id", "index_id", "event_type", "eligible", "anchor_date"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"regions is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["anchor_date"] = pd.to_datetime(result["anchor_date"], errors="coerce")
    if result["anchor_date"].isna().any():
        raise ValueError("regions contains an invalid anchor_date")
    eligible = _strict_bool(result["eligible"], "eligible")
    result = result[eligible & result["index_id"].astype(str).eq("all_a")].copy()
    if result.empty:
        raise ValueError("regions has no eligible all_a rows")
    if result["region_id"].astype(str).duplicated().any():
        raise ValueError("region_id must be unique")
    if not result["event_type"].isin(["top", "bottom"]).all():
        raise ValueError("region event_type must be top or bottom")
    result["region_id"] = result["region_id"].astype(str)
    return result


def _validate_lobes(frame: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    required = {"lobe_id", "region_id", "lobe_start", "lobe_end"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"lobes is missing columns: {sorted(missing)}")
    result = frame.copy()
    for column in ("lobe_start", "lobe_end"):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    if result[["lobe_start", "lobe_end"]].isna().any().any():
        raise ValueError("lobes contains an invalid date")
    result["region_id"] = result["region_id"].astype(str)
    result = result[result["region_id"].isin(regions["region_id"])].copy()
    if (result["lobe_start"] > result["lobe_end"]).any():
        raise ValueError("lobe_start must not exceed lobe_end")
    return result


def _validate_calendar(values: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    if len(dates) == 0 or dates.isna().any():
        raise ValueError("calendar must contain valid dates")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("calendar dates must be unique and increasing")
    return dates


def _strict_bool(values: pd.Series, name: str) -> pd.Series:
    if values.isna().any():
        raise ValueError(f"{name} must not contain missing values")
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise ValueError(f"{name} must contain only booleans")
    return normalized.map(mapping).astype(bool)
