"""Complete stage-D slices for region-location evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .region_matching import MATCH_COLUMNS, SERIES_COLUMNS


DIAGNOSTIC_COLUMNS = (
    "diagnostic_region_id",
    "diagnostic_timing",
    "diagnostic_region_form",
    "diagnostic_assignment",
)
METRIC_COLUMNS = (
    "signal_id",
    "direction",
    "version",
    "event_kind",
    "aggregation",
    "index_id",
    "index_name",
    "observation_unit",
    "match_scope",
    "timing_slice",
    "region_form_slice",
    "region_count",
    "matched_region_count",
    "missed_region_count",
    "region_recall",
    "episode_count",
    "matched_episode_count",
    "false_alarm_count",
    "isolated_false_alarm_count",
    "duplicate_alarm_count",
    "unclassified_episode_count",
    "episode_precision",
    "median_lead_lag_days",
    "q25_lead_lag_days",
    "q75_lead_lag_days",
)
MATCH_SCOPES = ("strict", "loose", "window")
TIMING_SLICES = ("all", "prediction", "confirmation")
REGION_FORM_SLICES = ("all", "single_lobe", "multi_lobe")


def add_diagnostic_region_slices(
    matches: pd.DataFrame,
    regions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach auditable timing/form slices to every alarm row.

    Primary and duplicate matches inherit their candidate region. A raw false
    alarm is assigned only for slicing to the nearest same-direction region in
    the evaluated coverage. This assignment never changes ``match_status`` or
    makes the alarm a match.
    """

    result = _validate_matches(matches).copy()
    region_frame = _validate_regions(regions)
    for column in DIAGNOSTIC_COLUMNS:
        result[column] = ""
    if result.empty:
        return result

    for position, row in result.iterrows():
        status = str(row["match_status"])
        if status in {"matched", "duplicate_alarm"}:
            result.loc[position, list(DIAGNOSTIC_COLUMNS)] = [
                str(row["region_id"]),
                str(row["timing"]),
                str(row["region_form"]),
                "primary_match" if status == "matched" else "duplicate_candidate",
            ]
            continue
        if status != "false_alarm":
            continue
        candidates = region_frame[
            region_frame["index_id"].astype(str).eq(str(row["index_id"]))
            & region_frame["event_type"].astype(str).eq(str(row["direction"]))
            & region_frame["anchor_date"].between(
                pd.Timestamp(row["coverage_start_date"]),
                pd.Timestamp(row["coverage_end_date"]),
            )
        ].copy()
        if candidates.empty:
            result.loc[position, "diagnostic_assignment"] = "no_region_in_coverage"
            continue
        event_date = pd.Timestamp(row["event_date"])
        candidates["_distance"] = (
            candidates["anchor_date"] - event_date
        ).abs()
        nearest = candidates.sort_values(["_distance", "anchor_date", "region_id"]).iloc[0]
        result.loc[position, list(DIAGNOSTIC_COLUMNS)] = [
            str(nearest["region_id"]),
            "prediction" if event_date <= nearest["anchor_date"] else "confirmation",
            "multi_lobe" if int(nearest["lobe_count"]) > 1 else "single_lobe",
            "nearest_region_for_slice_only",
        ]
    return result


def summarize_region_slices(matches: pd.DataFrame) -> pd.DataFrame:
    """Report direction/timing/form slices for indices and their pooled pairs."""

    frame = _validate_matches(matches)
    missing = set(DIAGNOSTIC_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(
            "matches must first be enriched by add_diagnostic_region_slices: "
            f"{sorted(missing)}"
        )
    if frame.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    records: list[dict[str, object]] = []
    base_keys = [*SERIES_COLUMNS, "event_kind"]
    for key, signal_group in frame.groupby(base_keys, sort=True):
        base_identity = dict(zip(base_keys, key))
        index_groups = [
            (
                "index",
                str(index_id),
                str(index_group["index_name"].iloc[0]),
                "region_or_episode",
                index_group,
            )
            for index_id, index_group in signal_group.groupby("index_id", sort=True)
        ]
        index_groups.append(
            (
                "all_indices",
                "__all__",
                "全部指数（指数-事件对）",
                "index_region_or_index_episode_pair",
                signal_group,
            )
        )
        for aggregation, index_id, index_name, unit, group in index_groups:
            for match_scope in MATCH_SCOPES:
                selected_match = _selected_primary_matches(group, match_scope)
                for timing_slice in TIMING_SLICES:
                    for form_slice in REGION_FORM_SLICES:
                        records.append(
                            _slice_record(
                                group,
                                selected_match,
                                base_identity=base_identity,
                                aggregation=aggregation,
                                index_id=index_id,
                                index_name=index_name,
                                observation_unit=unit,
                                match_scope=match_scope,
                                timing_slice=timing_slice,
                                form_slice=form_slice,
                            )
                        )
    return pd.DataFrame(records, columns=METRIC_COLUMNS).sort_values(
        [
            *SERIES_COLUMNS,
            "event_kind",
            "aggregation",
            "index_id",
            "match_scope",
            "timing_slice",
            "region_form_slice",
        ]
    ).reset_index(drop=True)


def _slice_record(
    group: pd.DataFrame,
    selected_match: pd.Series,
    *,
    base_identity: dict[str, object],
    aggregation: str,
    index_id: str,
    index_name: str,
    observation_unit: str,
    match_scope: str,
    timing_slice: str,
    form_slice: str,
) -> dict[str, object]:
    region_rows = group[group["match_status"].isin(["matched", "missed_region"])]
    if timing_slice == "prediction":
        region_rows = region_rows[
            _bool_series(region_rows["prediction_window_complete"])
        ]
    elif timing_slice == "confirmation":
        region_rows = region_rows[
            _bool_series(region_rows["confirmation_window_complete"])
        ]
    if form_slice != "all":
        region_rows = region_rows[region_rows["region_form"].eq(form_slice)]

    selected_rows = group[selected_match]
    if timing_slice != "all":
        selected_rows = selected_rows[selected_rows["timing"].eq(timing_slice)]
    if form_slice != "all":
        selected_rows = selected_rows[selected_rows["region_form"].eq(form_slice)]
    selected_rows = selected_rows[
        selected_rows["region_id"].isin(region_rows["region_id"])
    ]

    episode_rows = group[
        group["match_status"].isin(["matched", "duplicate_alarm", "false_alarm"])
    ]
    if timing_slice != "all":
        episode_rows = episode_rows[
            episode_rows["diagnostic_timing"].eq(timing_slice)
        ]
    if form_slice != "all":
        episode_rows = episode_rows[
            episode_rows["diagnostic_region_form"].eq(form_slice)
        ]
    selected_episode_rows = selected_rows[
        selected_rows["episode_id"].isin(episode_rows["episode_id"])
    ]

    region_count = len(region_rows)
    matched_region_count = len(selected_rows)
    episode_count = len(episode_rows)
    matched_episode_count = len(selected_episode_rows)
    leads = pd.to_numeric(
        selected_rows["lead_lag_days"], errors="coerce"
    ).dropna()
    return {
        **base_identity,
        "aggregation": aggregation,
        "index_id": index_id,
        "index_name": index_name,
        "observation_unit": observation_unit,
        "match_scope": match_scope,
        "timing_slice": timing_slice,
        "region_form_slice": form_slice,
        "region_count": region_count,
        "matched_region_count": matched_region_count,
        "missed_region_count": region_count - matched_region_count,
        "region_recall": (
            matched_region_count / region_count if region_count else np.nan
        ),
        "episode_count": episode_count,
        "matched_episode_count": matched_episode_count,
        "false_alarm_count": episode_count - matched_episode_count,
        "isolated_false_alarm_count": int(
            episode_rows["match_status"].eq("false_alarm").sum()
        ),
        "duplicate_alarm_count": int(
            episode_rows["match_status"].eq("duplicate_alarm").sum()
        ),
        "unclassified_episode_count": int(
            episode_rows["diagnostic_timing"].eq("").sum()
        ),
        "episode_precision": (
            matched_episode_count / episode_count if episode_count else np.nan
        ),
        "median_lead_lag_days": leads.median(),
        "q25_lead_lag_days": leads.quantile(0.25),
        "q75_lead_lag_days": leads.quantile(0.75),
    }


def _selected_primary_matches(group: pd.DataFrame, scope: str) -> pd.Series:
    selected = group["match_status"].eq("matched")
    if scope == "strict":
        return selected & _bool_series(group["strict_matched"])
    if scope == "loose":
        return selected & _bool_series(group["loose_matched"])
    return selected


def _validate_matches(matches: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(matches, pd.DataFrame):
        raise TypeError("matches must be a pandas DataFrame")
    missing = set(MATCH_COLUMNS).difference(matches.columns)
    if missing:
        raise ValueError(f"matches is missing columns: {sorted(missing)}")
    result = matches.copy()
    for column in (
        "event_date",
        "anchor_date",
        "coverage_start_date",
        "coverage_end_date",
    ):
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def _validate_regions(regions: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(regions, pd.DataFrame):
        raise TypeError("regions must be a pandas DataFrame")
    required = {
        "region_id",
        "index_id",
        "event_type",
        "anchor_date",
        "lobe_count",
    }
    missing = required.difference(regions.columns)
    if missing:
        raise ValueError(f"regions is missing columns: {sorted(missing)}")
    result = regions.copy()
    result["anchor_date"] = pd.to_datetime(result["anchor_date"], errors="coerce")
    if result["anchor_date"].isna().any():
        raise ValueError("regions contains an invalid anchor_date")
    if result["region_id"].duplicated().any():
        raise ValueError("regions contains duplicate region_id values")
    return result


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().eq("true")
