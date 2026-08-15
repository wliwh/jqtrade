"""Analyze the JQ four-industry Top1 signal against index turning points."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


LEAD_WINDOWS = (0, 5, 10, 20)
FORWARD_HORIZONS = (5, 10, 20, 60)
THRESHOLD_LEVELS = ("small", "medium", "large")
EVENT_TYPES = ("top", "bottom")
TARGET_IDS = ("bank", "coal", "nonferrous", "steel")
BREADTH_FILTER_THRESHOLD = 0.50
BREADTH_FILTERS = (
    ("breadth_le_50", "全市场宽度≤50%", "le"),
    ("breadth_gt_50", "全市场宽度>50%", "gt"),
)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PACKAGE_DIR = (
    PROJECT_DIR / "datas" / "all_a_breadth_v1_20120101_20260814"
)
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "artifacts" / "four_industry_top1"


def _as_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.lower()
    unknown = ~values.isin(["true", "false"])
    if unknown.any():
        raise ValueError("invalid boolean values: %s" % sorted(values[unknown].unique()))
    return values.eq("true")


def load_and_validate_package(package_dir):
    """Validate manifest hashes and return the two exported tables."""

    package_dir = Path(package_dir)
    manifest_path = package_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = {}
    for record in manifest["files"]:
        relative_path = record["path"]
        path = package_dir / relative_path
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != record["sha256"]:
            raise ValueError("SHA-256 mismatch: %s" % relative_path)
        frame = pd.read_csv(path, encoding=record.get("encoding", "utf-8-sig"))
        if len(frame) != record["rows"]:
            raise ValueError("row count mismatch: %s" % relative_path)
        if list(frame.columns) != record["columns"]:
            raise ValueError("column mismatch: %s" % relative_path)
        frames[relative_path] = frame

    daily = frames["data/daily_summary.csv"]
    industry = frames["data/industry_breadth.csv"]
    daily["date"] = pd.to_datetime(daily["date"])
    industry["date"] = pd.to_datetime(industry["date"])
    if not daily["date"].is_unique or not daily["date"].is_monotonic_increasing:
        raise ValueError("daily summary dates must be unique and increasing")
    if industry.duplicated(["date", "industry_code", "industry_name"]).any():
        raise ValueError("duplicate date-industry rows")
    return daily, industry, manifest


def _add_phase_columns(frame, trigger, prefix=""):
    """Annotate mutually exclusive phases for one boolean signal series."""

    def column(name):
        return "%s_%s" % (prefix, name) if prefix else name

    triggered = trigger.astype(bool)
    onset = triggered & ~triggered.shift(fill_value=False)
    continuation = triggered & ~onset
    exit_signal = ~triggered & triggered.shift(fill_value=False)

    frame[column("triggered")] = triggered
    frame[column("onset")] = onset
    frame[column("continuation")] = continuation
    frame[column("exit")] = exit_signal
    frame[column("phase")] = "inactive"
    frame.loc[exit_signal, column("phase")] = "exit"
    frame.loc[continuation, column("phase")] = "continuation"
    frame.loc[onset, column("phase")] = "onset"

    active_group = onset.cumsum()
    frame[column("episode_id")] = pd.Series(
        pd.NA, index=frame.index, dtype="Int64"
    )
    frame.loc[triggered, column("episode_id")] = active_group.loc[
        triggered
    ].astype(int)
    frame[column("episode_day")] = pd.Series(
        pd.NA, index=frame.index, dtype="Int64"
    )
    frame.loc[triggered, column("episode_day")] = (
        frame.loc[triggered]
        .groupby(column("episode_id"), sort=True)
        .cumcount()
        .add(1)
    )


def prepare_valid_signal(daily):
    """Keep only dates where all four target industries are observable."""

    daily = daily.copy()
    complete = daily["four_industry_present_count"].eq(len(TARGET_IDS))
    if not complete.any():
        raise ValueError("no dates have all four target industries")
    first_complete = daily.loc[complete, "date"].min()
    after_start = daily["date"].ge(first_complete)
    if not complete.loc[after_start].all():
        raise ValueError("four-industry coverage is not continuous after its first full date")

    signal = daily.loc[complete].copy().reset_index(drop=True)
    raw_trigger = _as_bool(signal["four_industry_top1_triggered"])
    _add_phase_columns(signal, raw_trigger)
    market_breadth = pd.to_numeric(signal["breadth_ma20"], errors="coerce")
    if market_breadth.isna().any():
        raise ValueError("breadth_ma20 is missing in the full-coverage period")
    market_high = market_breadth.gt(BREADTH_FILTER_THRESHOLD)
    market_low = ~market_high
    _add_phase_columns(signal, market_high, prefix="market_breadth_gt_50")
    _add_phase_columns(signal, market_low, prefix="market_breadth_le_50")
    for filter_id, _, direction in BREADTH_FILTERS:
        if direction == "le":
            regime = market_low
            regime_prefix = "market_breadth_le_50"
        else:
            regime = market_high
            regime_prefix = "market_breadth_gt_50"
        filtered_trigger = raw_trigger & regime
        _add_phase_columns(signal, filtered_trigger, prefix=filter_id)
        combined_onset = signal["%s_onset" % filter_id]
        industry_onset = signal["onset"]
        breadth_onset = signal["%s_onset" % regime_prefix]
        combined_exit = signal["%s_exit" % filter_id]
        industry_exit = signal["exit"]
        breadth_exit = signal["%s_exit" % regime_prefix]
        signal["%s_onset_industry_only" % filter_id] = (
            combined_onset & industry_onset & ~breadth_onset
        )
        signal["%s_onset_breadth_only" % filter_id] = (
            combined_onset & breadth_onset & ~industry_onset
        )
        signal["%s_onset_both" % filter_id] = (
            combined_onset & industry_onset & breadth_onset
        )
        signal["%s_exit_industry_only" % filter_id] = (
            combined_exit & industry_exit & ~breadth_exit
        )
        signal["%s_exit_breadth_only" % filter_id] = (
            combined_exit & breadth_exit & ~industry_exit
        )
        signal["%s_exit_both" % filter_id] = (
            combined_exit & industry_exit & breadth_exit
        )
    for target_id in TARGET_IDS:
        column = "target_%s_is_top1_ma20" % target_id
        signal["target_%s" % target_id] = _as_bool(signal[column])
    return signal


def _event_variants():
    """Return signal columns and the state used for conditional baselines."""

    variants = [
        ("four_industry_top1", "triggered", None, True),
        ("four_industry_top1_onset", "onset", "triggered", True),
        ("four_industry_top1_exit", "exit", "triggered", False),
        (
            "market_breadth_cross_up_50",
            "market_breadth_gt_50_onset",
            "market_breadth_gt_50_triggered",
            True,
        ),
        (
            "market_breadth_cross_down_50",
            "market_breadth_gt_50_exit",
            "market_breadth_gt_50_triggered",
            False,
        ),
    ]
    for filter_id, _, _ in BREADTH_FILTERS:
        regime_column = "market_%s_triggered" % filter_id
        variants.append(
            (
                "four_industry_top1_%s_onset" % filter_id,
                "%s_onset" % filter_id,
                regime_column,
                True,
            )
        )
        for reason in ("industry_only", "breadth_only", "both"):
            variants.append(
                (
                    "four_industry_top1_%s_onset_%s" % (filter_id, reason),
                    "%s_onset_%s" % (filter_id, reason),
                    regime_column,
                    True,
                )
            )
        variants.append(
            (
                "four_industry_top1_%s_exit" % filter_id,
                "%s_exit" % filter_id,
                "__mixed_exit__",
                False,
            )
        )
        for reason in ("industry_only", "breadth_only", "both"):
            variants.append(
                (
                    "four_industry_top1_%s_exit_%s" % (filter_id, reason),
                    "%s_exit_%s" % (filter_id, reason),
                    regime_column,
                    reason == "industry_only",
                )
            )
    return variants


def build_episodes(signal, prefix=""):
    """Collapse consecutive trigger days without altering the daily signal."""

    def column(name):
        return "%s_%s" % (prefix, name) if prefix else name

    records = []
    for episode_id, group in signal.loc[signal[column("triggered")]].groupby(
        column("episode_id"), sort=True
    ):
        target_ids = set()
        for value in group["four_industry_top1_ids"].fillna(""):
            target_ids.update(item for item in value.split("|") if item)
        end_position = int(group.index[-1])
        exit_date = pd.NaT
        if end_position + 1 < len(signal):
            exit_date = signal.loc[end_position + 1, "date"]
        records.append(
            {
                "filter_id": prefix or "unfiltered",
                "episode_id": int(episode_id),
                "start_date": group["date"].iloc[0],
                "end_date": group["date"].iloc[-1],
                "exit_date": exit_date,
                "trading_days": len(group),
                "target_ids_seen": "|".join(sorted(target_ids)),
                "max_top1_tie_count": int(group["top1_tie_count_ma20"].max()),
            }
        )
    return pd.DataFrame(records)


def build_coverage_tables(daily, signal, episodes):
    records = []
    for target_id in TARGET_IDS:
        mapped = daily["target_%s_mapping_count" % target_id].gt(0)
        top1 = _as_bool(daily["target_%s_is_top1_ma20" % target_id])
        records.append(
            {
                "target_id": target_id,
                "mapped_days": int(mapped.sum()),
                "first_mapped_date": daily.loc[mapped, "date"].min(),
                "last_mapped_date": daily.loc[mapped, "date"].max(),
                "top1_days_all_export": int(top1.sum()),
                "top1_days_full_coverage": int(signal["target_%s" % target_id].sum()),
            }
        )
    coverage = pd.DataFrame(records)

    quality = pd.DataFrame(
        [
            {
                "export_start": daily["date"].min(),
                "export_end": daily["date"].max(),
                "export_days": len(daily),
                "full_coverage_start": signal["date"].min(),
                "full_coverage_end": signal["date"].max(),
                "full_coverage_days": len(signal),
                "trigger_days": int(signal["triggered"].sum()),
                "trigger_rate": float(signal["triggered"].mean()),
                "episodes": len(episodes),
                "onset_days": int(signal["onset"].sum()),
                "continuation_days": int(signal["continuation"].sum()),
                "exit_days": int(signal["exit"].sum()),
                "inactive_days": int(signal["phase"].eq("inactive").sum()),
                "median_episode_days": float(episodes["trading_days"].median()),
                "max_episode_days": int(episodes["trading_days"].max()),
                "multi_top1_tie_days": int(
                    signal["top1_tie_count_ma20"].gt(1).sum()
                ),
                "trigger_days_with_multi_top1_tie": int(
                    (
                        signal["triggered"]
                        & signal["top1_tie_count_ma20"].gt(1)
                    ).sum()
                ),
                "max_top1_tie_count": int(signal["top1_tie_count_ma20"].max()),
                "max_close_missing": int(daily["close_missing_count"].max()),
                "max_status_missing": int(
                    daily[["paused_status_missing_count", "st_status_missing_count"]]
                    .max()
                    .max()
                ),
                "max_base_valid_missing_industry": int(
                    daily["base_valid_missing_industry_count"].max()
                ),
            }
        ]
    )
    return quality, coverage


def compute_event_metrics(signal, labels, outcomes):
    """Compute event metrics for the active state and episode onsets."""

    labels = labels.copy()
    labels["anchor_date"] = pd.to_datetime(labels["anchor_date"])
    labels["confirmation_date"] = pd.to_datetime(labels["confirmation_date"])
    labels["eligible"] = _as_bool(labels["eligible"])
    labels = labels[
        labels["status"].eq("confirmed") & labels["eligible"]
    ].copy()

    outcomes = outcomes.copy()
    outcomes["date"] = pd.to_datetime(outcomes["date"])
    signal_by_date = signal.set_index("date")
    start_date = signal["date"].min()
    end_date = signal["date"].max()
    metric_records = []
    match_records = []

    for (index_id, index_name), index_outcomes in outcomes.groupby(
        ["index_id", "index_name"], sort=False
    ):
        calendar = pd.DatetimeIndex(index_outcomes["date"].sort_values())
        calendar = calendar[(calendar >= start_date) & (calendar <= end_date)]
        aligned = signal_by_date.reindex(calendar)
        if aligned["triggered"].isna().any():
            raise ValueError("signal calendar mismatch: %s" % index_id)
        position_by_date = {date: position for position, date in enumerate(calendar)}

        for (
            signal_id,
            signal_column,
            conditional_column,
            conditional_value,
        ) in _event_variants():
            trigger_values = aligned[signal_column].to_numpy(dtype=bool)
            if conditional_column is None:
                conditional_values = np.ones(len(aligned), dtype=bool)
            elif conditional_column == "__mixed_exit__":
                conditional_values = np.zeros(len(aligned), dtype=bool)
            else:
                conditional_values = aligned[conditional_column].to_numpy(
                    dtype=bool
                )
                if not conditional_value:
                    conditional_values = ~conditional_values
            for threshold_level in THRESHOLD_LEVELS:
                for event_type in EVENT_TYPES:
                    selected = labels[
                        labels["index_id"].eq(index_id)
                        & labels["threshold_level"].eq(threshold_level)
                        & labels["event_type"].eq(event_type)
                    ].copy()
                    selected = selected[selected["anchor_date"].isin(calendar)]
                    selected["calendar_position"] = selected["anchor_date"].map(
                        position_by_date
                    )
                    selected = selected.sort_values("anchor_date")
                    event_positions = set(selected["calendar_position"].astype(int))

                    for lead_window in LEAD_WINDOWS:
                        evaluable_events = selected[
                            selected["calendar_position"].ge(lead_window)
                        ]
                        matched_leads = []
                        for event in evaluable_events.itertuples(index=False):
                            event_position = int(event.calendar_position)
                            window_start = event_position - lead_window
                            relative_hits = np.flatnonzero(
                                trigger_values[window_start : event_position + 1]
                            )
                            matched = len(relative_hits) > 0
                            trigger_date = pd.NaT
                            nearest_lead = np.nan
                            trigger_ids = ""
                            if matched:
                                trigger_position = window_start + int(relative_hits[-1])
                                trigger_date = calendar[trigger_position]
                                nearest_lead = event_position - trigger_position
                                trigger_ids = aligned.iloc[trigger_position][
                                    "four_industry_top1_ids"
                                ]
                                if pd.isna(trigger_ids):
                                    trigger_ids = ""
                                matched_leads.append(nearest_lead)
                            match_records.append(
                                {
                                    "index_id": index_id,
                                    "index_name": index_name,
                                    "signal_id": signal_id,
                                    "threshold_level": threshold_level,
                                    "event_type": event_type,
                                    "threshold": event.threshold,
                                    "anchor_date": event.anchor_date,
                                    "confirmation_date": event.confirmation_date,
                                    "lead_window_days": lead_window,
                                    "matched": matched,
                                    "nearest_trigger_date": trigger_date,
                                    "nearest_lead_days": nearest_lead,
                                    "nearest_trigger_ids": trigger_ids,
                                }
                            )

                        last_evaluable_start = len(calendar) - 1 - lead_window
                        evaluable_start_count = max(last_evaluable_start + 1, 0)
                        evaluable_signal = trigger_values[:evaluable_start_count]
                        signal_positions = np.flatnonzero(evaluable_signal)

                        def has_forward_event(position):
                            return any(
                                future_position in event_positions
                                for future_position in range(
                                    position, position + lead_window + 1
                                )
                            )

                        hit_signal_days = sum(
                            has_forward_event(int(position))
                            for position in signal_positions
                        )
                        all_positions = range(evaluable_start_count)
                        baseline_hits = sum(
                            has_forward_event(position) for position in all_positions
                        )
                        evaluable_conditional = conditional_values[
                            :evaluable_start_count
                        ]
                        conditional_positions = np.flatnonzero(
                            evaluable_conditional
                        )
                        conditional_hits = sum(
                            has_forward_event(int(position))
                            for position in conditional_positions
                        )
                        event_count = len(evaluable_events)
                        matched_count = len(matched_leads)
                        signal_count = len(signal_positions)
                        precision = (
                            float(hit_signal_days) / signal_count
                            if signal_count
                            else np.nan
                        )
                        baseline = (
                            float(baseline_hits) / evaluable_start_count
                            if evaluable_start_count
                            else np.nan
                        )
                        conditional_baseline = (
                            float(conditional_hits) / len(conditional_positions)
                            if len(conditional_positions)
                            else np.nan
                        )
                        metric_records.append(
                            {
                                "index_id": index_id,
                                "index_name": index_name,
                                "signal_id": signal_id,
                                "threshold_level": threshold_level,
                                "event_type": event_type,
                                "lead_window_days": lead_window,
                                "event_count": event_count,
                                "matched_event_count": matched_count,
                                "event_recall": (
                                    float(matched_count) / event_count
                                    if event_count
                                    else np.nan
                                ),
                                "evaluable_trigger_days": signal_count,
                                "hit_signal_days": hit_signal_days,
                                "signal_precision": precision,
                                "unconditional_window_event_rate": baseline,
                                "precision_lift": (
                                    precision / baseline if baseline > 0 else np.nan
                                ),
                                "conditional_baseline_days": len(
                                    conditional_positions
                                ),
                                "conditional_window_event_rate": conditional_baseline,
                                "conditional_precision_lift": (
                                    precision / conditional_baseline
                                    if conditional_baseline > 0
                                    else np.nan
                                ),
                                "median_nearest_lead_days": (
                                    float(np.median(matched_leads))
                                    if matched_leads
                                    else np.nan
                                ),
                            }
                        )
    return pd.DataFrame(metric_records), pd.DataFrame(match_records)


def _outcome_stats(frame, horizon):
    columns = {
        "max_down": "future_max_down_%dd" % horizon,
        "max_up": "future_max_up_%dd" % horizon,
        "return": "future_return_%dd" % horizon,
    }
    clean = frame[list(columns.values())].dropna()
    return {
        "n": len(clean),
        "median_max_down": clean[columns["max_down"]].median(),
        "median_max_up": clean[columns["max_up"]].median(),
        "median_return": clean[columns["return"]].median(),
        "mean_return": clean[columns["return"]].mean(),
        "negative_return_rate": clean[columns["return"]].lt(0).mean(),
    }


def compute_forward_comparisons(signal, outcomes):
    """Compare signal phases with explicit, non-overlapping control groups."""

    variants = {
        "four_industry_top1": {
            "sample": "triggered",
            "control": "inactive_state",
            "sample_definition": "all active days",
            "control_definition": "all inactive days",
        },
        "four_industry_top1_onset": {
            "sample": "onset",
            "control": "inactive_state",
            "sample_definition": "first active day of each episode",
            "control_definition": "all inactive days",
        },
        "four_industry_top1_continuation": {
            "sample": "continuation",
            "control": "inactive_state",
            "sample_definition": "active days after episode onset",
            "control_definition": "all inactive days",
        },
        "four_industry_top1_exit": {
            "sample": "exit",
            "control": "ordinary_inactive",
            "sample_definition": "first inactive day after each episode",
            "control_definition": "inactive days excluding exits",
        },
    }
    for filter_id, filter_name, _ in BREADTH_FILTERS:
        variants["four_industry_top1_%s_onset" % filter_id] = {
            "sample": "%s_onset" % filter_id,
            "control": "%s_inactive_state" % filter_id,
            "sample_definition": "first day after %s filter" % filter_name,
            "control_definition": "days inactive after %s filter" % filter_name,
        }
        variants["four_industry_top1_%s_exit" % filter_id] = {
            "sample": "%s_exit" % filter_id,
            "control": "%s_ordinary_inactive" % filter_id,
            "sample_definition": "first exit day after %s filter" % filter_name,
            "control_definition": (
                "filtered inactive days excluding exits after %s filter"
                % filter_name
            ),
        }
    for target_id in TARGET_IDS:
        variants["target_%s_top1" % target_id] = {
            "sample": "target_%s" % target_id,
            "control": "not_sample",
            "sample_definition": "%s Top1 days" % target_id,
            "control_definition": "%s not-Top1 days" % target_id,
        }
    merged = outcomes.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.merge(signal, on="date", how="inner")
    records = []
    metrics = (
        "median_max_down",
        "median_max_up",
        "median_return",
        "mean_return",
        "negative_return_rate",
    )
    for (index_id, index_name), group in merged.groupby(
        ["index_id", "index_name"], sort=False
    ):
        inactive_state = ~_as_bool(group["triggered"])
        ordinary_inactive = inactive_state & ~_as_bool(group["exit"])
        for signal_id, variant in variants.items():
            flag = _as_bool(group[variant["sample"]])
            if variant["control"] == "inactive_state":
                control_flag = inactive_state
            elif variant["control"] == "ordinary_inactive":
                control_flag = ordinary_inactive
            elif variant["control"].endswith("_inactive_state"):
                prefix = variant["control"][: -len("_inactive_state")]
                control_flag = ~_as_bool(group["%s_triggered" % prefix])
            elif variant["control"].endswith("_ordinary_inactive"):
                prefix = variant["control"][: -len("_ordinary_inactive")]
                control_flag = ~_as_bool(group["%s_triggered" % prefix]) & ~flag
            else:
                control_flag = ~flag
            for horizon in FORWARD_HORIZONS:
                triggered = _outcome_stats(group.loc[flag], horizon)
                control = _outcome_stats(group.loc[control_flag], horizon)
                record = {
                    "index_id": index_id,
                    "index_name": index_name,
                    "signal_id": signal_id,
                    "sample_definition": variant["sample_definition"],
                    "control_definition": variant["control_definition"],
                    "horizon_days": horizon,
                    "trigger_n": triggered["n"],
                    "control_n": control["n"],
                }
                for metric in metrics:
                    record["trigger_%s" % metric] = triggered[metric]
                    record["control_%s" % metric] = control[metric]
                    record["difference_%s" % metric] = (
                        triggered[metric] - control[metric]
                    )
                records.append(record)
    return pd.DataFrame(records)


def _percent(value):
    return "" if pd.isna(value) else "%.1f%%" % (100.0 * value)


def _number(value, digits=2):
    return "" if pd.isna(value) else ("%%.%df" % digits) % value


def _markdown_table(frame):
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    quality, coverage, episodes, filtered_episodes, event_metrics, forward
):
    quality_row = quality.iloc[0]
    medium_all_windows = event_metrics[
        event_metrics["threshold_level"].eq("medium")
    ].copy()
    medium_events = medium_all_windows[
        medium_all_windows["lead_window_days"].eq(5)
    ].copy()

    def event_comparison_table(event_type):
        selected = medium_events[medium_events["event_type"].eq(event_type)]
        active = selected[
            selected["signal_id"].eq("four_industry_top1")
        ].reset_index(drop=True)
        onset = selected[
            selected["signal_id"].eq("four_industry_top1_onset")
        ].reset_index(drop=True)
        return pd.DataFrame(
            {
                "指数": active["index_name"],
                "事件数": active["event_count"],
                "活跃日召回": active["event_recall"].map(_percent),
                "活跃日精确率倍数": active["precision_lift"].map(
                    lambda x: _number(x, 2)
                ),
                "首次触发召回": onset["event_recall"].map(_percent),
                "首次触发精确率": onset["signal_precision"].map(_percent),
                "首次触发倍数": onset["precision_lift"].map(
                    lambda x: _number(x, 2)
                ),
                "首次触发最近提前量": onset["median_nearest_lead_days"].map(
                    lambda x: "" if pd.isna(x) else "%.1f日" % x
                ),
            }
        )

    def filtered_transition_table(filter_id):
        records = []
        index_order = medium_events["index_id"].drop_duplicates().tolist()
        for index_id in index_order:
            for stage, stage_name in (("onset", "首次触发"), ("exit", "退出")):
                signal_id = "four_industry_top1_%s_%s" % (filter_id, stage)

                def selected(event_type, lead_window):
                    return medium_all_windows[
                        medium_all_windows["index_id"].eq(index_id)
                        & medium_all_windows["signal_id"].eq(signal_id)
                        & medium_all_windows["event_type"].eq(event_type)
                        & medium_all_windows["lead_window_days"].eq(lead_window)
                    ].iloc[0]

                top_0 = selected("top", 0)
                top_5 = selected("top", 5)
                bottom_0 = selected("bottom", 0)
                bottom_5 = selected("bottom", 5)
                records.append(
                    {
                        "指数": top_5["index_name"],
                        "阶段": stage_name,
                        "顶部0日命中": "%d/%d" % (
                            top_0["matched_event_count"],
                            top_0["event_count"],
                        ),
                        "顶部0—5日召回/倍数": "%s / %s" % (
                            _percent(top_5["event_recall"]),
                            _number(top_5["precision_lift"], 2),
                        ),
                        "底部0日命中": "%d/%d" % (
                            bottom_0["matched_event_count"],
                            bottom_0["event_count"],
                        ),
                        "底部0—5日召回/倍数": "%s / %s" % (
                            _percent(bottom_5["event_recall"]),
                            _number(bottom_5["precision_lift"], 2),
                        ),
                    }
                )
        return pd.DataFrame(records)

    duration_records = []
    for label, mask in (
        ("1日", episodes["trading_days"].eq(1)),
        ("2—5日", episodes["trading_days"].between(2, 5)),
        ("6—10日", episodes["trading_days"].between(6, 10)),
        ("11日以上", episodes["trading_days"].ge(11)),
    ):
        selected = episodes.loc[mask]
        duration_records.append(
            {
                "区间长度": label,
                "区间数": len(selected),
                "区间占比": _percent(float(len(selected)) / len(episodes)),
                "贡献活跃日": int(selected["trading_days"].sum()),
            }
        )
    duration_table = pd.DataFrame(duration_records)

    combined_20 = forward[
        forward["signal_id"].eq("four_industry_top1")
        & forward["horizon_days"].eq(20)
    ].copy()
    forward_table = pd.DataFrame(
        {
            "指数": combined_20["index_name"],
            "触发样本": combined_20["trigger_n"],
            "非触发样本": combined_20["control_n"],
            "触发后中位最大下行": combined_20["trigger_median_max_down"].map(_percent),
            "非触发中位最大下行": combined_20["control_median_max_down"].map(_percent),
            "中位期末收益差": combined_20["difference_median_return"].map(_percent),
        }
    )

    phase_ids = (
        "four_industry_top1",
        "four_industry_top1_onset",
        "four_industry_top1_continuation",
        "four_industry_top1_exit",
    )
    phase_labels = {
        "four_industry_top1": "全部活跃日",
        "four_industry_top1_onset": "首次触发",
        "four_industry_top1_continuation": "持续期",
        "four_industry_top1_exit": "退出日",
    }
    phase_20 = forward[
        forward["signal_id"].isin(phase_ids)
        & forward["horizon_days"].eq(20)
    ].copy()
    phase_rows = []
    for index_id, group in phase_20.groupby("index_id", sort=False):
        by_signal = group.set_index("signal_id")
        row = {"指数": group["index_name"].iloc[0]}
        for signal_id in phase_ids:
            value = by_signal.loc[signal_id]
            row[phase_labels[signal_id]] = "%s（n=%d）" % (
                _percent(value["difference_median_return"]),
                value["trigger_n"],
            )
        phase_rows.append(row)
    phase_table = pd.DataFrame(phase_rows)

    all_a_onset = forward[
        forward["index_id"].eq("all_a")
        & forward["signal_id"].eq("four_industry_top1_onset")
    ].copy()
    onset_horizon_table = pd.DataFrame(
        {
            "期限": all_a_onset["horizon_days"].map(lambda x: "%d日" % x),
            "首次触发样本": all_a_onset["trigger_n"],
            "中位最大下行": all_a_onset["trigger_median_max_down"].map(_percent),
            "中位期末收益": all_a_onset["trigger_median_return"].map(_percent),
            "相对非活跃日收益差": all_a_onset["difference_median_return"].map(
                _percent
            ),
        }
    )

    filtered_forward_ids = []
    for filter_id, _, _ in BREADTH_FILTERS:
        filtered_forward_ids.extend(
            [
                "four_industry_top1_%s_onset" % filter_id,
                "four_industry_top1_%s_exit" % filter_id,
            ]
        )
    filtered_all_a = forward[
        forward["index_id"].eq("all_a")
        & forward["signal_id"].isin(filtered_forward_ids)
    ].copy()
    filter_name_by_id = {
        filter_id: filter_name for filter_id, filter_name, _ in BREADTH_FILTERS
    }
    filtered_forward_records = []
    for row in filtered_all_a.itertuples(index=False):
        filter_id = next(
            item for item in filter_name_by_id if item in row.signal_id
        )
        stage_name = "首次触发" if row.signal_id.endswith("_onset") else "退出"
        filtered_forward_records.append(
            {
                "宽度过滤": filter_name_by_id[filter_id],
                "阶段": stage_name,
                "期限": "%d日" % row.horizon_days,
                "样本": row.trigger_n,
                "中位最大下行": _percent(row.trigger_median_max_down),
                "中位期末收益": _percent(row.trigger_median_return),
                "相对对照组收益差": _percent(row.difference_median_return),
            }
        )
    filtered_forward_table = pd.DataFrame(filtered_forward_records)

    filtered_episode_records = []
    for filter_id, filter_name, _ in BREADTH_FILTERS:
        selected = filtered_episodes[filtered_episodes["filter_id"].eq(filter_id)]
        filtered_episode_records.append(
            {
                "宽度过滤": filter_name,
                "连续区间": len(selected),
                "活跃日": int(selected["trading_days"].sum()),
                "区间长度中位数": _number(selected["trading_days"].median(), 1),
                "最长区间": int(selected["trading_days"].max()),
            }
        )
    filtered_episode_table = pd.DataFrame(filtered_episode_records)

    target_all_a = forward[
        forward["index_id"].eq("all_a")
        & forward["signal_id"].isin(
            ["target_%s_top1" % target_id for target_id in TARGET_IDS]
        )
        & forward["horizon_days"].isin([20, 60])
    ].copy()
    target_all_a["目标行业"] = target_all_a["signal_id"].str.replace(
        "target_", "", regex=False
    ).str.replace("_top1", "", regex=False)
    target_table = pd.DataFrame(
        {
            "目标行业": target_all_a["目标行业"],
            "期限": target_all_a["horizon_days"].map(lambda x: "%d日" % x),
            "样本": target_all_a["trigger_n"],
            "中位最大下行": target_all_a["trigger_median_max_down"].map(_percent),
            "中位期末收益": target_all_a["trigger_median_return"].map(_percent),
            "相对非触发收益差": target_all_a["difference_median_return"].map(_percent),
        }
    )

    active_events = medium_events[
        medium_events["signal_id"].eq("four_industry_top1")
    ]
    top_lift_count = int(
        active_events.loc[active_events["event_type"].eq("top"), "precision_lift"]
        .gt(1.0)
        .sum()
    )
    bottom_lift_count = int(
        active_events.loc[
            active_events["event_type"].eq("bottom"), "precision_lift"
        ]
        .gt(1.0)
        .sum()
    )
    negative_counts = {
        signal_id: int(
            phase_20.loc[
                phase_20["signal_id"].eq(signal_id), "difference_median_return"
            ]
            .lt(0)
            .sum()
        )
        for signal_id in phase_ids
    }
    low_onset_id = "four_industry_top1_breadth_le_50_onset"
    high_onset_id = "four_industry_top1_breadth_gt_50_onset"

    def event_lift_count(signal_id, event_type):
        return int(
            medium_events.loc[
                medium_events["signal_id"].eq(signal_id)
                & medium_events["event_type"].eq(event_type),
                "precision_lift",
            ]
            .gt(1.0)
            .sum()
        )

    high_onset_top_same_day = medium_all_windows[
        medium_all_windows["signal_id"].eq(high_onset_id)
        & medium_all_windows["event_type"].eq("top")
        & medium_all_windows["lead_window_days"].eq(0)
    ]

    def all_a_20d_difference(signal_id):
        return forward.loc[
            forward["index_id"].eq("all_a")
            & forward["signal_id"].eq(signal_id)
            & forward["horizon_days"].eq(20),
            "difference_median_return",
        ].iloc[0]

    lines = [
        "# 四行业 MA20 Top1 与指数顶底关系",
        "",
        "本报告只研究关系，不开发交易策略。结果为描述性统计；连续触发和未来窗口重叠使普通独立样本显著性检验不适用。",
        "",
        "## 数据与覆盖",
        "",
        "- JQ 导出覆盖 %s—%s，共 %d 个交易日。"
        % (
            quality_row["export_start"].date(),
            quality_row["export_end"].date(),
            quality_row["export_days"],
        ),
        "- 严格四行业完整覆盖从 %s 开始，共 %d 个交易日；此前缺失行业不作为未触发。"
        % (quality_row["full_coverage_start"].date(), quality_row["full_coverage_days"]),
        "- 完整覆盖期触发 %d 天（%s），形成 %d 个连续区间；区间长度中位数 %.0f 天、最长 %d 天。"
        % (
            quality_row["trigger_days"],
            _percent(quality_row["trigger_rate"]),
            quality_row["episodes"],
            quality_row["median_episode_days"],
            quality_row["max_episode_days"],
        ),
        "- %d 个交易日存在并列 Top1，其中 %d 天触发四行业信号；单日最多 %d 个行业并列第一。"
        % (
            quality_row["multi_top1_tie_days"],
            quality_row["trigger_days_with_multi_top1_tie"],
            quality_row["max_top1_tie_count"],
        ),
        "",
        _markdown_table(
            pd.DataFrame(
                {
                    "行业": coverage["target_id"],
                    "首次可映射": coverage["first_mapped_date"].dt.strftime("%Y-%m-%d"),
                    "完整覆盖期Top1天数": coverage["top1_days_full_coverage"],
                }
            )
        ),
        "",
        "## 连续信号口径",
        "",
        "连续触发先合并为区间，再区分四种互斥阶段：`onset` 是每个区间首日，`continuation` 是同一区间后续活跃日，`exit` 是区间结束后的首个非触发日，`inactive` 是其余非触发日。",
        "",
        "- %d 个活跃日由 %d 个首次触发日和 %d 个持续期交易日组成；首次触发以区间为单位，每段只计一次。"
        % (
            quality_row["trigger_days"],
            len(episodes),
            quality_row["trigger_days"] - len(episodes),
        ),
        "- 未来结果中，全部活跃日、首次触发和持续期统一与全部非活跃日比较；退出日与剔除退出日后的普通非活跃日比较。首次触发的对照组不再混入持续期交易日。",
        "- `trading_days` 和 `episode_day` 按交易记录计数，不受周末和休市日影响。",
        "",
        _markdown_table(duration_table),
        "",
        "## 中级顶底标签：活跃状态与首次触发",
        "",
        "窗口为锚点当日及此前5个交易日。活跃日口径回答“事件是否发生在信号状态附近”；首次触发口径回答“新一段信号是否提供预警”。两者都使用收盘数据，因此0日匹配属于同时发生。",
        "",
        "顶部：",
        "",
        _markdown_table(event_comparison_table("top")),
        "",
        "底部方向对照：",
        "",
        _markdown_table(event_comparison_table("bottom")),
        "",
        "## 全市场MA20宽度过滤：首次触发与退出",
        "",
        "先以全市场 `breadth_ma20=50%` 为固定分界过滤每日四行业Top1状态，再在每个过滤结果上重新识别连续区间。`≤50%` 与 `>50%` 两组并列报告，不根据顶底结果选择方向或阈值。",
        "",
        _markdown_table(filtered_episode_table),
        "",
        "以下事件表的窗口均包含锚点当日；“0日命中”明确计为预测正确。`0—5日召回/倍数` 依次表示事件召回率和相对无条件事件概率的精确率倍数。样本开始后不足5个交易日的事件不能评价5日提前窗口，因此个别指数的0日与5日事件分母可能不同。",
        "",
        "全市场宽度≤50%：",
        "",
        _markdown_table(filtered_transition_table("breadth_le_50")),
        "",
        "全市场宽度>50%：",
        "",
        _markdown_table(filtered_transition_table("breadth_gt_50")),
        "",
        "全A过滤后首次触发与退出的未来结果：",
        "",
        _markdown_table(filtered_forward_table),
        "",
        "## 连续信号分层后的未来结果",
        "",
        "先保留全部活跃日的20日路径，便于衡量状态效应：",
        "",
        _markdown_table(forward_table),
        "",
        "下表给出各阶段相对其明确对照组的20日中位期末收益差；括号内为该阶段具有完整20日窗口的样本数。首次触发与持续期使用相同的非活跃日基准，可以直接判断弱势主要出现在区间开始还是区间内部。",
        "",
        _markdown_table(phase_table),
        "",
        "全A首次触发的分期限结果：",
        "",
        _markdown_table(onset_horizon_table),
        "",
        "## 四行业分项：全A",
        "",
        _markdown_table(target_table),
        "",
        "## 初步判断",
        "",
        "- 加入固定50%%全市场MA20宽度过滤后，宽度>50%%的首次触发在5日顶部窗口中精确率倍数大于1的指数为 %d/7，宽度≤50%%时为 %d/7；全市场较强时的新一段四行业Top1是更值得继续复核的顶部候选。"
        % (
            event_lift_count(high_onset_id, "top"),
            event_lift_count(low_onset_id, "top"),
        ),
        "- 0日命中已经计为正确，但宽度>50%%的首次触发在七指数合计只有 %d 个同日顶部命中，分布于 %d 个指数；其5日结果主要来自提前1—5日的覆盖，而不是同日重合。"
        % (
            int(high_onset_top_same_day["matched_event_count"].sum()),
            int(high_onset_top_same_day["matched_event_count"].gt(0).sum()),
        ),
        "- 全A未来20日结果呈现不同含义：宽度≤50%%的首次触发相对对照组为 %s，宽度>50%%为 %s。前者更像弱势状态开始，后者虽更接近顶部事件，但没有同步显示20日负收益；不能把顶部命中直接等同于随后下跌。"
        % (
            _percent(all_a_20d_difference(low_onset_id)),
            _percent(all_a_20d_difference(high_onset_id)),
        ),
        "- 未过滤的每日活跃状态在5日窗口内，顶部精确率高于无条件基准的指数为 %d/7，底部方向为 %d/7；不能用未过滤结果替代上述首次触发分组。"
        % (top_lift_count, bottom_lift_count),
        "- 20日中位期末收益弱于对照组的指数数目依次为：全部活跃日 %d/7、首次触发 %d/7、持续期 %d/7、退出日 %d/7。若活跃日明显弱而首次触发不弱，应该解释为状态效应，而不是首次预警能力。"
        % (
            negative_counts["four_industry_top1"],
            negative_counts["four_industry_top1_onset"],
            negative_counts["four_industry_top1_continuation"],
            negative_counts["four_industry_top1_exit"],
        ),
        "- 四行业不是同质信号，应保留分项结果；当前样本中煤炭 Top1 的后续表现最弱，而有色、钢铁并未呈现相同方向。此项是描述性发现，不能据此重新选择行业或参数。",
        "- 中级标签在完整覆盖期内每个指数仅约8—12个顶部事件，且信号触发率较高；结论需要在更多时间或其他预先登记信号上复核。",
        "",
        "完整小/中/大标签、0/5/10/20日窗口、每日阶段、连续区间和5/10/20/60日未来结果见同目录 CSV。",
    ]
    return "\n".join(lines) + "\n"


def build_concise_incremental_report(quality, signal, event_metrics):
    """Build the short report comparing market breadth and industry transitions."""

    quality_row = quality.iloc[0]
    medium = event_metrics[event_metrics["threshold_level"].eq("medium")]
    five_day = medium[medium["lead_window_days"].eq(5)]
    same_day = medium[medium["lead_window_days"].eq(0)]
    column_by_id = {
        signal_id: column
        for signal_id, column, _, _ in _event_variants()
    }

    onset_candidates = [
        ("four_industry_top1_onset", "四行业单独首次触发"),
        ("market_breadth_cross_up_50", "全市场宽度上穿50%"),
        ("four_industry_top1_breadth_gt_50_onset", "宽度>50%组合首次触发"),
        (
            "four_industry_top1_breadth_gt_50_onset_industry_only",
            "  ├ 行业变化导致",
        ),
        (
            "four_industry_top1_breadth_gt_50_onset_breadth_only",
            "  ├ 宽度上穿导致",
        ),
        ("four_industry_top1_breadth_gt_50_onset_both", "  └ 两者同日"),
        ("market_breadth_cross_down_50", "全市场宽度下穿至≤50%"),
        ("four_industry_top1_breadth_le_50_onset", "宽度≤50%组合首次触发"),
        (
            "four_industry_top1_breadth_le_50_onset_industry_only",
            "  ├ 行业变化导致",
        ),
        (
            "four_industry_top1_breadth_le_50_onset_breadth_only",
            "  ├ 宽度下穿导致",
        ),
        ("four_industry_top1_breadth_le_50_onset_both", "  └ 两者同日"),
    ]
    exit_candidates = [
        ("four_industry_top1_exit", "四行业单独退出"),
        ("market_breadth_cross_down_50", "全市场宽度下穿至≤50%"),
        ("four_industry_top1_breadth_gt_50_exit", "宽度>50%组合退出"),
        (
            "four_industry_top1_breadth_gt_50_exit_industry_only",
            "  ├ 行业退出导致",
        ),
        (
            "four_industry_top1_breadth_gt_50_exit_breadth_only",
            "  ├ 宽度下穿导致",
        ),
        ("four_industry_top1_breadth_gt_50_exit_both", "  └ 两者同日"),
        ("market_breadth_cross_up_50", "全市场宽度上穿50%"),
        ("four_industry_top1_breadth_le_50_exit", "宽度≤50%组合退出"),
        (
            "four_industry_top1_breadth_le_50_exit_industry_only",
            "  ├ 行业退出导致",
        ),
        (
            "four_industry_top1_breadth_le_50_exit_breadth_only",
            "  ├ 宽度上穿导致",
        ),
        ("four_industry_top1_breadth_le_50_exit_both", "  └ 两者同日"),
    ]

    def table(candidates):
        records = []
        for signal_id, label in candidates:
            selected_5 = five_day[five_day["signal_id"].eq(signal_id)]
            selected_0 = same_day[same_day["signal_id"].eq(signal_id)]
            top = selected_5[selected_5["event_type"].eq("top")]
            bottom = selected_5[selected_5["event_type"].eq("bottom")]
            top_0 = selected_0[selected_0["event_type"].eq("top")]
            bottom_0 = selected_0[selected_0["event_type"].eq("bottom")]

            def compact(frame):
                conditional = frame["conditional_precision_lift"].dropna()
                conditional_median = (
                    conditional.median() if not conditional.empty else np.nan
                )
                return "%d/7；%s；%s" % (
                    int(frame["precision_lift"].gt(1.0).sum()),
                    _number(frame["precision_lift"].median(), 2),
                    _number(conditional_median, 2) or "—",
                )

            records.append(
                {
                    "信号": label,
                    "次数": int(signal[column_by_id[signal_id]].sum()),
                    "顶部5日：>1指数/中位倍数/条件倍数": compact(top),
                    "顶部0日命中": int(top_0["matched_event_count"].sum()),
                    "底部5日：>1指数/中位倍数/条件倍数": compact(bottom),
                    "底部0日命中": int(bottom_0["matched_event_count"].sum()),
                }
            )
        return pd.DataFrame(records)

    lines = [
        "# 全市场宽度与四行业Top1增量关系",
        "",
        "样本为 %s—%s，共 %d 个交易日。只研究顶底关系，不开发策略。"
        % (
            quality_row["full_coverage_start"].date(),
            quality_row["full_coverage_end"].date(),
            quality_row["full_coverage_days"],
        ),
        "",
        "## 口径",
        "",
        "- 全市场宽度为站上MA20的全A股票比例，固定以50%分界。",
        "- 分别比较宽度单独跨线、四行业Top1单独变化和两者组合后的变化。",
        "- 组合变化按行业导致、宽度导致、两者同日拆分；先过滤，再识别首次触发和退出。",
        "- 0日匹配计为正确；主表使用0—5个交易日窗口。",
        "- 表内依次为：精确率倍数大于1的指数数、七指数中位倍数、同状态条件基准倍数。条件倍数大于1才表示相对同一市场状态仍有增量信息。",
        "",
        "## 首次触发",
        "",
        _markdown_table(table(onset_candidates)),
        "",
        "## 退出",
        "",
        _markdown_table(table(exit_candidates)),
        "",
        "## 结论",
        "",
        "- 宽度>50%的组合首次触发，顶部中位精确率倍数为1.46，但相对同为宽度>50%的交易日，条件倍数仅0.79；原先的7/7主要来自强市场状态筛选，不能证明四行业提供了普遍增量信息。",
        "- 只有“四行业首次触发与宽度上穿50%同日发生”仍有条件增量：7次样本、顶部条件倍数中位数1.58。样本过少，只能列为后续复核候选。",
        "- 其余首次触发、退出和底部方向的条件倍数没有形成跨指数一致优势；退出信号目前不成立。",
        "",
        "## 注意",
        "",
        "七个指数高度相关，不是七次独立验证；0日命中也是七指数合计数，可能对应同一市场日期。中级顶底事件每个指数约8—12次。完整逐指数、小/中/大标签及0/5/10/20日窗口见 `event_metrics.csv` 和 `event_matches.csv`。",
    ]
    return "\n".join(lines) + "\n"


def run_analysis(package_dir, output_dir):
    package_dir = Path(package_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily, industry, _ = load_and_validate_package(package_dir)
    signal = prepare_valid_signal(daily)
    episodes = build_episodes(signal)
    filtered_episodes = pd.concat(
        [
            build_episodes(signal, prefix=filter_id)
            for filter_id, _, _ in BREADTH_FILTERS
        ],
        ignore_index=True,
    )
    quality, coverage = build_coverage_tables(daily, signal, episodes)
    labels = pd.read_csv(PROJECT_DIR / "artifacts" / "turning_point_labels.csv")
    outcomes = pd.read_csv(PROJECT_DIR / "artifacts" / "forward_outcomes.csv")
    event_metrics, event_matches = compute_event_metrics(
        signal, labels, outcomes
    )
    forward = compute_forward_comparisons(signal, outcomes)

    outputs = {
        "quality": output_dir / "quality_summary.csv",
        "coverage": output_dir / "target_coverage.csv",
        "daily_phases": output_dir / "signal_daily_phases.csv",
        "episodes": output_dir / "trigger_episodes.csv",
        "filtered_episodes": output_dir / "filtered_trigger_episodes.csv",
        "event_metrics": output_dir / "event_metrics.csv",
        "event_matches": output_dir / "event_matches.csv",
        "forward": output_dir / "forward_comparisons.csv",
        "report": output_dir / "report.md",
    }
    quality.to_csv(outputs["quality"], index=False)
    coverage.to_csv(outputs["coverage"], index=False)
    daily_phase_columns = [
        "date",
        "breadth_ma20",
        "triggered",
        "onset",
        "continuation",
        "exit",
        "phase",
        "episode_id",
        "episode_day",
        "four_industry_top1_ids",
        "top1_tie_count_ma20",
    ]
    for filter_id, _, _ in BREADTH_FILTERS:
        daily_phase_columns.extend(
            [
                "%s_triggered" % filter_id,
                "%s_onset" % filter_id,
                "%s_continuation" % filter_id,
                "%s_exit" % filter_id,
                "%s_phase" % filter_id,
                "%s_episode_id" % filter_id,
                "%s_episode_day" % filter_id,
                "%s_onset_industry_only" % filter_id,
                "%s_onset_breadth_only" % filter_id,
                "%s_onset_both" % filter_id,
                "%s_exit_industry_only" % filter_id,
                "%s_exit_breadth_only" % filter_id,
                "%s_exit_both" % filter_id,
            ]
        )
    for prefix in ("market_breadth_le_50", "market_breadth_gt_50"):
        daily_phase_columns.extend(
            [
                "%s_triggered" % prefix,
                "%s_onset" % prefix,
                "%s_exit" % prefix,
            ]
        )
    daily_phase_columns.extend(
        ["target_%s" % target_id for target_id in TARGET_IDS]
    )
    signal[daily_phase_columns].to_csv(outputs["daily_phases"], index=False)
    episodes.to_csv(outputs["episodes"], index=False)
    filtered_episodes.to_csv(outputs["filtered_episodes"], index=False)
    event_metrics.to_csv(outputs["event_metrics"], index=False)
    event_matches.to_csv(outputs["event_matches"], index=False)
    forward.to_csv(outputs["forward"], index=False)
    outputs["report"].write_text(
        build_concise_incremental_report(quality, signal, event_metrics),
        encoding="utf-8",
    )
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    outputs = run_analysis(args.package_dir, args.output_dir)
    for name, path in outputs.items():
        print("%s: %s" % (name, path))


if __name__ == "__main__":
    main()
