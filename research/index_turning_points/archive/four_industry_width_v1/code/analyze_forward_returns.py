"""Study close-to-future-close returns after four-industry Top1 signals."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze_breadth import (
    DEFAULT_PACKAGE_DIR,
    TARGET_IDS,
    load_and_validate_package,
    prepare_valid_signal,
)


HORIZONS = (1, 2, 3, 5, 10, 20, 60)
TARGET_NAMES = {
    "bank": "银行",
    "coal": "煤炭",
    "nonferrous": "有色金属",
    "steel": "钢铁",
}
DURATION_BUCKETS = (
    ("day_1", "第1日", 1, 1),
    ("day_2_3", "第2—3日", 2, 3),
    ("day_4_5", "第4—5日", 4, 5),
    ("day_6_10", "第6—10日", 6, 10),
    ("day_11_plus", "第11日以上", 11, None),
)
ACTUAL_TARGET_INDEX_ID = "csi2000"
PROXY_TARGET_INDEX_ID = "cni2000"
MIN_INFERENCE_EPISODES = 20
MIN_INFERENCE_SAMPLE = 30

PROJECT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_OUTCOMES_PATH = (
    PROJECT_DIR
    / "artifacts"
    / "ground_truth"
    / "index_ohlc_20260814"
    / "forward_outcomes.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(tempfile.gettempdir())
    / "jqtrade_four_industry_width_v1"
    / "four_industry_forward_returns"
)


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.astype(str).str.strip().str.lower()
    unknown = ~values.isin(["true", "false"])
    if unknown.any():
        raise ValueError("invalid boolean values: %s" % sorted(values[unknown].unique()))
    return values.eq("true")


def _sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg adjusted p-values, preserving missing values."""

    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(len(values), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return pd.Series(adjusted, index=pvalues.index, dtype=float)

    valid = values[valid_positions]
    order = np.argsort(valid, kind="mergesort")
    ranked = valid[order]
    ranks = np.arange(1, len(ranked) + 1, dtype=float)
    ranked_adjusted = ranked * len(ranked) / ranks
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    restored = np.empty_like(ranked_adjusted)
    restored[order] = np.minimum(ranked_adjusted, 1.0)
    adjusted[valid_positions] = restored
    return pd.Series(adjusted, index=pvalues.index, dtype=float)


def build_phase_frame(active: pd.Series) -> pd.DataFrame:
    """Split one daily signal into active, onset, continuation and exit phases."""

    active = _as_bool(pd.Series(active).reset_index(drop=True))
    onset = active & ~active.shift(fill_value=False)
    continuation = active & ~onset
    exit_signal = ~active & active.shift(fill_value=False)
    episode_group = onset.cumsum()
    episode_id = pd.Series(pd.NA, index=active.index, dtype="Int64")
    episode_id.loc[active] = episode_group.loc[active].astype(int)
    episode_day = pd.Series(pd.NA, index=active.index, dtype="Int64")
    episode_day.loc[active] = (
        episode_id.loc[active].groupby(episode_id.loc[active], sort=True).cumcount() + 1
    )
    return pd.DataFrame(
        {
            "active": active,
            "onset": onset,
            "continuation": continuation,
            "exit": exit_signal,
            "ordinary_inactive": ~active & ~exit_signal,
            "episode_id": episode_id,
            "episode_day": episode_day,
        }
    )


def build_episode_table(
    dates: pd.Series,
    phase: pd.DataFrame,
    *,
    signal_id: str,
    signal_label: str,
    industry_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Collapse consecutive active dates into one row per episode."""

    records = []
    dates = pd.to_datetime(dates).reset_index(drop=True)
    for episode_id, group in phase.loc[phase["active"]].groupby(
        "episode_id", sort=True
    ):
        positions = group.index.to_numpy(dtype=int)
        exit_position = int(positions[-1]) + 1
        records.append(
            {
                "signal_id": signal_id,
                "signal_label": signal_label,
                "industry_ids": ",".join(industry_ids),
                "industry_count": len(industry_ids),
                "episode_id": int(episode_id),
                "start_date": dates.iloc[positions[0]],
                "end_date": dates.iloc[positions[-1]],
                "exit_date": (
                    dates.iloc[exit_position] if exit_position < len(dates) else pd.NaT
                ),
                "trading_days": len(positions),
            }
        )
    return pd.DataFrame.from_records(records)


def compute_future_returns(
    close: pd.Series, horizons: tuple[int, ...] = HORIZONS
) -> pd.DataFrame:
    """Calculate close(t+N) / close(t) - 1 for complete future windows."""

    close = pd.to_numeric(close, errors="coerce")
    if close.isna().any() or close.le(0).any():
        raise ValueError("close must contain finite positive values")
    result = pd.DataFrame(index=close.index)
    for horizon in horizons:
        if not isinstance(horizon, int) or horizon <= 0:
            raise ValueError("horizons must contain positive integers")
        result["future_return_%dd" % horizon] = close.shift(-horizon).div(close).sub(1)
    return result


def _normal_two_sided_p(z_value: float) -> float:
    if not np.isfinite(z_value):
        return np.nan
    return math.erfc(abs(float(z_value)) / math.sqrt(2.0))


def newey_west_ols(
    outcome: pd.Series,
    design: pd.DataFrame,
    *,
    max_lag: int,
) -> pd.DataFrame:
    """Fit OLS and return Bartlett-kernel Newey-West inference by coefficient."""

    y = pd.to_numeric(outcome, errors="coerce").to_numpy(dtype=float)
    x = design.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    y = y[valid]
    x = x[valid]
    n, parameter_count = x.shape
    if n <= parameter_count or not n:
        return pd.DataFrame(
            {
                "coefficient": np.nan,
                "standard_error": np.nan,
                "z_value": np.nan,
                "p_value": np.nan,
                "nobs": n,
            },
            index=design.columns,
        )

    xtx_inverse = np.linalg.pinv(x.T @ x)
    coefficients = xtx_inverse @ (x.T @ y)
    residuals = y - x @ coefficients
    score = x * residuals[:, None]
    covariance_meat = score.T @ score
    effective_lag = min(int(max_lag), n - 1)
    for lag in range(1, effective_lag + 1):
        weight = 1.0 - lag / float(effective_lag + 1)
        lag_covariance = score[lag:].T @ score[:-lag]
        covariance_meat += weight * (lag_covariance + lag_covariance.T)
    covariance = xtx_inverse @ covariance_meat @ xtx_inverse
    covariance *= n / float(max(n - parameter_count, 1))
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    z_values = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan),
        where=standard_errors > 0,
    )
    return pd.DataFrame(
        {
            "coefficient": coefficients,
            "standard_error": standard_errors,
            "z_value": z_values,
            "p_value": [_normal_two_sided_p(value) for value in z_values],
            "nobs": n,
        },
        index=design.columns,
    )


def _describe(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "n": len(clean),
        "mean": clean.mean(),
        "median": clean.median(),
        "negative_rate": clean.lt(0).mean(),
    }


def _standardized_controls(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce")
    breadth = pd.to_numeric(frame["breadth_ma20"], errors="coerce")
    daily_return = close.pct_change(fill_method=None)
    controls = pd.DataFrame(
        {
            "breadth": breadth,
            "breadth_squared": breadth.pow(2),
            "breadth_change_1d": breadth.diff(),
            "breadth_change_5d": breadth.diff(5),
            "momentum_5d": close.div(close.shift(5)).sub(1),
            "momentum_20d": close.div(close.shift(20)).sub(1),
            "volatility_20d": daily_return.rolling(20).std(),
        },
        index=frame.index,
    )
    for column in controls:
        standard_deviation = controls[column].std(ddof=0)
        if np.isfinite(standard_deviation) and standard_deviation > 0:
            controls[column] = (
                controls[column] - controls[column].mean()
            ) / standard_deviation
        else:
            controls[column] = np.nan
    year_dummies = pd.get_dummies(
        pd.to_datetime(frame["date"]).dt.year,
        prefix="year",
        drop_first=True,
        dtype=float,
    )
    year_dummies.index = frame.index
    return pd.concat([controls, year_dummies], axis=1)


def _signal_definitions(
    signal: pd.DataFrame, *, exhaustive: bool
) -> list[dict[str, object]]:
    definitions = []
    sizes = range(1, len(TARGET_IDS) + 1) if exhaustive else (1, len(TARGET_IDS))
    for size in sizes:
        for industry_ids in combinations(TARGET_IDS, size):
            if not exhaustive and size == 1:
                signal_id = "target_%s" % industry_ids[0]
            elif not exhaustive:
                signal_id = "any_four"
            else:
                signal_id = "subset_%s" % "__".join(industry_ids)
            columns = ["target_%s" % target_id for target_id in industry_ids]
            active = signal[columns].apply(_as_bool).any(axis=1)
            definitions.append(
                {
                    "signal_id": signal_id,
                    "signal_label": "+".join(
                        TARGET_NAMES[target_id] for target_id in industry_ids
                    ),
                    "industry_ids": industry_ids,
                    "active": active,
                }
            )
    return definitions


def _phase_event_count(phase: pd.DataFrame, phase_name: str) -> int:
    if phase_name in ("active", "onset"):
        return int(phase["onset"].sum())
    if phase_name == "continuation":
        return int(
            phase.loc[phase["continuation"], "episode_id"].dropna().nunique()
        )
    return int(phase["exit"].sum())


def _add_inference_fields(
    tests: pd.DataFrame, *, family_prefix: str, include_phase: bool
) -> pd.DataFrame:
    """Gate small samples and apply FDR only to eligible tests."""

    tests = tests.copy()
    tests["inference_eligible"] = (
        tests["episode_count"].ge(MIN_INFERENCE_EPISODES)
        & tests["sample_n"].ge(MIN_INFERENCE_SAMPLE)
        & tests["control_n"].ge(MIN_INFERENCE_SAMPLE)
    )

    def exclusion_reason(row: pd.Series) -> str:
        reasons = []
        if row["episode_count"] < MIN_INFERENCE_EPISODES:
            reasons.append("episode_count<%d" % MIN_INFERENCE_EPISODES)
        if row["sample_n"] < MIN_INFERENCE_SAMPLE:
            reasons.append("sample_n<%d" % MIN_INFERENCE_SAMPLE)
        if row["control_n"] < MIN_INFERENCE_SAMPLE:
            reasons.append("control_n<%d" % MIN_INFERENCE_SAMPLE)
        return ";".join(reasons)

    tests["inference_exclusion_reason"] = tests.apply(exclusion_reason, axis=1)
    tests["family_id"] = family_prefix + ":" + tests["index_id"].astype(str)
    if include_phase:
        tests["family_id"] += ":" + tests["phase"].astype(str)
    tests["family_test_count"] = tests.groupby("family_id", sort=False)[
        "inference_eligible"
    ].transform("sum")
    tests["raw_q_value"] = np.nan
    tests["controlled_q_value"] = np.nan
    tests["global_raw_q_value"] = np.nan
    tests["global_controlled_q_value"] = np.nan
    for _, family in tests.loc[tests["inference_eligible"]].groupby(
        "family_id", sort=False
    ):
        tests.loc[family.index, "raw_q_value"] = benjamini_hochberg(
            family["raw_p_value"]
        )
        tests.loc[family.index, "controlled_q_value"] = benjamini_hochberg(
            family["controlled_p_value"]
        )
    eligible = tests.loc[tests["inference_eligible"]]
    tests.loc[eligible.index, "global_raw_q_value"] = benjamini_hochberg(
        eligible["raw_p_value"]
    )
    tests.loc[
        eligible.index, "global_controlled_q_value"
    ] = benjamini_hochberg(eligible["controlled_p_value"])
    return tests


def _test_record(
    *,
    index_id: str,
    index_name: str,
    signal_id: str,
    signal_label: str,
    industry_ids: tuple[str, ...],
    phase_name: str,
    horizon: int,
    phase: pd.DataFrame,
    outcome: pd.Series,
    raw_result: pd.Series,
    controlled_result: pd.Series,
) -> dict[str, object]:
    sample_flag = phase[phase_name]
    control_flag = ~phase["active"] if phase_name == "active" else phase[
        "ordinary_inactive"
    ]
    sample = _describe(outcome.loc[sample_flag])
    control = _describe(outcome.loc[control_flag])
    return {
        "index_id": index_id,
        "index_name": index_name,
        "signal_id": signal_id,
        "signal_label": signal_label,
        "industry_ids": ",".join(industry_ids),
        "industry_count": len(industry_ids),
        "phase": phase_name,
        "horizon_days": horizon,
        "hac_max_lag": max(5, horizon),
        "episode_count": _phase_event_count(phase, phase_name),
        "sample_n": sample["n"],
        "control_n": control["n"],
        "sample_mean_return": sample["mean"],
        "control_mean_return": control["mean"],
        "difference_mean_return": sample["mean"] - control["mean"],
        "sample_median_return": sample["median"],
        "control_median_return": control["median"],
        "difference_median_return": sample["median"] - control["median"],
        "sample_negative_rate": sample["negative_rate"],
        "control_negative_rate": control["negative_rate"],
        "difference_negative_rate": (
            sample["negative_rate"] - control["negative_rate"]
        ),
        "raw_beta": raw_result["coefficient"],
        "raw_standard_error": raw_result["standard_error"],
        "raw_z_value": raw_result["z_value"],
        "raw_p_value": raw_result["p_value"],
        "raw_nobs": int(raw_result["nobs"]),
        "controlled_beta": controlled_result["coefficient"],
        "controlled_standard_error": controlled_result["standard_error"],
        "controlled_z_value": controlled_result["z_value"],
        "controlled_p_value": controlled_result["p_value"],
        "controlled_nobs": int(controlled_result["nobs"]),
    }


def evaluate_signal_definitions(
    signal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    exhaustive: bool,
    horizons: tuple[int, ...] = HORIZONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate predefined signals and return tests plus collapsed episodes."""

    definitions = _signal_definitions(signal, exhaustive=exhaustive)
    episode_frames = [
        build_episode_table(
            signal["date"],
            build_phase_frame(definition["active"]),
            signal_id=str(definition["signal_id"]),
            signal_label=str(definition["signal_label"]),
            industry_ids=definition["industry_ids"],
        )
        for definition in definitions
    ]
    episodes = pd.concat(episode_frames, ignore_index=True)

    merged = outcomes[["index_id", "index_name", "date", "close"]].copy()
    merged["date"] = pd.to_datetime(merged["date"])
    signal_columns = ["date", "breadth_ma20"] + [
        "target_%s" % target_id for target_id in TARGET_IDS
    ]
    merged = merged.merge(signal[signal_columns], on="date", how="inner")
    records = []
    for (index_id, index_name), group in merged.groupby(
        ["index_id", "index_name"], sort=False
    ):
        group = group.sort_values("date").reset_index(drop=True)
        future_returns = compute_future_returns(group["close"], horizons)
        controls = _standardized_controls(group)
        for definition in definitions:
            active_columns = [
                "target_%s" % target_id
                for target_id in definition["industry_ids"]
            ]
            active = group[active_columns].apply(_as_bool).any(axis=1)
            phase = build_phase_frame(active)
            for horizon in horizons:
                outcome = future_returns["future_return_%dd" % horizon]
                max_lag = max(5, horizon)
                active_design = pd.DataFrame(
                    {"intercept": 1.0, "active": phase["active"].astype(float)}
                )
                phase_design = pd.DataFrame(
                    {
                        "intercept": 1.0,
                        "onset": phase["onset"].astype(float),
                        "continuation": phase["continuation"].astype(float),
                        "exit": phase["exit"].astype(float),
                    }
                )
                raw_active = newey_west_ols(
                    outcome, active_design, max_lag=max_lag
                )
                raw_phases = newey_west_ols(outcome, phase_design, max_lag=max_lag)
                controlled_active = newey_west_ols(
                    outcome,
                    pd.concat([active_design, controls], axis=1),
                    max_lag=max_lag,
                )
                controlled_phases = newey_west_ols(
                    outcome,
                    pd.concat([phase_design, controls], axis=1),
                    max_lag=max_lag,
                )
                for phase_name in ("active", "onset", "continuation", "exit"):
                    raw_result = (
                        raw_active.loc["active"]
                        if phase_name == "active"
                        else raw_phases.loc[phase_name]
                    )
                    controlled_result = (
                        controlled_active.loc["active"]
                        if phase_name == "active"
                        else controlled_phases.loc[phase_name]
                    )
                    records.append(
                        _test_record(
                            index_id=index_id,
                            index_name=index_name,
                            signal_id=str(definition["signal_id"]),
                            signal_label=str(definition["signal_label"]),
                            industry_ids=definition["industry_ids"],
                            phase_name=phase_name,
                            horizon=horizon,
                            phase=phase,
                            outcome=outcome,
                            raw_result=raw_result,
                            controlled_result=controlled_result,
                        )
                    )
    tests = pd.DataFrame.from_records(records)
    family_prefix = "exploratory_subsets" if exhaustive else "primary"
    tests = _add_inference_fields(
        tests, family_prefix=family_prefix, include_phase=True
    )
    return tests, episodes


def evaluate_duration_buckets(
    signal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Evaluate episode-day buckets for the union of all four industries."""

    signal_columns = ["date", "breadth_ma20"] + [
        "target_%s" % target_id for target_id in TARGET_IDS
    ]
    merged = outcomes[["index_id", "index_name", "date", "close"]].copy()
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.merge(signal[signal_columns], on="date", how="inner")
    records = []
    for (index_id, index_name), group in merged.groupby(
        ["index_id", "index_name"], sort=False
    ):
        group = group.sort_values("date").reset_index(drop=True)
        active = group[["target_%s" % target_id for target_id in TARGET_IDS]].apply(
            _as_bool
        ).any(axis=1)
        phase = build_phase_frame(active)
        controls = _standardized_controls(group)
        future_returns = compute_future_returns(group["close"], horizons)
        bucket_flags = {}
        for bucket_id, _, start_day, end_day in DURATION_BUCKETS:
            if end_day is None:
                bucket_flags[bucket_id] = phase["episode_day"].ge(start_day)
            else:
                bucket_flags[bucket_id] = phase["episode_day"].between(
                    start_day, end_day
                )
        raw_design = pd.DataFrame({"intercept": 1.0}, index=group.index)
        for bucket_id, flag in bucket_flags.items():
            raw_design[bucket_id] = flag.astype(float)
        raw_design["exit"] = phase["exit"].astype(float)
        for horizon in horizons:
            outcome = future_returns["future_return_%dd" % horizon]
            max_lag = max(5, horizon)
            raw_results = newey_west_ols(outcome, raw_design, max_lag=max_lag)
            controlled_results = newey_west_ols(
                outcome,
                pd.concat([raw_design, controls], axis=1),
                max_lag=max_lag,
            )
            control = _describe(outcome.loc[phase["ordinary_inactive"]])
            for bucket_id, bucket_label, _, _ in DURATION_BUCKETS:
                flag = bucket_flags[bucket_id]
                sample = _describe(outcome.loc[flag])
                episode_count = int(
                    phase.loc[flag, "episode_id"].dropna().nunique()
                )
                raw = raw_results.loc[bucket_id]
                controlled = controlled_results.loc[bucket_id]
                records.append(
                    {
                        "index_id": index_id,
                        "index_name": index_name,
                        "bucket_id": bucket_id,
                        "bucket_label": bucket_label,
                        "horizon_days": horizon,
                        "hac_max_lag": max_lag,
                        "episode_count": episode_count,
                        "sample_n": sample["n"],
                        "control_n": control["n"],
                        "sample_mean_return": sample["mean"],
                        "control_mean_return": control["mean"],
                        "difference_mean_return": sample["mean"] - control["mean"],
                        "sample_median_return": sample["median"],
                        "control_median_return": control["median"],
                        "difference_median_return": (
                            sample["median"] - control["median"]
                        ),
                        "sample_negative_rate": sample["negative_rate"],
                        "control_negative_rate": control["negative_rate"],
                        "difference_negative_rate": (
                            sample["negative_rate"] - control["negative_rate"]
                        ),
                        "raw_beta": raw["coefficient"],
                        "raw_standard_error": raw["standard_error"],
                        "raw_z_value": raw["z_value"],
                        "raw_p_value": raw["p_value"],
                        "raw_nobs": int(raw["nobs"]),
                        "controlled_beta": controlled["coefficient"],
                        "controlled_standard_error": controlled["standard_error"],
                        "controlled_z_value": controlled["z_value"],
                        "controlled_p_value": controlled["p_value"],
                        "controlled_nobs": int(controlled["nobs"]),
                    }
                )
    tests = pd.DataFrame.from_records(records)
    return _add_inference_fields(
        tests, family_prefix="duration", include_phase=False
    )


def _percent(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "—"
    return ("%%.%df%%%%" % digits) % (100.0 * float(value))


def _number(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "—"
    return ("%%.%df" % digits) % float(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def build_report(
    signal: pd.DataFrame,
    outcomes: pd.DataFrame,
    primary: pd.DataFrame,
    subsets: pd.DataFrame,
    durations: pd.DataFrame,
    episodes: pd.DataFrame,
) -> str:
    index_ids = set(outcomes["index_id"].astype(str))
    has_actual_target = ACTUAL_TARGET_INDEX_ID in index_ids
    display_index = (
        ACTUAL_TARGET_INDEX_ID
        if has_actual_target
        else PROXY_TARGET_INDEX_ID
        if PROXY_TARGET_INDEX_ID in index_ids
        else str(outcomes["index_id"].iloc[0])
    )
    display_name = outcomes.loc[
        outcomes["index_id"].eq(display_index), "index_name"
    ].iloc[0]
    selected_horizons = (1, 5, 10, 20)
    combined = primary[
        primary["index_id"].eq(display_index)
        & primary["signal_id"].eq("any_four")
        & primary["phase"].isin(["active", "onset"])
        & primary["horizon_days"].isin(selected_horizons)
    ].copy()
    combined["阶段"] = combined["phase"].map(
        {"active": "全部活跃日", "onset": "首次触发"}
    )
    combined_table = pd.DataFrame(
        {
            "阶段": combined["阶段"],
            "期限": combined["horizon_days"].map(lambda value: "%d日" % value),
            "样本": combined["sample_n"],
            "均值收益": combined["sample_mean_return"].map(_percent),
            "相对对照差": combined["difference_mean_return"].map(_percent),
            "HAC p": combined["raw_p_value"].map(_number),
            "FDR q": combined["raw_q_value"].map(_number),
            "控制后差": combined["controlled_beta"].map(_percent),
            "控制后q": combined["controlled_q_value"].map(_number),
        }
    )

    targets = primary[
        primary["index_id"].eq(display_index)
        & primary["signal_id"].str.startswith("target_")
        & primary["phase"].eq("active")
        & primary["horizon_days"].eq(20)
    ].copy()
    target_table = pd.DataFrame(
        {
            "行业": targets["signal_label"],
            "样本日": targets["sample_n"],
            "区间数": targets["episode_count"],
            "推断资格": targets["inference_eligible"].map(
                {True: "正式", False: "仅描述"}
            ),
            "20日收益差": targets["difference_mean_return"].map(_percent),
            "HAC p": targets["raw_p_value"].map(_number),
            "FDR q": targets["raw_q_value"].map(_number),
        }
    )

    duration_candidates = durations[
        durations["index_id"].eq(display_index)
        & durations["inference_eligible"]
        & durations["raw_q_value"].lt(0.05)
    ].copy()
    if duration_candidates.empty:
        duration_text = "- 连续天数的35项检验中，没有结果通过5% FDR。"
    else:
        duration_table = pd.DataFrame(
            {
                "持续阶段": duration_candidates["bucket_label"],
                "期限": duration_candidates["horizon_days"].map(
                    lambda value: "%d日" % value
                ),
                "样本日": duration_candidates["sample_n"],
                "区间数": duration_candidates["episode_count"],
                "收益差": duration_candidates["difference_mean_return"].map(
                    _percent
                ),
                "HAC p": duration_candidates["raw_p_value"].map(_number),
                "FDR q": duration_candidates["raw_q_value"].map(_number),
            }
        )
        duration_text = _markdown_table(duration_table)

    combined_episodes = episodes[episodes["signal_id"].eq("any_four")]
    raw_significant = int(primary["raw_q_value"].lt(0.05).sum())
    controlled_significant = int(primary["controlled_q_value"].lt(0.05).sum())
    global_raw_significant = int(primary["global_raw_q_value"].lt(0.05).sum())
    global_controlled_significant = int(
        primary["global_controlled_q_value"].lt(0.05).sum()
    )
    subset_significant = int(subsets["raw_q_value"].lt(0.05).sum())
    subset_controlled_significant = int(
        subsets["controlled_q_value"].lt(0.05).sum()
    )
    subset_global_significant = int(
        subsets["global_raw_q_value"].lt(0.05).sum()
    )
    subset_global_controlled_significant = int(
        subsets["global_controlled_q_value"].lt(0.05).sum()
    )
    local_candidates = primary[primary["raw_q_value"].lt(0.05)].copy()
    if local_candidates.empty:
        local_candidate_text = "- 没有项目通过局部5% FDR。"
    else:
        local_candidate_table = pd.DataFrame(
            {
                "指数": local_candidates["index_name"],
                "信号": local_candidates["signal_label"],
                "阶段": local_candidates["phase"].map(
                    {
                        "active": "全部活跃日",
                        "onset": "首次触发",
                        "continuation": "持续期",
                        "exit": "退出日",
                    }
                ),
                "期限": local_candidates["horizon_days"].map(
                    lambda value: "%d日" % value
                ),
                "收益差": local_candidates["difference_mean_return"].map(_percent),
                "局部q": local_candidates["raw_q_value"].map(_number),
                "全局q": local_candidates["global_raw_q_value"].map(_number),
            }
        )
        local_candidate_text = _markdown_table(local_candidate_table)
    target_note = (
        "- 输入包含真正的中证2000指数，主展示对象为中证2000。"
        if has_actual_target
        else (
            "- 输入不含中证2000指数 `932000`；主表使用国证2000 `399303` 作为代理，不能解释为对 `563300` 的直接验证。"
            if display_index == PROXY_TARGET_INDEX_ID
            else "- 输入不含中证2000指数 `932000`；主表使用 `%s`，仅用于验证分析流程。"
            % display_name
        )
    )
    lines = [
        "# 四行业 Top1 后续收益研究",
        "",
        "## 数据与边界",
        "",
        "- 信号样本为 %s—%s，共 %d 个交易日。"
        % (
            signal["date"].min().date(),
            signal["date"].max().date(),
            len(signal),
        ),
        "- 使用点时全A、申万一级行业、收盘价高于MA20的行业宽度Top1信号。",
        target_note,
        "- 收益定义为 `close(t+N) / close(t) - 1`。信号依赖当日完整收盘数据，因此这是信息含量研究，不等同于可在当日收盘成交的策略收益。",
        "",
        "## 连续信号处理",
        "",
        "- 四行业任一触发形成 %d 个连续区间；中位长度 %.1f 日，最长 %d 日。"
        % (
            len(combined_episodes),
            combined_episodes["trading_days"].median(),
            combined_episodes["trading_days"].max(),
        ),
        "- 每个信号分别重建 `active / onset / continuation / exit`，首次触发每个连续区间只计一次。",
        "- 重叠的未来N日收益使用Bartlett核Newey-West协方差，最大滞后为 `max(5, N)`。",
        "- 少于%d个独立连续区间或少于%d个有效收益观测的项目只保留描述统计，不进入FDR或显著项计数。"
        % (MIN_INFERENCE_EPISODES, MIN_INFERENCE_SAMPLE),
        "- 控制模型加入全市场宽度及平方项、1/5日宽度变化、指数5/20日动量、20日波动率和年份固定效应。",
        "",
        "## %s：四行业合并" % display_name,
        "",
        _markdown_table(combined_table),
        "",
        "## %s：单行业" % display_name,
        "",
        _markdown_table(target_table),
        "",
        "## 持续天数探索",
        "",
        duration_text,
        "",
        "## 多重比较",
        "",
        f"- 主检验族按“指数 × 阶段”分别校正，每族最多包含5个预定义行业信号和7个期限，共35项；局部FDR下原始模型有{raw_significant}项、控制模型有{controlled_significant}项。把全部指数和阶段作为一个探索族再次校正后，分别剩{global_raw_significant}项和{global_controlled_significant}项。",
        f"- 15种行业子集属于探索检验，按“指数 × 阶段”分别最多对105项校正；局部FDR下原始模型有{subset_significant}项、控制模型有{subset_controlled_significant}项，全局校正后分别剩{subset_global_significant}项和{subset_global_controlled_significant}项。",
        "- 七个指数高度相关，不能把跨指数同方向视为独立重复验证。",
        "",
        "### 局部FDR候选",
        "",
        local_candidate_text,
        "",
        "## 使用建议",
        "",
        "- 优先看首次触发与持续期是否方向一致；若只有持续若干日后出现差异，应把持续阈值登记为新假设并用新样本复核。",
        "- 单行业方向不一致时，不把它们无差别合并为一个风险开关。",
        "- 若要评估可交易性，下一层应补充中证2000指数或ETF真实行情，并计算次日开盘至未来收盘、手续费、价差和冲击成本。",
    ]
    return "\n".join(lines) + "\n"


def analyze_frames(
    signal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, object]:
    """Run the complete in-memory analysis for tested callers and the CLI."""

    signal = signal.copy()
    signal["date"] = pd.to_datetime(signal["date"])
    outcomes = outcomes.copy()
    outcomes["date"] = pd.to_datetime(outcomes["date"])
    required_signal = {"date", "breadth_ma20"} | {
        "target_%s" % target_id for target_id in TARGET_IDS
    }
    missing_signal = required_signal - set(signal.columns)
    if missing_signal:
        raise ValueError("missing signal columns: %s" % sorted(missing_signal))
    required_outcomes = {"index_id", "index_name", "date", "close"}
    missing_outcomes = required_outcomes - set(outcomes.columns)
    if missing_outcomes:
        raise ValueError("missing outcome columns: %s" % sorted(missing_outcomes))
    if signal["date"].duplicated().any() or not signal["date"].is_monotonic_increasing:
        raise ValueError("signal dates must be unique and increasing")
    if outcomes.duplicated(["index_id", "date"]).any():
        raise ValueError("outcome dates must be unique within each index")

    primary, primary_episodes = evaluate_signal_definitions(
        signal, outcomes, exhaustive=False, horizons=horizons
    )
    subsets, subset_episodes = evaluate_signal_definitions(
        signal, outcomes, exhaustive=True, horizons=horizons
    )
    durations = evaluate_duration_buckets(signal, outcomes, horizons=horizons)
    report = build_report(
        signal,
        outcomes,
        primary,
        subsets,
        durations,
        primary_episodes,
    )
    return {
        "primary": primary,
        "subsets": subsets,
        "durations": durations,
        "primary_episodes": primary_episodes,
        "subset_episodes": subset_episodes,
        "report": report,
    }


def run_analysis(
    package_dir: Path | str = DEFAULT_PACKAGE_DIR,
    outcomes_path: Path | str = DEFAULT_OUTCOMES_PATH,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
) -> dict[str, Path]:
    """Validate inputs, run the study and write deterministic artifacts."""

    package_dir = Path(package_dir)
    outcomes_path = Path(outcomes_path)
    output_dir = Path(output_dir)
    daily, _, _ = load_and_validate_package(package_dir)
    signal = prepare_valid_signal(daily)
    outcomes = pd.read_csv(outcomes_path)
    result = analyze_frames(signal, outcomes)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "primary": output_dir / "primary_tests.csv",
        "subsets": output_dir / "subset_tests.csv",
        "durations": output_dir / "duration_tests.csv",
        "primary_episodes": output_dir / "primary_episodes.csv",
        "subset_episodes": output_dir / "subset_episodes.csv",
        "report": output_dir / "report.md",
        "manifest": output_dir / "manifest.json",
    }
    for key in (
        "primary",
        "subsets",
        "durations",
        "primary_episodes",
        "subset_episodes",
    ):
        result[key].to_csv(outputs[key], index=False)
    outputs["report"].write_text(str(result["report"]), encoding="utf-8")
    manifest = {
        "analysis_id": "four_industry_forward_returns_v1",
        "horizons": list(HORIZONS),
        "actual_target_index_id": ACTUAL_TARGET_INDEX_ID,
        "proxy_target_index_id": PROXY_TARGET_INDEX_ID,
        "actual_target_present": ACTUAL_TARGET_INDEX_ID
        in set(outcomes["index_id"].astype(str)),
        "signal_start": str(signal["date"].min().date()),
        "signal_end": str(signal["date"].max().date()),
        "signal_days": len(signal),
        "input": {
            "package_manifest": str(package_dir / "manifest.json"),
            "package_manifest_sha256": _sha256(package_dir / "manifest.json"),
            "outcomes": str(outcomes_path),
            "outcomes_sha256": _sha256(outcomes_path),
        },
        "inference": {
            "covariance": "Newey-West Bartlett kernel",
            "max_lag": "max(5, horizon_days)",
            "minimum_episode_count": MIN_INFERENCE_EPISODES,
            "minimum_sample_count": MIN_INFERENCE_SAMPLE,
            "primary_fdr_family": "index_id x phase; 5 signals x 7 horizons",
            "subset_fdr_family": "index_id x phase; 15 subsets x 7 horizons",
            "duration_fdr_family": "index_id; 5 buckets x 7 horizons",
            "global_fdr": "all inference-eligible rows in each output table",
        },
        "outputs": {},
    }
    for key, path in outputs.items():
        if key == "manifest":
            continue
        manifest["outputs"][key] = {
            "path": path.name,
            "sha256": _sha256(path),
            "rows": (
                len(result[key]) if isinstance(result.get(key), pd.DataFrame) else None
            ),
        }
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    outputs = run_analysis(args.package_dir, args.outcomes, args.output_dir)
    for name, path in outputs.items():
        print("%s: %s" % (name, path))


if __name__ == "__main__":
    main()
