"""Deterministic Markdown reports for the two independent evaluations."""

from __future__ import annotations

import math

import pandas as pd


REGION_REPORT_SECTION_ORDER = (
    "口径",
    "跨指数总览",
    "各指数 window 口径",
    "明细状态计数",
    "产物索引",
    "分组发现与注意事项",
)
FORWARD_REPORT_SECTION_ORDER = (
    "口径",
    "可用性",
    "描述统计与推断",
    "产物索引",
    "分组发现与注意事项",
)


def render_region_report(
    metrics: pd.DataFrame,
    matches: pd.DataFrame,
    *,
    evaluation_version: str,
    label_version: str,
) -> str:
    """Render the region-location report without inventing a composite score."""

    overview = metrics[
        metrics["aggregation"].eq("all_indices")
        & metrics["timing_slice"].eq("all")
        & metrics["region_form_slice"].eq("all")
    ].copy()
    overview = overview[
        [
            "signal_id",
            "direction",
            "version",
            "event_kind",
            "match_scope",
            "region_count",
            "matched_region_count",
            "region_recall",
            "episode_count",
            "matched_episode_count",
            "episode_precision",
            "false_alarm_count",
            "duplicate_alarm_count",
            "median_lead_lag_days",
        ]
    ]
    per_index = metrics[
        metrics["aggregation"].eq("index")
        & metrics["match_scope"].eq("window")
        & metrics["timing_slice"].eq("all")
        & metrics["region_form_slice"].eq("all")
    ].copy()
    per_index = per_index[
        [
            "signal_id",
            "direction",
            "event_kind",
            "index_name",
            "region_count",
            "region_recall",
            "episode_count",
            "episode_precision",
            "false_alarm_count",
            "duplicate_alarm_count",
            "median_lead_lag_days",
        ]
    ]
    status_counts = (
        matches.groupby("match_status", dropna=False).size().rename("rows").reset_index()
    )
    findings = collect_region_findings(metrics, matches)
    return f"""# 顶底区域定位评测

- 评测版本：`{evaluation_version}`
- 区域标签：`{label_version}`
- 本报告只评价区域定位，不与信号后价格结果合成总分。

## 口径

- 顶部信号只与顶部区域匹配，底部信号只与底部区域匹配；每个 episode 和区域最多形成一个主匹配。
- `strict` 只认核心峰瓣，`loose` 认核心峰瓣或连续包络，`window` 再纳入冻结的 5/10/20 日窗口。
- `lead_lag_days <= 0` 为预测，正值为确认。预测/确认召回率只使用对应 20 日窗口完整的区域作分母。
- 单峰/多峰由标准答案的 `lobe_count` 决定。误报为便于切片，会关联覆盖期内最近的同向区域；该诊断关联不会改变误报状态或主匹配。
- `all_indices` 以“指数×区域”和“指数×episode”对汇总；指数间相关，不能把这些对当成独立统计样本。

## 跨指数总览

{_markdown_table(overview)}

## 各指数 window 口径

{_markdown_table(per_index)}

## 明细状态计数

{_markdown_table(status_counts)}

## 产物索引

完整的预测/确认、单峰/多峰、strict/loose/window 与指数/汇总笛卡尔切片见 `region_metrics.csv`；逐条主匹配、重复报警、误报和漏检见 `region_matches.csv`。

## 分组发现与注意事项

{_findings_markdown(findings)}
"""


def render_forward_report(
    metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    evaluation_version: str,
    min_event_count: int,
    min_baseline_count: int,
) -> str:
    """Render event-path descriptions and inference assumptions."""

    table = metrics[
        [
            "signal_id",
            "direction",
            "event_kind",
            "index_name",
            "horizon",
            "outcome_name",
            "event_count",
            "event_mean",
            "baseline_count",
            "baseline_mean",
            "mean_difference",
            "ci95_lower",
            "ci95_upper",
            "hac_p_value",
            "local_fdr_q_value",
            "global_fdr_q_value",
            "inference_eligible",
        ]
    ]
    availability = (
        outcomes.groupby(
            ["index_name", "event_kind", "horizon"], dropna=False
        )
        .agg(
            events=("episode_id", "size"),
            event_dates_available=("event_date_available", "sum"),
            complete_windows=("complete_window", "sum"),
        )
        .reset_index()
    )
    findings = collect_forward_findings(metrics, outcomes)
    return f"""# 信号后 OHLC 结果评测

- 评测版本：`{evaluation_version}`
- 本报告与区域定位报告相互独立，不生成综合总分，也不构成交易回测。

## 口径

对事件日 `t` 和未来 `h` 个指数交易日：

```text
terminal_return_h = close[t+h] / close[t] - 1
max_up_h          = max(high[t+1:t+h]) / close[t] - 1
max_down_h        = min(low[t+1:t+h]) / close[t] - 1
```

- 分别评价 onset 与 capped confirmation 的 5/10/20 日路径；事件日缺失和尾部窗口不完整会保留在明细中，但不进入统计。
- 基线是同一信号覆盖期、同一指数、同一期限的完整非事件日。
- 均值差使用 `结果 ~ 常数 + 事件指示变量`，Newey–West 滞后阶数等于期限；95% 区间使用正态近似。
- 推断至少需要 {min_event_count} 个完整事件和 {min_baseline_count} 个完整基线日。样本不足时保留分布描述，显著性和 FDR 留空。
- 局部 FDR 家族为同一 signal/direction/version/event kind 下的全部指数、期限和结果；全局 FDR 覆盖 bundle 中全部合格检验。
- `close[t]` 只是统一参考价，不代表信号能在该收盘价成交；结果不能解释成含成本交易收益。

## 可用性

{_markdown_table(availability)}

## 描述统计与推断

{_markdown_table(table)}

## 产物索引

逐事件、逐指数、逐期限的完整路径见 `forward_event_outcomes.csv`，包括事件日可用性、未来窗口完整性和窗口终止日。

## 分组发现与注意事项

{_findings_markdown(findings)}
"""


def collect_region_findings(
    metrics: pd.DataFrame,
    matches: pd.DataFrame,
) -> list[str]:
    """Return deterministic, group-labelled notes appended after the report."""

    findings: list[str] = []
    keys = ["signal_id", "direction", "version", "event_kind"]
    for key, group in metrics.groupby(keys, sort=True):
        notes: list[str] = []
        label = _group_label(key)
        match_group = _select_group(matches, keys, key)
        core = group[
            group["aggregation"].eq("all_indices")
            & group["match_scope"].eq("window")
            & group["timing_slice"].eq("all")
            & group["region_form_slice"].eq("all")
        ]
        if core.empty:
            notes.append("缺少跨指数 window 总览行，需检查指标完整性。")
        else:
            row = core.iloc[0]
            if int(row["episode_count"]) == 0:
                notes.append("覆盖期内没有可评估 episode。")
            if int(row["region_count"]) == 0:
                notes.append("覆盖期内没有同向标准答案区域。")
            recall = float(row["region_recall"])
            precision = float(row["episode_precision"])
            if recall >= 0.8 and precision < 0.2:
                notes.append(
                    "window 汇总呈现高区域召回但低 episode 精确率："
                    f"{recall:.1%}/{precision:.1%}；需同时关注报警密度，"
                    "不能只读取召回率。"
                )

        region_rows = match_group[
            match_group["match_status"].isin(["matched", "missed_region"])
        ]
        prediction_incomplete = int(
            (~_report_bool(region_rows["prediction_window_complete"])).sum()
        )
        confirmation_incomplete = int(
            (~_report_bool(region_rows["confirmation_window_complete"])).sum()
        )
        if prediction_incomplete or confirmation_incomplete:
            notes.append(
                "区域窗口不完整："
                f"预测 {prediction_incomplete} 个、确认 {confirmation_incomplete} 个；"
                "对应时点召回切片已从分母排除。"
            )
        duplicate_count = int(
            match_group["match_status"].eq("duplicate_alarm").sum()
        )
        if duplicate_count:
            notes.append(
                f"存在 {duplicate_count} 个指数×episode 重复报警，未计为主匹配。"
            )
        alarm_rows = match_group[
            match_group["match_status"].isin(
                ["matched", "duplicate_alarm", "false_alarm"]
            )
        ]
        unclassified = int(alarm_rows["diagnostic_timing"].fillna("").eq("").sum())
        if unclassified:
            notes.append(
                f"有 {unclassified} 个指数×episode 因覆盖期内无同向区域，"
                "无法进入时点/形态切片。"
            )

        scope_rows = group[
            group["aggregation"].eq("all_indices")
            & group["timing_slice"].eq("all")
            & group["region_form_slice"].eq("all")
        ].set_index("match_scope")
        if set(["strict", "loose", "window"]).issubset(scope_rows.index):
            counts = [
                int(scope_rows.loc[scope, "matched_region_count"])
                for scope in ("strict", "loose", "window")
            ]
            if len(set(counts)) > 1:
                notes.append(
                    "区域命中对口径敏感：strict/loose/window 分别为 "
                    f"{counts[0]}/{counts[1]}/{counts[2]}。"
                )
        if notes:
            findings.append(f"`{label}`：" + " ".join(notes))
    return findings


def collect_forward_findings(
    metrics: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> list[str]:
    """Return availability and inference notes for each forward-test group."""

    findings: list[str] = []
    keys = ["signal_id", "direction", "version", "event_kind"]
    for key, group in metrics.groupby(keys, sort=True):
        notes: list[str] = []
        label = _group_label(key)
        outcome_group = _select_group(outcomes, keys, key)
        availability_notes = []
        for horizon, detail in outcome_group.groupby("horizon", sort=True):
            unavailable_mask = ~_report_bool(detail["event_date_available"])
            incomplete_mask = ~_report_bool(detail["complete_window"])
            unavailable = int(unavailable_mask.sum())
            incomplete = int(incomplete_mask.sum())
            if unavailable or incomplete:
                affected_indices = int(
                    detail.loc[unavailable_mask | incomplete_mask, "index_name"].nunique()
                )
                availability_notes.append(
                    f"{int(horizon)}日：事件日缺失 {unavailable}、窗口不完整 "
                    f"{incomplete}（涉及 {affected_indices} 个指数）"
                )
        if availability_notes:
            notes.append("数据可用性——" + "；".join(availability_notes) + "。")

        eligible = _report_bool(group["inference_eligible"])
        ineligible_count = int((~eligible).sum())
        if ineligible_count:
            notes.append(
                f"{ineligible_count}/{len(group)} 项检验未达到样本门槛，"
                "仅可读取描述统计。"
            )
        p_values = pd.to_numeric(group["hac_p_value"], errors="coerce")
        global_q = pd.to_numeric(group["global_fdr_q_value"], errors="coerce")
        eligible_p = p_values[eligible & p_values.notna()]
        global_significant = int((eligible & global_q.lt(0.05)).sum())
        nominal_only = int((p_values.lt(0.05) & global_q.ge(0.05)).sum())
        if len(eligible_p) and not eligible_p.lt(0.05).any():
            notes.append(
                f"{len(eligible_p)} 项合格检验均未达到名义 p<0.05，"
                "因此也没有全局 FDR 发现。"
            )
        elif nominal_only:
            notes.append(
                f"{nominal_only} 项仅名义 p<0.05、未通过全局 FDR，不能按显著结果解读。"
            )
        if global_significant:
            notes.append(f"有 {global_significant} 项检验通过全局 FDR。")

        longest_horizon = int(group["horizon"].max())
        terminal = group[
            group["horizon"].eq(longest_horizon)
            & group["outcome_name"].eq("terminal_return")
            & eligible
        ]
        differences = pd.to_numeric(terminal["mean_difference"], errors="coerce").dropna()
        direction = str(key[1])
        expected_sign = differences.lt(0) if direction == "top" else differences.gt(0)
        if len(differences) >= 2 and expected_sign.all():
            sign_text = "负" if direction == "top" else "正"
            notes.append(
                f"最长 {longest_horizon} 日 terminal 均值差在 "
                f"{len(differences)}/{len(differences)} 个指数均为{sign_text}；"
                "这是跨指数方向一致的描述性现象，显著性仍以 HAC/FDR 为准。"
            )
        if notes:
            findings.append(f"`{label}`：" + " ".join(notes))
    return findings


def _select_group(
    frame: pd.DataFrame,
    columns: list[str],
    values: tuple[object, ...],
) -> pd.DataFrame:
    selected = pd.Series(True, index=frame.index)
    for column, value in zip(columns, values):
        selected &= frame[column].astype(str).eq(str(value))
    return frame[selected]


def _group_label(values: tuple[object, ...]) -> str:
    return "/".join(str(value) for value in values)


def _report_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin(["true", "1"])


def _findings_markdown(findings: list[str]) -> str:
    if not findings:
        return "- 各测试组未发现超出主体表格口径的额外注意事项。"
    return "\n".join(f"- {finding}" for finding in findings)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_无记录_"
    columns = [str(column) for column in frame.columns]
    rows = [[_format_value(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    header = "| " + " | ".join(_escape(value) for value in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_escape(value) for value in row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return str(value)


def _escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
