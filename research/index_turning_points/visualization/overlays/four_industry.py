"""Overlay the archived four-industry Top1 signal on the turning-point viewer."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from ..viewer import ViewerPage

SIGNAL_COLOR = "#B7791F"
INDUSTRY_SPECS = (
    ("bank", "银行", "#456A9E"),
    ("coal", "煤炭", "#555C65"),
    ("nonferrous", "有色", "#B7791F"),
    ("steel", "钢铁", "#785A7C"),
)
TARGET_COLUMNS = tuple("target_%s" % item[0] for item in INDUSTRY_SPECS)

FOUR_INDUSTRY_PAGE = ViewerPage(
    browser_title="指数顶底区域与四行业宽度信号",
    heading="指数顶底区域与四行业宽度信号",
    subtitle_suffix=" · 四行业 Top1 V1",
    extra_css=(
        ".phase-swatch.is-signal { background: %s; opacity: .72; }" % SIGNAL_COLOR
    ),
    extra_legend_html=(
        '<span class="phase-key" role="listitem"><i class="phase-swatch '
        'is-signal" aria-hidden="true"></i>四行业 Top1</span>'
    ),
)


def _align_signal(daily: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    start = signal["date"].min()
    end = signal["date"].max()
    in_signal_period = daily.index[(daily.index >= start) & (daily.index <= end)]
    missing = in_signal_period.difference(pd.DatetimeIndex(signal["date"]))
    if len(missing):
        raise ValueError(
            "four-industry signal is missing index trading dates: %s"
            % missing[:5].strftime("%Y-%m-%d").tolist()
        )

    aligned = signal.set_index("date").reindex(daily.index)
    for column in ("triggered", "onset", "continuation") + TARGET_COLUMNS:
        aligned[column] = aligned[column].eq(True)
    return aligned


def _industry_names(row: pd.Series) -> str:
    names = [
        name
        for industry_id, name, _ in INDUSTRY_SPECS
        if bool(row["target_%s" % industry_id])
    ]
    return "、".join(names)


def add_four_industry_signal(
    figure: go.Figure,
    daily: pd.DataFrame,
    signal: pd.DataFrame,
) -> go.Figure:
    """Add a scale-independent signal ribbon and per-industry onset markers."""

    aligned = _align_signal(daily, signal)
    active = aligned["triggered"]
    active_customdata = [
        (
            [
                int(row.episode_id),
                int(row.episode_day),
                _industry_names(row),
                float(row.breadth_ma20),
                int(row.top1_tie_count_ma20),
                "首次触发" if bool(row.onset) else "持续",
            ]
            if bool(row.triggered)
            else ["", "", "", None, "", ""]
        )
        for _, row in aligned.iterrows()
    ]

    figure.add_trace(
        go.Scatter(
            x=aligned.index,
            y=[0.965 if value else None for value in active],
            yaxis="y2",
            mode="lines+markers",
            connectgaps=False,
            name="四行业 Top1 活跃期",
            legendrank=30,
            line=dict(color="rgba(183,121,31,0.34)", width=8, shape="hv"),
            marker=dict(color=SIGNAL_COLOR, size=4, opacity=0.68),
            customdata=active_customdata,
            hovertemplate=(
                "四行业 Top1 · %{customdata[5]}"
                "<br>%{x|%Y-%m-%d} · 区间 %{customdata[0]} 第 %{customdata[1]} 日"
                "<br>Top1 行业 %{customdata[2]}"
                "<br>全A MA20宽度 %{customdata[3]:.1%}"
                "<br>全市场并列 Top1 数 %{customdata[4]}<extra></extra>"
            ),
        )
    )

    for industry_id, industry_name, color in INDUSTRY_SPECS:
        selected = aligned[aligned["onset"] & aligned["target_%s" % industry_id]]
        customdata = [
            [
                int(row.episode_id),
                float(row.breadth_ma20),
                int(row.top1_tie_count_ma20),
            ]
            for row in selected.itertuples(index=False)
        ]
        figure.add_trace(
            go.Scatter(
                x=selected.index,
                y=[0.965] * len(selected),
                yaxis="y2",
                mode="markers",
                name="首次触发 · %s" % industry_name,
                legendrank=31,
                marker=dict(
                    symbol="diamond",
                    size=10,
                    color=color,
                    line=dict(color="#FFFFFF", width=1),
                ),
                customdata=customdata,
                hovertemplate=(
                    "首次触发 · %s"
                    "<br>日期 %%{x|%%Y-%%m-%%d} · 区间 %%{customdata[0]}"
                    "<br>全A MA20宽度 %%{customdata[1]:.1%%}"
                    "<br>全市场并列 Top1 数 %%{customdata[2]}<extra></extra>"
                )
                % industry_name,
            )
        )

    figure.update_layout(
        yaxis2=dict(
            overlaying="y",
            side="right",
            range=[0.0, 1.0],
            fixedrange=True,
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            visible=False,
        )
    )
    return figure
