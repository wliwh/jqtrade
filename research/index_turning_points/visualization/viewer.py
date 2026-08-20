"""One-page Plotly viewer for index OHLC and turning-point labels."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from ..adapters.tdx import INDEX_SPECS, read_tdx_daily, threshold_for_index
from ..ground_truth.labels import directional_change_labels
from ..ground_truth.regions import build_turning_point_regions


COLORS = {
    "background": "#F4F4F1",
    "paper": "#FFFFFF",
    "ink": "#202327",
    "muted": "#6B7078",
    "line": "#E1E2DE",
    "up": "#C84B47",
    "down": "#2A8A68",
    "top": "#A4322E",
    "bottom": "#167153",
    "initial": "#737982",
    "pending": "#C58B22",
}

PHASE_OPACITY = {
    "up": 0.055,
    "down": 0.055,
    "pending": 0.09,
}
REGION_ENVELOPE_OPACITY = 0.025
REGION_LOBE_OPACITY = 0.11


@dataclass(frozen=True)
class ViewerPage:
    """Small presentation interface; Plotly and HTML details stay internal."""

    browser_title: str = "指数顶底区域检查"
    heading: str = "指数顶底区域检查"
    subtitle_suffix: str = ""
    extra_css: str = ""
    extra_legend_html: str = ""


@dataclass
class ViewerPanel:
    """One index figure plus the aligned OHLC needed by explicit overlays."""

    index_id: str
    index_name: str
    daily: pd.DataFrame
    figure: go.Figure


def make_figure(
    daily: pd.DataFrame,
    index_name: str,
    threshold: float = 0.10,
    index_id: str = "visual",
) -> go.Figure:
    """Build one OHLC chart with causal phases and post-hoc regions."""

    labels = directional_change_labels(daily["high"], daily["low"], threshold)
    small_labels = directional_change_labels(
        daily["high"],
        daily["low"],
        threshold / 2.0,
    )
    regions, lobes = build_turning_point_regions(
        daily,
        labels,
        index_id=index_id,
        index_name=index_name,
        small_labels=small_labels,
    )
    confirmed = labels[labels["status"] == "confirmed"]
    initial = confirmed[~confirmed["eligible"]]
    pending = labels[labels["status"] == "unconfirmed"]
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    non_trading_dates = calendar.difference(daily.index)

    figure = go.Figure(
        go.Candlestick(
            x=daily.index,
            open=daily["open"],
            high=daily["high"],
            low=daily["low"],
            close=daily["close"],
            name="日K",
            increasing_line_color=COLORS["up"],
            increasing_fillcolor=COLORS["up"],
            decreasing_line_color=COLORS["down"],
            decreasing_fillcolor=COLORS["down"],
            hoverlabel=dict(namelength=0),
        )
    )

    figure.update_layout(
        shapes=(
            _phase_background_shapes(labels, daily.index[-1])
            + _region_background_shapes(daily, regions, lobes)
        )
    )
    _add_region_lobe_markers(figure, regions, lobes, "top")
    _add_region_lobe_markers(figure, regions, lobes, "bottom")
    _add_initial_marker(figure, daily, initial)
    _add_pending_marker(figure, daily, pending)

    figure.update_layout(
        title=dict(
            text=f"{index_name} · {threshold:.1%} 方向变化区域",
            x=0.01,
            xanchor="left",
            font=dict(size=20, color=COLORS["ink"]),
        ),
        template="plotly_white",
        autosize=True,
        height=540,
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["paper"],
        font=dict(
            family="Noto Sans CJK SC, Source Han Sans SC, sans-serif",
            color=COLORS["ink"],
            size=13,
        ),
        margin=dict(l=58, r=24, t=82, b=52),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=COLORS["ink"]),
        legend=dict(
            orientation="h",
            x=0.01,
            y=1.02,
            xanchor="left",
            yanchor="bottom",
            bgcolor="rgba(255,255,255,0)",
            font=dict(size=12),
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            rangebreaks=[dict(values=non_trading_dates.strftime("%Y-%m-%d").tolist())],
            rangeslider=dict(visible=True, thickness=0.07),
            rangeselector=dict(
                x=1,
                xanchor="right",
                y=1.12,
                yanchor="top",
                bgcolor=COLORS["background"],
                activecolor=COLORS["line"],
                buttons=[
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(count=3, label="3年", step="year", stepmode="backward"),
                    dict(count=5, label="5年", step="year", stepmode="backward"),
                    dict(step="all", label="全部"),
                ],
            ),
        ),
        yaxis=dict(
            title=None,
            fixedrange=False,
            gridcolor=COLORS["line"],
            gridwidth=0.7,
            zeroline=False,
        ),
    )
    return figure


def _phase_intervals(
    labels: pd.DataFrame,
    sample_end: pd.Timestamp,
) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Return mutually exclusive pending/up/down intervals."""

    if labels.empty:
        return []

    ordered = labels.sort_values("anchor_date").reset_index(drop=True)
    intervals = []
    for position, row in ordered.iterrows():
        anchor_date = pd.Timestamp(row["anchor_date"])
        if row["status"] != "confirmed":
            if anchor_date <= sample_end:
                intervals.append(("pending", anchor_date, sample_end))
            continue

        confirmation_date = min(pd.Timestamp(row["confirmation_date"]), sample_end)
        if anchor_date <= confirmation_date:
            intervals.append(("pending", anchor_date, confirmation_date))

        next_anchor = sample_end
        if position + 1 < len(ordered):
            next_anchor = min(
                pd.Timestamp(ordered.iloc[position + 1]["anchor_date"]),
                sample_end,
            )
        if confirmation_date <= next_anchor:
            state = "up" if row["event_type"] == "bottom" else "down"
            intervals.append((state, confirmation_date, next_anchor))
    return intervals


def _phase_background_shapes(
    labels: pd.DataFrame,
    sample_end: pd.Timestamp,
) -> list[dict[str, object]]:
    phase_colors = {
        "up": COLORS["up"],
        "down": COLORS["down"],
        "pending": COLORS["pending"],
    }
    return [
        _vrect(
            start,
            end,
            fillcolor=phase_colors[state],
            opacity=PHASE_OPACITY[state],
            line={"width": 0},
            name=f"phase:{state}",
        )
        for state, start, end in _phase_intervals(
            labels,
            pd.Timestamp(sample_end),
        )
    ]


def _region_background_shapes(
    daily: pd.DataFrame,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
) -> list[dict[str, object]]:
    shapes = []
    for row in regions.itertuples(index=False):
        color = COLORS[row.event_type]
        x0, x1 = _span_bounds(
            daily.index,
            int(row.region_start_position),
            int(row.region_end_position),
        )
        shapes.append(
            _vrect(
                x0,
                x1,
                fillcolor=color,
                opacity=REGION_ENVELOPE_OPACITY,
                line={"color": color, "width": 0.7, "dash": "dot"},
                name=f"region-envelope:{row.event_type}:{row.region_id}",
            )
        )

    for row in lobes.itertuples(index=False):
        color = COLORS[row.event_type]
        x0, x1 = _span_bounds(
            daily.index,
            int(row.lobe_start_position),
            int(row.lobe_end_position),
        )
        shapes.append(
            _vrect(
                x0,
                x1,
                fillcolor=color,
                opacity=REGION_LOBE_OPACITY,
                line={"width": 0},
                name=f"region-lobe:{row.event_type}:{row.lobe_id}",
            )
        )
    return shapes


def _vrect(
    x0: pd.Timestamp,
    x1: pd.Timestamp,
    *,
    fillcolor: str,
    opacity: float,
    line: dict[str, object],
    name: str,
) -> dict[str, object]:
    return {
        "type": "rect",
        "xref": "x",
        "yref": "y domain",
        "x0": x0,
        "x1": x1,
        "y0": 0,
        "y1": 1,
        "fillcolor": fillcolor,
        "opacity": opacity,
        "line": line,
        "layer": "below",
        "name": name,
    }


def _span_bounds(
    dates: pd.Index,
    start_position: int,
    end_position: int,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(dates[start_position])
    end = pd.Timestamp(dates[end_position])
    if start_position > 0:
        previous = pd.Timestamp(dates[start_position - 1])
        start = previous + (start - previous) / 2
    else:
        start -= pd.Timedelta(hours=12)
    if end_position + 1 < len(dates):
        following = pd.Timestamp(dates[end_position + 1])
        end += (following - end) / 2
    else:
        end += pd.Timedelta(hours=12)
    return start, end


def _add_region_lobe_markers(
    figure: go.Figure,
    regions: pd.DataFrame,
    lobes: pd.DataFrame,
    event_type: str,
) -> None:
    selected_lobes = lobes[lobes["event_type"] == event_type]
    if selected_lobes.empty:
        return

    is_top = event_type == "top"
    region_fields = regions[
        [
            "region_id",
            "region_start",
            "region_end",
            "anchor_date",
            "anchor_position",
            "anchor_price",
            "lobe_count",
            "confirmation_date",
            "confirmation_lag",
            "price_band_pct",
        ]
    ]
    selected = selected_lobes.merge(
        region_fields,
        on="region_id",
        how="left",
        validate="many_to_one",
    ).sort_values(["representative_position", "lobe_number"])
    anchor_lobe = selected["anchor_position"].between(
        selected["lobe_start_position"],
        selected["lobe_end_position"],
    )
    dates = pd.DatetimeIndex(selected["representative_date"])
    visual_price = selected["representative_price"].astype(float).to_numpy()
    visual_price *= 1.012 if is_top else 0.988
    customdata = [
        [
            f"{int(row.lobe_number)}/{int(row.lobe_count)}",
            "主峰瓣" if bool(is_anchor) else "次峰瓣",
            pd.Timestamp(row.lobe_start).strftime("%Y-%m-%d"),
            pd.Timestamp(row.lobe_end).strftime("%Y-%m-%d"),
            float(row.representative_price),
            pd.Timestamp(row.anchor_date).strftime("%Y-%m-%d"),
            float(row.anchor_price),
            pd.Timestamp(row.region_start).strftime("%Y-%m-%d"),
            pd.Timestamp(row.region_end).strftime("%Y-%m-%d"),
            pd.Timestamp(row.confirmation_date).strftime("%Y-%m-%d"),
            int(row.confirmation_lag),
            float(row.price_band_pct),
        ]
        for row, is_anchor in zip(selected.itertuples(index=False), anchor_lobe)
    ]

    figure.add_trace(
        go.Scatter(
            x=dates,
            y=visual_price,
            mode="markers",
            name="顶部区域峰瓣" if is_top else "底部区域峰瓣",
            marker=dict(
                symbol="triangle-down" if is_top else "triangle-up",
                size=[12 if value else 9 for value in anchor_lobe],
                color=COLORS["top" if is_top else "bottom"],
                line=dict(color="#FFFFFF", width=1),
            ),
            opacity=0.92,
            cliponaxis=False,
            customdata=customdata,
            hovertemplate=(
                ("顶部区域峰瓣" if is_top else "底部区域峰瓣")
                + " %{customdata[0]} · %{customdata[1]}"
                + "<br>代表极值 %{x|%Y-%m-%d} · %{customdata[4]:.2f}"
                + "<br>峰瓣 %{customdata[2]}—%{customdata[3]}"
                + "<br>区域 %{customdata[7]}—%{customdata[8]}"
                + "<br>规范锚点 %{customdata[5]} · %{customdata[6]:.2f}"
                + "<br>确认日 %{customdata[9]} · 滞后 %{customdata[10]} 日"
                + "<br>价格带 %{customdata[11]:.2%}<extra></extra>"
            ),
        )
    )


def _add_initial_marker(
    figure: go.Figure,
    daily: pd.DataFrame,
    initial: pd.DataFrame,
) -> None:
    if initial.empty:
        return
    row = initial.iloc[0]
    date = pd.Timestamp(row["anchor_date"])
    figure.add_trace(
        go.Scatter(
            x=[date],
            y=[float(row["anchor_price"])],
            mode="markers",
            name="初始化事件",
            marker=dict(symbol="x", size=10, color=COLORS["initial"]),
            customdata=[
                [
                    row["event_type"],
                    float(row["anchor_price"]),
                    pd.Timestamp(row["confirmation_date"]).strftime("%Y-%m-%d"),
                ]
            ],
            hovertemplate=(
                "初始化事件（不进入主统计）"
                "<br>类型 %{customdata[0]}"
                "<br>锚点 %{x|%Y-%m-%d}"
                "<br>锚点价 %{customdata[1]:.2f}"
                "<br>确认日 %{customdata[2]}<extra></extra>"
            ),
        )
    )


def _add_pending_marker(
    figure: go.Figure,
    daily: pd.DataFrame,
    pending: pd.DataFrame,
) -> None:
    if pending.empty:
        return
    row = pending.iloc[0]
    date = pd.Timestamp(row["anchor_date"])
    is_top = row["event_type"] == "top"
    visual_price = daily.loc[date, "high" if is_top else "low"]
    visual_price *= 1.012 if is_top else 0.988
    figure.add_trace(
        go.Scatter(
            x=[date],
            y=[visual_price],
            mode="markers",
            name="未确认候选",
            marker=dict(
                symbol="diamond-open",
                size=12,
                color=COLORS["pending"],
                line=dict(width=2),
            ),
            customdata=[[row["event_type"], float(row["anchor_price"])]],
            hovertemplate=(
                "末端未确认候选"
                "<br>候选类型 %{customdata[0]}"
                "<br>锚点 %{x|%Y-%m-%d}"
                "<br>锚点价 %{customdata[1]:.2f}"
                "<br>尚未达到反转阈值<extra></extra>"
            ),
        )
    )


def build_viewer_panels(
    vipdoc: Path | str,
    threshold: float = 0.10,
) -> list[ViewerPanel]:
    """Build the seven base figures without choosing a page variant."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    vipdoc = Path(vipdoc)
    result = []
    for index_id, index_name, _, relative_path, float_prices in INDEX_SPECS:
        daily = read_tdx_daily(vipdoc / relative_path, float_prices=float_prices)
        adjusted_threshold = threshold_for_index(index_id, threshold)
        figure = make_figure(daily, index_name, adjusted_threshold, index_id=index_id)
        result.append(ViewerPanel(index_id, index_name, daily, figure))
    return result


def write_viewer_panels(
    panels: list[ViewerPanel],
    output_path: Path | str,
    threshold: float = 0.10,
    page: ViewerPage = ViewerPage(),
) -> Path:
    """Render prepared panels through the shared offline-page implementation."""

    if not panels:
        raise ValueError("panels must not be empty")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tabs = []
    panel_html = []
    chart_specs = []
    for position, panel in enumerate(panels):
        chart_id = f"chart-{panel.index_id}"
        chart_specs.append(f'"{chart_id}":{pio.to_json(panel.figure)}')
        active = position == 0
        tabs.append(
            f'<button class="tab{" is-active" if active else ""}" role="tab" '
            f'aria-selected="{str(active).lower()}" aria-controls="panel-{panel.index_id}" '
            f'data-panel="panel-{panel.index_id}" data-chart="{chart_id}">{escape(panel.index_name)}</button>'
        )
        panel_html.append(
            f'<section id="panel-{panel.index_id}" class="panel{" is-active" if active else ""}" '
            f'role="tabpanel"><div class="chart-shell"><div id="{chart_id}" '
            f'class="plotly-graph-div"></div></div></section>'
        )

    html = _page_html(
        tabs="".join(tabs),
        panels="".join(panel_html),
        plotly_js=get_plotlyjs(),
        threshold=threshold,
        chart_specs="{" + ",".join(chart_specs) + "}",
        page=page,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def write_viewer(
    vipdoc: Path | str,
    output_path: Path | str,
    threshold: float = 0.10,
) -> Path:
    """Write the base offline viewer."""

    return write_viewer_panels(
        build_viewer_panels(vipdoc, threshold),
        output_path,
        threshold,
    )


def _page_html(
    *,
    tabs: str,
    panels: str,
    plotly_js: str,
    threshold: float,
    chart_specs: str,
    page: ViewerPage = ViewerPage(),
) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(page.browser_title)} · 基础尺度 {threshold:.0%}</title>
  <style>
    :root {{
      --background: {COLORS['background']};
      --paper: {COLORS['paper']};
      --ink: {COLORS['ink']};
      --muted: {COLORS['muted']};
      --line: {COLORS['line']};
      --up: {COLORS['up']};
      --down: {COLORS['down']};
      --pending: {COLORS['pending']};
      --top: {COLORS['top']};
      --bottom: {COLORS['bottom']};
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: var(--background); color: var(--ink); }}
    body {{ font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif; }}
    .topbar {{
      position: sticky; top: 0; z-index: 10; display: flex; align-items: center;
      gap: 22px; min-height: 62px; padding: 10px 20px; background: rgba(244,244,241,.96);
      border-bottom: 1px solid var(--line); backdrop-filter: blur(10px);
    }}
    .heading {{ min-width: max-content; }}
    .heading strong {{ display: block; font-size: 15px; letter-spacing: .02em; }}
    .heading span {{
      display: block; margin-top: 2px; color: var(--muted); font-size: 12px;
      text-wrap: pretty;
    }}
    .phase-legend {{
      display: flex; flex: 0 0 auto; align-items: center; gap: 11px;
      color: var(--muted); font-size: 12px; white-space: nowrap;
    }}
    .phase-key {{ display: inline-flex; align-items: center; gap: 5px; }}
    .phase-swatch {{
      width: 10px; height: 10px; border: 1px solid rgba(32,35,39,.14);
      border-radius: 2px;
    }}
    .phase-swatch.is-up {{ background: var(--up); opacity: .46; }}
    .phase-swatch.is-down {{ background: var(--down); opacity: .46; }}
    .phase-swatch.is-pending {{ background: var(--pending); opacity: .58; }}
    .phase-swatch.is-top {{ background: var(--top); opacity: .76; }}
    .phase-swatch.is-bottom {{ background: var(--bottom); opacity: .76; }}
    {page.extra_css}
    .tabs {{
      display: flex; flex: 1 1 auto; min-width: 0; gap: 4px;
      overflow-x: auto; scrollbar-width: thin;
    }}
    .tab {{
      min-height: 38px; padding: 0 13px; border: 1px solid transparent; border-radius: 4px;
      background: transparent; color: var(--muted); font: inherit; font-size: 13px;
      white-space: nowrap; cursor: pointer;
    }}
    .tab:hover {{ color: var(--ink); background: var(--paper); }}
    .tab:focus-visible {{ outline: 2px solid #737982; outline-offset: 2px; }}
    .tab.is-active {{ color: var(--ink); background: var(--paper); border-color: var(--line); font-weight: 600; }}
    main {{ width: 100%; padding: 14px 16px 18px; }}
    .panel {{ display: none; width: 100%; }}
    .panel.is-active {{ display: block; }}
    .chart-shell {{
      position: relative; width: 100%; height: 540px; background: var(--paper);
    }}
    .chart-shell::before {{
      content: "载入图表…"; position: absolute; inset: 0; display: grid; place-items: center;
      color: var(--muted); font-size: 13px; letter-spacing: .04em;
    }}
    .chart-shell.is-rendered::before {{ display: none; }}
    .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
    @media (max-width: 1120px) {{
      .topbar {{ flex-wrap: wrap; gap: 8px 20px; }}
      .tabs {{ flex-basis: 100%; }}
    }}
    @media (max-width: 760px) {{
      .topbar {{ display: block; padding: 10px 12px 8px; }}
      .heading {{ margin-bottom: 7px; }}
      .phase-legend {{ margin-bottom: 7px; }}
      main {{ padding: 8px 0 0; }}
      .chart-shell {{ height: 540px; }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header class="topbar">
    <div class="heading"><strong>{escape(page.heading)}</strong><span>基础 {threshold:.0%} · 分指数波动调整 · 区域顶底 · 日K最高/最低价确认{escape(page.subtitle_suffix)}</span></div>
    <div class="phase-legend" role="list" aria-label="阶段背景图例">
      <span class="phase-key" role="listitem"><i class="phase-swatch is-up" aria-hidden="true"></i>上行期</span>
      <span class="phase-key" role="listitem"><i class="phase-swatch is-down" aria-hidden="true"></i>下行期</span>
      <span class="phase-key" role="listitem"><i class="phase-swatch is-pending" aria-hidden="true"></i>待确认期</span>
      <span class="phase-key" role="listitem"><i class="phase-swatch is-top" aria-hidden="true"></i>顶部区域</span>
      <span class="phase-key" role="listitem"><i class="phase-swatch is-bottom" aria-hidden="true"></i>底部区域</span>
      {page.extra_legend_html}
    </div>
    <nav class="tabs" role="tablist" aria-label="指数切换">{tabs}</nav>
  </header>
  <main>{panels}</main>
  <script>
    const tabButtons = Array.from(document.querySelectorAll('.tab'));
    const chartSpecs = {chart_specs};
    const renderPromises = new Map();
    const plotConfig = {{
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    }};
    function renderChart(chartId) {{
      if (!renderPromises.has(chartId)) {{
        const chart = document.getElementById(chartId);
        const spec = chartSpecs[chartId];
        const promise = Plotly.newPlot(chart, spec.data, spec.layout, plotConfig).then(() => {{
          chart.closest('.chart-shell').classList.add('is-rendered');
          return chart;
        }});
        renderPromises.set(chartId, promise);
      }}
      return renderPromises.get(chartId);
    }}
    function activateTab(button) {{
      tabButtons.forEach((item) => {{
        const active = item === button;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
        document.getElementById(item.dataset.panel).classList.toggle('is-active', active);
      }});
      requestAnimationFrame(() => {{
        renderChart(button.dataset.chart).then((chart) => Plotly.Plots.resize(chart));
      }});
    }}
    tabButtons.forEach((button, index) => {{
      button.addEventListener('click', () => activateTab(button));
      button.addEventListener('keydown', (event) => {{
        if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
        event.preventDefault();
        const direction = event.key === 'ArrowRight' ? 1 : -1;
        const next = tabButtons[(index + direction + tabButtons.length) % tabButtons.length];
        next.focus(); activateTab(next);
      }});
    }});
    window.addEventListener('resize', () => {{
      const active = document.querySelector('.tab.is-active');
      if (active && renderPromises.has(active.dataset.chart)) {{
        renderPromises.get(active.dataset.chart).then((chart) => Plotly.Plots.resize(chart));
      }}
    }});
    requestAnimationFrame(() => activateTab(document.querySelector('.tab.is-active')));
  </script>
</body>
</html>
"""
