"""One-page Plotly viewer for index OHLC and turning-point labels."""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from .labels import directional_change_labels
from .pipeline import INDEX_SPECS, read_tdx_daily


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


def make_figure(
    daily: pd.DataFrame,
    index_name: str,
    threshold: float = 0.10,
) -> go.Figure:
    """Build one full-width OHLC chart with directional-change markers."""

    labels = directional_change_labels(daily["close"], threshold)
    confirmed = labels[labels["status"] == "confirmed"]
    eligible = confirmed[confirmed["eligible"]]
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

    _add_confirmed_markers(figure, daily, eligible, "top")
    _add_confirmed_markers(figure, daily, eligible, "bottom")
    _add_initial_marker(figure, daily, initial)
    _add_pending_marker(figure, daily, pending)

    figure.update_layout(
        title=dict(
            text=f"{index_name} · {threshold:.0%} 方向变化",
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


def _add_confirmed_markers(
    figure: go.Figure,
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    event_type: str,
) -> None:
    selected = labels[labels["event_type"] == event_type]
    if selected.empty:
        return

    is_top = event_type == "top"
    dates = pd.DatetimeIndex(selected["anchor_date"])
    visual_price = daily.loc[dates, "high" if is_top else "low"].to_numpy()
    visual_price *= 1.012 if is_top else 0.988
    customdata = np.column_stack(
        [
            selected["anchor_price"].astype(float),
            pd.to_datetime(selected["confirmation_date"]).dt.strftime("%Y-%m-%d"),
            selected["confirmation_price"].astype(float),
            selected["confirmation_lag"].astype(int),
            selected["reversal_return"].astype(float),
        ]
    )

    figure.add_trace(
        go.Scatter(
            x=dates,
            y=visual_price,
            mode="markers",
            name="已确认顶部" if is_top else "已确认底部",
            marker=dict(
                symbol="triangle-down" if is_top else "triangle-up",
                size=11,
                color=COLORS["top" if is_top else "bottom"],
                line=dict(color="#FFFFFF", width=1),
            ),
            customdata=customdata,
            hovertemplate=(
                ("已确认顶部" if is_top else "已确认底部")
                + "<br>锚点 %{x|%Y-%m-%d}"
                + "<br>锚点收盘 %{customdata[0]:.2f}"
                + "<br>确认日 %{customdata[1]}"
                + "<br>确认收盘 %{customdata[2]:.2f}"
                + "<br>确认滞后 %{customdata[3]} 个交易日"
                + "<br>反转幅度 %{customdata[4]:.2%}<extra></extra>"
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
            y=[daily.loc[date, "close"]],
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
                "<br>锚点收盘 %{customdata[1]:.2f}"
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
                "<br>锚点收盘 %{customdata[1]:.2f}"
                "<br>尚未达到反转阈值<extra></extra>"
            ),
        )
    )


def write_viewer(
    vipdoc: Path | str,
    output_path: Path | str,
    threshold: float = 0.10,
) -> Path:
    """Write one offline HTML with one full-width tab per index."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    vipdoc = Path(vipdoc)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tabs = []
    panels = []
    chart_ids = []
    for position, (index_id, index_name, _, relative_path, float_prices) in enumerate(INDEX_SPECS):
        daily = read_tdx_daily(vipdoc / relative_path, float_prices=float_prices)
        figure = make_figure(daily, index_name, threshold)
        chart_id = f"chart-{index_id}"
        chart_ids.append(chart_id)
        figure_html = pio.to_html(
            figure,
            full_html=False,
            include_plotlyjs=False,
            default_width="100%",
            default_height="100%",
            div_id=chart_id,
            config={
                "displaylogo": False,
                "responsive": True,
                "scrollZoom": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )
        active = position == 0
        tabs.append(
            f'<button class="tab{" is-active" if active else ""}" role="tab" '
            f'aria-selected="{str(active).lower()}" aria-controls="panel-{index_id}" '
            f'data-panel="panel-{index_id}" data-chart="{chart_id}">{escape(index_name)}</button>'
        )
        panels.append(
            f'<section id="panel-{index_id}" class="panel{" is-active" if active else ""}" '
            f'role="tabpanel"><div class="chart-shell">{figure_html}</div></section>'
        )

    html = _page_html(
        tabs="".join(tabs),
        panels="".join(panels),
        plotly_js=get_plotlyjs(),
        threshold=threshold,
        chart_ids=chart_ids,
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


def _page_html(
    *,
    tabs: str,
    panels: str,
    plotly_js: str,
    threshold: float,
    chart_ids: list[str],
) -> str:
    chart_ids_js = ",".join(f'"{chart_id}"' for chart_id in chart_ids)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>指数顶底检查 · {threshold:.0%}</title>
  <style>
    :root {{
      --background: {COLORS['background']};
      --paper: {COLORS['paper']};
      --ink: {COLORS['ink']};
      --muted: {COLORS['muted']};
      --line: {COLORS['line']};
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; min-height: 100%; background: var(--background); color: var(--ink); }}
    body {{ font-family: "Noto Sans CJK SC", "Source Han Sans SC", sans-serif; }}
    .topbar {{
      position: sticky; top: 0; z-index: 10; display: flex; align-items: center;
      gap: 24px; min-height: 62px; padding: 10px 20px; background: rgba(244,244,241,.96);
      border-bottom: 1px solid var(--line); backdrop-filter: blur(10px);
    }}
    .heading {{ min-width: max-content; }}
    .heading strong {{ display: block; font-size: 15px; letter-spacing: .02em; }}
    .heading span {{ display: block; margin-top: 2px; color: var(--muted); font-size: 12px; }}
    .tabs {{ display: flex; gap: 4px; overflow-x: auto; scrollbar-width: thin; }}
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
    .chart-shell {{ width: 100%; height: 540px; background: var(--paper); }}
    .plotly-graph-div {{ width: 100% !important; height: 100% !important; }}
    @media (max-width: 760px) {{
      .topbar {{ display: block; padding: 10px 12px 8px; }}
      .heading {{ margin-bottom: 8px; }}
      main {{ padding: 8px 0 0; }}
      .chart-shell {{ height: 540px; }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header class="topbar">
    <div class="heading"><strong>指数顶底检查</strong><span>{threshold:.0%} 方向变化 · 收盘价确认</span></div>
    <nav class="tabs" role="tablist" aria-label="指数切换">{tabs}</nav>
  </header>
  <main>{panels}</main>
  <script>
    const tabButtons = Array.from(document.querySelectorAll('.tab'));
    const chartIds = [{chart_ids_js}];
    function activateTab(button) {{
      tabButtons.forEach((item) => {{
        const active = item === button;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
        document.getElementById(item.dataset.panel).classList.toggle('is-active', active);
      }});
      requestAnimationFrame(() => Plotly.Plots.resize(document.getElementById(button.dataset.chart)));
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
      if (active) Plotly.Plots.resize(document.getElementById(active.dataset.chart));
    }});
    requestAnimationFrame(() => chartIds.forEach((id) => Plotly.Plots.resize(document.getElementById(id))));
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vipdoc",
        type=Path,
        default=Path.home() / ".local/share/tdxcfv/drive_c/tc/vipdoc",
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts/index_turning_points.html",
    )
    args = parser.parse_args()
    print(write_viewer(args.vipdoc, args.output, args.threshold))


if __name__ == "__main__":
    main()
