import json
import struct

import pandas as pd
import pytest

from research.index_turning_points.visualization import viewer as visualize


def sample_daily():
    index = pd.bdate_range("2020-01-01", periods=8)
    close = pd.Series([100, 95, 104.5, 110, 99, 90, 99, 105], index=index)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "amount": 1000.0,
            "volume": 100,
            "reserved": 0,
        }
    )


def sample_ma20_breadth():
    dates = pd.bdate_range("2020-01-01", periods=8)
    above = pd.Series([60, 75, 90, 120, 150, 135, 105, 80], dtype=int)
    valid = pd.Series([200] * len(dates), dtype=int)
    return pd.DataFrame(
        {
            "date": dates,
            "breadth_ma20": above.div(valid),
            "above_count_ma20": above,
            "valid_count_ma20": valid,
        }
    )


def sample_ma20_signals():
    dates = pd.bdate_range("2020-01-01", periods=8)
    rows = []
    specifications = (
        ("top", "ma20_breadth_reversal_top", 4, 0.72, -0.08),
        ("bottom", "ma20_breadth_reversal_bottom", 6, 0.25, 0.10),
    )
    for direction, signal_id, onset_position, raw_value, change in specifications:
        for position, date in enumerate(dates):
            onset = position == onset_position
            rows.append(
                {
                    "date": date,
                    "signal_id": signal_id,
                    "direction": direction,
                    "raw_value": raw_value,
                    "breadth_change_5d": change,
                    "triggered": onset,
                    "event_onset": onset,
                    "episode_id": f"{signal_id}::1" if onset else pd.NA,
                    "episode_number": 1 if onset else pd.NA,
                    "ma_window": 20,
                }
            )
    return pd.DataFrame(rows)


def multi_lobe_daily():
    index = pd.bdate_range("2020-01-01", periods=13)
    close = pd.Series(
        [80, 70, 78, 90, 100, 95, 94, 99, 89, 82, 80, 85, 90],
        index=index,
        dtype=float,
    )
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "amount": 1000.0,
            "volume": 100,
            "reserved": 0,
        }
    )


def test_phase_intervals_split_pending_and_confirmed_directions():
    labels = pd.DataFrame(
        [
            {
                "event_type": "bottom",
                "status": "confirmed",
                "anchor_date": "2020-01-01",
                "confirmation_date": "2020-01-03",
            },
            {
                "event_type": "top",
                "status": "confirmed",
                "anchor_date": "2020-01-08",
                "confirmation_date": "2020-01-10",
            },
            {
                "event_type": "bottom",
                "status": "unconfirmed",
                "anchor_date": "2020-01-15",
                "confirmation_date": pd.NaT,
            },
        ]
    )

    intervals = visualize._phase_intervals(labels, pd.Timestamp("2020-01-20"))

    assert intervals == [
        ("pending", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-03")),
        ("up", pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-08")),
        ("pending", pd.Timestamp("2020-01-08"), pd.Timestamp("2020-01-10")),
        ("down", pd.Timestamp("2020-01-10"), pd.Timestamp("2020-01-15")),
        ("pending", pd.Timestamp("2020-01-15"), pd.Timestamp("2020-01-20")),
    ]


def test_figure_has_phase_backgrounds_region_lobes_and_trading_breaks():
    figure = visualize.make_figure(sample_daily(), "测试指数", threshold=0.10)

    assert figure.layout.height == 540
    assert {trace.name for trace in figure.data} == {
        "日K",
        "顶部区域峰瓣",
        "底部区域峰瓣",
        "初始化事件",
        "未确认候选",
    }
    shape_names = {shape.name for shape in figure.layout.shapes}
    assert {"phase:up", "phase:down", "phase:pending"} <= shape_names
    assert any(name.startswith("region-envelope:top:") for name in shape_names)
    assert any(name.startswith("region-envelope:bottom:") for name in shape_names)
    assert any(name.startswith("region-lobe:top:") for name in shape_names)
    assert any(name.startswith("region-lobe:bottom:") for name in shape_names)
    breaks = set(figure.layout.xaxis.rangebreaks[0].values)
    assert "2020-01-04" in breaks
    assert "2020-01-05" in breaks


def test_figure_can_add_shared_full_a_ma20_breadth_row():
    figure = visualize.make_figure(
        sample_daily(),
        "测试指数",
        threshold=0.10,
        ma20_breadth=sample_ma20_breadth(),
    )

    breadth = next(trace for trace in figure.data if trace.name == "全 A MA20 宽度")
    assert figure.layout.height == 720
    assert breadth.xaxis == "x2"
    assert breadth.yaxis == "y2"
    assert list(breadth.y) == [0.30, 0.375, 0.45, 0.60, 0.75, 0.675, 0.525, 0.40]
    assert figure.layout.xaxis.matches == "x2"
    assert figure.layout.xaxis.showticklabels is False
    assert figure.layout.xaxis.rangeslider.visible is False
    assert figure.layout.xaxis2.rangeslider.visible is True
    assert figure.layout.yaxis2.range == (0.0, 1.0)
    assert figure.layout.yaxis2.dtick == 0.2
    assert breadth.customdata[0][2] == "—"
    assert breadth.customdata[5][2] == "+37.5%"
    threshold_shapes = {
        shape.name: shape for shape in figure.layout.shapes if shape.name
    }
    assert threshold_shapes["ma20-threshold:top"].y0 == 0.70
    assert threshold_shapes["ma20-threshold:bottom"].y0 == 0.30
    assert threshold_shapes["ma20-threshold:top"].xref == "x2 domain"


def test_ma20_signal_onsets_are_distinct_markers_on_the_kline_row():
    daily = sample_daily()
    figure = visualize.make_figure(
        daily,
        "测试指数",
        threshold=0.10,
        ma20_breadth=sample_ma20_breadth(),
        ma20_signals=sample_ma20_signals(),
    )

    top = next(trace for trace in figure.data if trace.name == "MA20顶部信号首日")
    bottom = next(trace for trace in figure.data if trace.name == "MA20底部信号首日")
    region_top = next(trace for trace in figure.data if trace.name == "顶部区域峰瓣")
    breadth = next(trace for trace in figure.data if trace.name == "全 A MA20 宽度")
    assert top.marker.symbol == "circle-open-dot"
    assert bottom.marker.symbol == "circle-open-dot"
    assert region_top.marker.symbol == "triangle-down"
    assert top.marker.symbol != region_top.marker.symbol
    assert top.xaxis is None and top.yaxis is None
    assert bottom.xaxis is None and bottom.yaxis is None
    assert breadth.xaxis == "x2" and breadth.yaxis == "y2"
    assert list(pd.DatetimeIndex(top.x)) == [daily.index[4]]
    assert list(pd.DatetimeIndex(bottom.x)) == [daily.index[6]]
    assert top.y[0] == pytest.approx(daily.iloc[4]["high"] * 1.035)
    assert bottom.y[0] == pytest.approx(daily.iloc[6]["low"] * 0.965)
    assert top.customdata[0][1:3] == [0.72, -0.08]


def test_ma20_signal_onset_must_also_be_triggered():
    signals = sample_ma20_signals()
    signals.loc[signals["event_onset"].idxmax(), "triggered"] = False

    with pytest.raises(ValueError, match="must also be triggered"):
        visualize.make_figure(
            sample_daily(),
            "测试指数",
            threshold=0.10,
            ma20_signals=signals,
        )


def test_ma20_breadth_rejects_values_inconsistent_with_counts():
    breadth = sample_ma20_breadth()
    breadth.loc[0, "breadth_ma20"] = 0.31

    with pytest.raises(ValueError, match="does not match"):
        visualize.make_figure(
            sample_daily(),
            "测试指数",
            threshold=0.10,
            ma20_breadth=breadth,
        )


def test_full_a_is_the_default_viewer_panel():
    assert visualize._ordered_index_specs()[0][0] == "all_a"


def test_multi_lobe_top_has_one_marker_and_band_per_peak():
    daily = multi_lobe_daily()

    figure = visualize.make_figure(daily, "M顶样例", threshold=0.10)

    top_trace = next(trace for trace in figure.data if trace.name == "顶部区域峰瓣")
    top_lobes = [
        shape
        for shape in figure.layout.shapes
        if shape.name.startswith("region-lobe:top:")
    ]
    assert list(pd.DatetimeIndex(top_trace.x)) == [daily.index[4], daily.index[7]]
    assert len(top_lobes) == 2
    assert [row[0] for row in top_trace.customdata] == ["1/2", "2/2"]


def test_writes_one_html_with_tabs_and_full_width_chart(tmp_path, monkeypatch):
    vipdoc = tmp_path / "vipdoc"
    relative_path = "sh/lday/test.day"
    source = vipdoc / relative_path
    source.parent.mkdir(parents=True)
    daily = sample_daily()
    source.write_bytes(
        b"".join(
            struct.pack(
                "<IIIIIfII",
                int(date.strftime("%Y%m%d")),
                round(row.open * 100),
                round(row.high * 100),
                round(row.low * 100),
                round(row.close * 100),
                row.amount,
                int(row.volume),
                0,
            )
            for date, row in daily.iterrows()
        )
    )
    monkeypatch.setattr(
        visualize,
        "INDEX_SPECS",
        (("test", "测试指数", "TEST", relative_path, False),),
    )
    monkeypatch.setattr(
        visualize,
        "threshold_for_index",
        lambda index_id, base_threshold: base_threshold,
    )

    output = visualize.write_viewer(vipdoc, tmp_path / "viewer.html", 0.10)
    html = output.read_text(encoding="utf-8")

    assert html.count('role="tab"') == 1
    assert html.count('role="tabpanel"') == 1
    assert 'class="plotly-graph-div"' in html
    assert "height: 540px" in html
    assert "const renderPromises = new Map()" in html
    assert html.count("const promise = Plotly.newPlot(") == 1
    assert "2020-01-04" in html
    assert "区域顶底 · 日K最高/最低价确认" in html
    assert "基础 10% · 分指数波动调整" in html
    assert "上行期" in html
    assert "下行期" in html
    assert "待确认期" in html
    assert "顶部区域" in html
    assert "底部区域" in html

    ma20_output = visualize.write_viewer(
        vipdoc,
        tmp_path / "viewer-ma20.html",
        0.10,
        ma20_breadth=sample_ma20_breadth(),
        ma20_signals=sample_ma20_signals(),
    )
    ma20_html = ma20_output.read_text(encoding="utf-8")
    assert all(line == line.rstrip() for line in ma20_html.splitlines())
    assert "指数顶底区域与全 A MA20 宽度" in ma20_html
    assert "下排固定为全 A MA20 宽度" in ma20_html
    assert "上排空心圆点为点时信号" in ma20_html
    assert json.dumps("MA20顶部信号首日", ensure_ascii=True)[1:-1] in ma20_html
    assert json.dumps("MA20底部信号首日", ensure_ascii=True)[1:-1] in ma20_html
    assert "height: 720px" in ma20_html
    assert "delete layout.xaxis.rangeselector" in ma20_html
