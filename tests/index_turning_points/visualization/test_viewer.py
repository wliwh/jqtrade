import struct

import pandas as pd

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
