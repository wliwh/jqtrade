import struct

import pandas as pd

from research.index_turning_points import visualize


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


def test_figure_has_markers_compact_height_and_non_trading_breaks():
    figure = visualize.make_figure(sample_daily(), "测试指数", threshold=0.10)

    assert figure.layout.height == 540
    assert {trace.name for trace in figure.data} == {
        "日K",
        "已确认顶部",
        "已确认底部",
        "初始化事件",
        "未确认候选",
    }
    breaks = set(figure.layout.xaxis.rangebreaks[0].values)
    assert "2020-01-04" in breaks
    assert "2020-01-05" in breaks


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

    output = visualize.write_viewer(vipdoc, tmp_path / "viewer.html", 0.10)
    html = output.read_text(encoding="utf-8")

    assert html.count('role="tab"') == 1
    assert html.count('role="tabpanel"') == 1
    assert 'class="plotly-graph-div"' in html
    assert ".chart-shell { width: 100%; height: 540px" in html
    assert "2020-01-04" in html
