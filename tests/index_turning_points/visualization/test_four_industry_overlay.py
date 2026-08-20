import json
import struct
import warnings

import pandas as pd
import pytest

from research.index_turning_points.adapters import legacy_four_industry_v1
from research.index_turning_points.pipelines import render_viewer
from research.index_turning_points.visualization import viewer
from research.index_turning_points.visualization.overlays import four_industry


def sample_daily():
    index = pd.bdate_range("2022-01-03", periods=6)
    close = pd.Series([100, 95, 105, 110, 99, 103], index=index, dtype=float)
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


def sample_signal():
    dates = pd.bdate_range("2022-01-03", periods=6)
    return pd.DataFrame(
        {
            "date": dates,
            "breadth_ma20": [0.40, 0.42, 0.45, 0.48, 0.46, 0.44],
            "triggered": [False, True, True, False, True, False],
            "onset": [False, True, False, False, True, False],
            "continuation": [False, False, True, False, False, False],
            "episode_id": [pd.NA, 1, 1, pd.NA, 2, pd.NA],
            "episode_day": [pd.NA, 1, 2, pd.NA, 1, pd.NA],
            "four_industry_top1_ids": [pd.NA, "bank", "coal", pd.NA, "steel", pd.NA],
            "top1_tie_count_ma20": [1, 1, 1, 1, 1, 1],
            "target_bank": [False, True, False, False, False, False],
            "target_coal": [False, False, True, False, False, False],
            "target_nonferrous": [False, False, False, False, False, False],
            "target_steel": [False, False, False, False, True, False],
        }
    )


def test_load_signal_validates_phase_semantics(tmp_path):
    path = tmp_path / "signal.csv"
    sample_signal().to_csv(path, index=False)

    loaded = legacy_four_industry_v1.load_four_industry_v1_signal(path)

    assert loaded["triggered"].sum() == 3
    assert loaded["onset"].sum() == 2
    assert list(loaded.loc[loaded["onset"], "date"]) == [
        pd.Timestamp("2022-01-04"),
        pd.Timestamp("2022-01-07"),
    ]

    invalid = sample_signal()
    invalid.loc[1, "onset"] = False
    invalid.to_csv(path, index=False)
    with pytest.raises(ValueError, match="onset is inconsistent"):
        legacy_four_industry_v1.load_four_industry_v1_signal(path)


def test_adds_active_ribbon_and_industry_onset_markers():
    daily = sample_daily()
    figure = viewer.make_figure(daily, "测试指数", threshold=0.10)

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        four_industry.add_four_industry_signal(
            figure,
            daily,
            sample_signal(),
        )

    traces = {trace.name: trace for trace in figure.data}
    active = traces["四行业 Top1 活跃期"]
    assert list(pd.DatetimeIndex(active.x)) == list(daily.index)
    assert list(active.y) == [None, 0.965, 0.965, None, 0.965, None]
    assert list(pd.DatetimeIndex(traces["首次触发 · 银行"].x)) == [
        pd.Timestamp("2022-01-04")
    ]
    assert list(pd.DatetimeIndex(traces["首次触发 · 钢铁"].x)) == [
        pd.Timestamp("2022-01-07")
    ]
    assert list(traces["首次触发 · 有色"].x) == []
    assert active.yaxis == "y2"
    assert figure.layout.yaxis2.range == (0.0, 1.0)


def test_alignment_before_signal_start_has_no_future_downcast_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        aligned = four_industry._align_signal(
            sample_daily(),
            sample_signal().iloc[1:].reset_index(drop=True),
        )

    assert not bool(aligned.iloc[0]["triggered"])


def test_writes_separate_html_with_signal_legend(tmp_path, monkeypatch):
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
    signal_path = tmp_path / "signal.csv"
    sample_signal().to_csv(signal_path, index=False)
    monkeypatch.setattr(
        viewer,
        "INDEX_SPECS",
        (("test", "测试指数", "TEST", relative_path, False),),
    )
    monkeypatch.setattr(
        viewer,
        "threshold_for_index",
        lambda index_id, base_threshold: base_threshold,
    )

    output = render_viewer.render_four_industry_v1_viewer(
        vipdoc,
        signal_path,
        tmp_path / "viewer.html",
        0.10,
    )
    html = output.read_text(encoding="utf-8")

    assert "指数顶底区域与四行业宽度信号" in html
    assert "四行业 Top1 V1" in html
    assert json.dumps("四行业 Top1 活跃期", ensure_ascii=True)[1:-1] in html
    assert json.dumps("首次触发 · 银行", ensure_ascii=True)[1:-1] in html
    assert "phase-swatch is-signal" in html
    assert html.count('role="tabpanel"') == 1
