from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.index_turning_points.visualization.bsearch_index import (
    KEYWORD_SPECS,
    VIEWER_INDEX_SPECS,
    KeywordSpec,
    align_keyword_to_prices,
    build_viewer_payload,
    load_regions,
    point_in_time_heat_z,
    read_bsearch_csv,
)


def _eligible_rows() -> pd.DataFrame:
    rows = []
    for spec in KEYWORD_SPECS:
        rows.extend(
            [
                {"date": "2024-01-05", "keyword": spec.keyword, "type": "all", "count": 10.0},
                {"date": "2024-01-06", "keyword": spec.keyword, "type": "all", "count": 99.0},
                {"date": "2024-01-08", "keyword": spec.keyword, "type": "all", "count": 20.0},
            ]
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [99.0, 101.0],
            "high": [101.0, 103.0],
            "low": [98.0, 100.0],
            "close": [100.0, 102.0],
        },
        index=pd.DatetimeIndex(["2024-01-05", "2024-01-08"], name="date"),
    )


def test_eligible_source_keywords_have_local_default_indices() -> None:
    keywords = [spec.keyword for spec in KEYWORD_SPECS]
    assert keywords == [
        "上证指数",
        "股票",
        "a股",
        "基金",
        "股市",
        "上证",
        "熊市",
        "沪深300",
        "上证50",
        "牛市",
        "中证500",
        "创业板指",
        "科创50",
    ]
    assert not set(keywords).intersection(
        {"恒生指数", "道琼斯指数", "纳斯达克指数", "港股", "原油", "铜价", "人民币汇率"}
    )
    mappings = {spec.keyword: spec.index_id for spec in KEYWORD_SPECS}
    assert mappings["沪深300"] == "csi300"
    assert mappings["上证50"] == "sse50"
    assert mappings["中证500"] == "csi500"
    assert mappings["创业板指"] == "chinext"
    assert mappings["科创50"] == "star50"
    assert mappings["牛市"] == "sse_composite"


def test_read_bsearch_csv_normalizes_unnamed_date_column(tmp_path: Path) -> None:
    frame = _eligible_rows().copy()
    frame = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    {"date": pd.Timestamp("2024-01-05"), "keyword": "恒生指数", "type": "all", "count": 30.0},
                    {"date": pd.Timestamp("2024-01-05"), "keyword": "原油", "type": "all", "count": 40.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    path = tmp_path / "bsearch.csv"
    frame.rename(columns={"date": ""}).to_csv(path, index=False)

    actual = read_bsearch_csv(path)

    assert list(actual.columns) == ["date", "keyword", "type", "count"]
    assert actual["date"].dtype.kind == "M"
    assert {"恒生指数", "原油"}.issubset(set(actual["keyword"]))


def test_align_keyword_to_prices_drops_non_trading_dates() -> None:
    spec = KEYWORD_SPECS[0]
    aligned = align_keyword_to_prices(_eligible_rows(), _prices(), spec)

    assert aligned["date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-05", "2024-01-08"]
    assert aligned["count"].tolist() == [10.0, 20.0]


def test_point_in_time_heat_z_is_truncation_invariant() -> None:
    count = pd.Series(range(1, 151), dtype=float)
    short = point_in_time_heat_z(count.iloc[:100], window=20, min_periods=5)
    long = point_in_time_heat_z(count, window=20, min_periods=5)

    pd.testing.assert_series_equal(short, long.iloc[:100])


def test_payload_exposes_only_eligible_keywords_and_all_local_indices() -> None:
    prices = {spec.index_id: _prices() for spec in VIEWER_INDEX_SPECS}
    source = pd.concat(
        [
            _eligible_rows(),
            pd.DataFrame(
                [
                    {"date": pd.Timestamp("2024-01-05"), "keyword": "道琼斯指数", "type": "all", "count": 99.0},
                    {"date": pd.Timestamp("2024-01-05"), "keyword": "螺纹钢", "type": "all", "count": 88.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    payload = build_viewer_payload(source, prices, regions_by_index={"csi300": []})

    assert payload["keywords"] == [spec.keyword for spec in KEYWORD_SPECS]
    assert payload["index_ids"] == [spec.index_id for spec in VIEWER_INDEX_SPECS]
    assert set(payload["indices"]) == {spec.index_id for spec in VIEWER_INDEX_SPECS}
    assert payload["indices"]["sse_composite"]["index_name"] == "上证指数"
    assert payload["indices"]["sse_composite"]["open"] == [99.0, 101.0]
    assert payload["indices"]["sse_composite"]["high"] == [101.0, 103.0]
    assert payload["indices"]["sse_composite"]["low"] == [98.0, 100.0]
    assert payload["indices"]["sse_composite"]["close"] == [100.0, 102.0]
    assert set(payload["series"]) == {spec.keyword for spec in KEYWORD_SPECS}
    assert not {"道琼斯指数", "螺纹钢"}.intersection(payload["series"])
    assert payload["series"]["上证指数"]["skipped_rows"] == 1
    assert payload["series"]["上证指数"]["recommended_index_id"] == "sse_composite"
    assert payload["series"]["上证50"]["recommended_index_id"] == "sse50"
    assert payload["series"]["沪深300"]["recommended_index_id"] == "csi300"
    assert payload["series"]["科创50"]["recommended_index_id"] == "star50"
    assert payload["series"]["上证指数"]["heat_z252"] == [None, None]
    assert "close" not in payload["series"]["上证指数"]


def test_missing_price_mapping_fails_clearly() -> None:
    spec = KeywordSpec("上证指数", "missing")
    with pytest.raises(ValueError, match="price frame must provide"):
        align_keyword_to_prices(_eligible_rows(), pd.DataFrame(), spec)


def test_load_regions_groups_only_eligible_rows(tmp_path: Path) -> None:
    path = tmp_path / "regions.csv"
    pd.DataFrame(
        [
            {"index_id": "sse_composite", "event_type": "bottom", "eligible": True, "region_start": "2024-01-02", "region_end": "2024-01-04"},
            {"index_id": "csi300", "event_type": "top", "eligible": True, "region_start": "2024-02-01", "region_end": "2024-02-02"},
            {"index_id": "csi300", "event_type": "bottom", "eligible": False, "region_start": "2024-03-01", "region_end": "2024-03-02"},
        ]
    ).to_csv(path, index=False)

    actual = load_regions(path)

    assert set(actual) == {"sse_composite", "csi300"}
    assert actual["csi300"] == [{"event_type": "top", "start": "2024-02-01", "end": "2024-02-02"}]


def test_template_uses_ohlc_and_keyword_scoped_manual_peak_annotations() -> None:
    template = (
        Path("research/index_turning_points/visualization/templates")
        / "bsearch_index_viewer.html"
    ).read_text(encoding="utf-8")

    assert "type: 'scattergl'" not in template
    assert template.count("type: 'scatter'") == 3
    assert "type: 'candlestick'" in template
    assert "open: index.open" in template
    assert "high: index.high" in template
    assert "low: index.low" in template
    assert "close: index.close" in template
    assert 'id="indexSelect"' in template
    assert 'id="keywordSelect"' in template
    assert 'id="peakModeButton"' in template
    assert 'id="clearPeaksButton"' in template
    assert 'id="exportPeaksButton"' in template
    assert 'id="importPeaksButton"' in template
    assert 'id="peakFileInput"' in template
    assert 'id="heatButtons"' not in template
    assert "yaxis: 'y2'" in template
    assert "yaxis: 'y3'" in template
    assert "series.count" in template
    assert "series.heat_z252" in template
    assert "dash: 'solid'" in template
    assert "dash: 'dash'" not in template
    assert "DATASET.regions_by_index[index.index_id]" in template
    assert "state.indexId = DATASET.series[state.keyword].recommended_index_id" in template
    assert "bsearch-index-manual-peaks-v1" in template
    assert "elements.chart.on('plotly_click', handleChartClick)" in template
    assert "state.manualPeaks[state.keyword]" in template
    assert "localStorage.setItem(PEAK_STORAGE_KEY" in template
    assert "dragmode: 'pan'" in template
    assert "scrollZoom: true" in template
    assert "Plotly.restyle(elements.chart" in template
    assert "Plotly.relayout(elements.chart" in template
    assert "bsearch-index-manual-peaks/v1" in template
    assert "URL.createObjectURL(blob)" in template
    assert "JSON.parse(await file.text())" in template
    assert "payload.bsearch_sha256 !== METADATA.bsearch_sha256" in template
    assert "state.manualPeaks = imported" in template
    assert "state.annotationMode ? false : 'pan'" not in template
    assert "scrollZoom: !state.annotationMode" not in template
    assert "当前发现" not in template
