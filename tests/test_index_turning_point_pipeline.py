import struct

import pandas as pd
import pytest

from research.index_turning_points import pipeline


def write_standard(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            struct.pack(
                "<IIIIIfII",
                date,
                round(open_ * 100),
                round(high * 100),
                round(low * 100),
                round(close * 100),
                amount,
                volume,
                0,
            )
            for date, open_, high, low, close, amount, volume in rows
        )
    )


def write_float(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"".join(
            struct.pack(
                "<IfffffII",
                date,
                open_,
                high,
                low,
                close,
                amount,
                volume,
                0,
            )
            for date, open_, high, low, close, amount, volume in rows
        )
    )


def sample_rows():
    return [
        (20200102, 100, 101, 99, 100, 1000.0, 100),
        (20200103, 119, 121, 118, 120, 1200.0, 120),
        (20200106, 97, 98, 95, 96, 1100.0, 110),
        (20200107, 75, 76, 74, 75, 900.0, 90),
        (20200108, 89, 91, 88, 90, 1050.0, 105),
        (20200109, 104, 106, 103, 105, 1150.0, 115),
    ]


@pytest.mark.filterwarnings("error::FutureWarning")
def test_reads_standard_integer_price_file(tmp_path):
    path = tmp_path / "standard.day"
    write_standard(path, sample_rows())

    daily = pipeline.read_tdx_daily(path)

    assert daily.index.tolist() == list(pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]))
    assert daily["close"].tolist() == [100, 120, 96, 75, 90, 105]


def test_reads_float_price_file_and_removes_exact_duplicate(tmp_path):
    path = tmp_path / "float.day"
    rows = sample_rows()
    write_float(path, rows[:2] + [rows[1]] + rows[2:])

    daily = pipeline.read_tdx_daily(path, float_prices=True)

    assert len(daily) == len(rows)
    assert daily.index.is_unique
    assert daily["close"].tolist() == [100, 120, 96, 75, 90, 105]


def test_rejects_conflicting_duplicate_date(tmp_path):
    path = tmp_path / "conflict.day"
    rows = sample_rows()
    conflict = (20200103, 109, 112, 108, 111, 1200.0, 120)
    write_float(path, rows[:2] + [conflict] + rows[2:])

    with pytest.raises(ValueError, match="conflicting rows"):
        pipeline.read_tdx_daily(path, float_prices=True)


def test_forward_outcomes_require_complete_window():
    close = pd.Series(
        [100.0, 90.0, 110.0, 80.0],
        index=pd.date_range("2020-01-01", periods=4),
    )

    outcomes = pipeline.forward_outcomes(close, horizons=(2,))

    assert outcomes.iloc[0]["future_max_down_2d"] == pytest.approx(-0.10)
    assert outcomes.iloc[0]["future_max_up_2d"] == pytest.approx(0.10)
    assert outcomes.iloc[0]["future_return_2d"] == pytest.approx(0.10)
    assert outcomes.iloc[1]["future_max_down_2d"] == pytest.approx(80 / 90 - 1)
    assert outcomes.iloc[1]["future_max_up_2d"] == pytest.approx(110 / 90 - 1)
    assert outcomes.iloc[1]["future_return_2d"] == pytest.approx(80 / 90 - 1)
    assert outcomes.iloc[2:]["future_return_2d"].isna().all()


def test_pipeline_writes_three_csv_files(tmp_path, monkeypatch):
    vipdoc = tmp_path / "vipdoc"
    relative_path = "sh/lday/test.day"
    write_standard(vipdoc / relative_path, sample_rows())
    monkeypatch.setattr(
        pipeline,
        "INDEX_SPECS",
        (("test", "测试指数", "TEST", relative_path, False),),
    )

    outputs = pipeline.run_pipeline(vipdoc, tmp_path / "output")

    assert set(outputs) == {"manifest", "labels", "outcomes"}
    assert all(path.exists() for path in outputs.values())
    manifest = pd.read_csv(outputs["manifest"])
    labels = pd.read_csv(outputs["labels"])
    outcomes = pd.read_csv(outputs["outcomes"])
    assert manifest.loc[0, "rows"] == 6
    assert set(labels["threshold"]) == {0.05, 0.10, 0.20}
    assert {"date", "close", "future_return_5d"} <= set(outcomes.columns)
