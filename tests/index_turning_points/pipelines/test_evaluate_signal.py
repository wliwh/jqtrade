import hashlib
import json
import struct

import pandas as pd
import pytest

from research.index_turning_points.ground_truth.regions import DEFAULT_REGION_PROTOCOL
from research.index_turning_points.pipelines import evaluate_signal as pipeline
from research.index_turning_points.signals.events import build_signal_events


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_standard(path, dates):
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for position, date in enumerate(dates):
        close = 100.0 + position * 0.2
        records.append(
            struct.pack(
                "<IIIIIfII",
                int(date.strftime("%Y%m%d")),
                round(close * 100),
                round((close + 1.0) * 100),
                round((close - 1.0) * 100),
                round(close * 100),
                1000.0,
                100,
                0,
            )
        )
    path.write_bytes(b"".join(records))


def _write_signal_bundle(root, dates):
    source = pd.DataFrame(
        {
            "date": dates,
            "signal_id": "test_signal",
            "direction": "top",
            "raw_value": 0.0,
            "triggered": [False] * 10
            + [True, True]
            + [False] * 18
            + [True, True, True]
            + [False] * (len(dates) - 33),
            "universe_size": 100,
            "valid_count": 95,
            "version": "signal_v1",
        }
    )
    daily, episodes = build_signal_events(source, capped_confirmation_n=2)
    daily_path = root / "signal_daily.csv"
    episodes_path = root / "signal_episodes.csv"
    root.mkdir(parents=True, exist_ok=True)
    daily.to_csv(daily_path, index=False)
    episodes.to_csv(episodes_path, index=False)
    return daily_path, episodes_path


def _write_ground_truth(root, dates, source_path, source_relative):
    region_dir = root / "regions" / DEFAULT_REGION_PROTOCOL.label_version
    region_dir.mkdir(parents=True, exist_ok=True)
    regions = pd.DataFrame(
        [
            {
                "region_id": "test_top_1",
                "index_id": "test_index",
                "index_name": "测试指数",
                "event_type": "top",
                "status": "confirmed",
                "eligible": True,
                "region_start": dates[10],
                "region_end": dates[11],
                "anchor_date": dates[10],
                "lobe_count": 1,
                "label_version": DEFAULT_REGION_PROTOCOL.label_version,
            },
            {
                "region_id": "test_top_2",
                "index_id": "test_index",
                "index_name": "测试指数",
                "event_type": "top",
                "status": "confirmed",
                "eligible": True,
                "region_start": dates[30],
                "region_end": dates[32],
                "anchor_date": dates[31],
                "lobe_count": 1,
                "label_version": DEFAULT_REGION_PROTOCOL.label_version,
            },
        ]
    )
    lobes = pd.DataFrame(
        [
            {
                "region_id": "test_top_1",
                "lobe_id": "test_top_1_lobe_1",
                "lobe_start": dates[10],
                "lobe_end": dates[11],
            },
            {
                "region_id": "test_top_2",
                "lobe_id": "test_top_2_lobe_1",
                "lobe_start": dates[30],
                "lobe_end": dates[32],
            },
        ]
    )
    regions_path = region_dir / "turning_point_regions.csv"
    lobes_path = region_dir / "turning_point_region_lobes.csv"
    regions.to_csv(regions_path, index=False)
    lobes.to_csv(lobes_path, index=False)
    manifest = {
        "label_version": DEFAULT_REGION_PROTOCOL.label_version,
        "protocol": DEFAULT_REGION_PROTOCOL.to_dict(),
        "source_files": [
            {
                "index_id": "test_index",
                "path": source_relative,
                "sha256": _sha256(source_path),
            }
        ],
        "outputs": [
            {"path": regions_path.name, "sha256": _sha256(regions_path)},
            {"path": lobes_path.name, "sha256": _sha256(lobes_path)},
        ],
    }
    (region_dir / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _inputs(tmp_path, monkeypatch):
    dates = pd.bdate_range("2020-01-01", periods=80)
    vipdoc = tmp_path / "vipdoc"
    source_relative = "sh/lday/test.day"
    source_path = vipdoc / source_relative
    _write_standard(source_path, dates)
    signal_daily, signal_episodes = _write_signal_bundle(
        tmp_path / "signal", dates
    )
    ground_truth = tmp_path / "ground_truth"
    _write_ground_truth(ground_truth, dates, source_path, source_relative)
    monkeypatch.setattr(
        pipeline,
        "INDEX_SPECS",
        (("test_index", "测试指数", "TEST", source_relative, False),),
    )
    return signal_daily, signal_episodes, ground_truth, vipdoc


def test_pipeline_writes_two_reports_csvs_and_auditable_manifest(
    tmp_path, monkeypatch
):
    signal_daily, signal_episodes, ground_truth, vipdoc = _inputs(
        tmp_path, monkeypatch
    )
    output_dir = tmp_path / "evaluation"

    outputs = pipeline.run_pipeline(
        signal_daily,
        signal_episodes,
        ground_truth,
        vipdoc,
        output_dir,
        evaluation_version="evaluation_v1",
        min_event_count=1,
        min_baseline_count=5,
    )

    assert set(outputs) == {
        "region_matches",
        "region_metrics",
        "region_report",
        "forward_event_outcomes",
        "forward_metrics",
        "forward_report",
        "manifest",
    }
    assert all(path.exists() for path in outputs.values())
    region_matches = pd.read_csv(outputs["region_matches"])
    region_metrics = pd.read_csv(outputs["region_metrics"])
    forward_outcomes = pd.read_csv(outputs["forward_event_outcomes"])
    forward_metrics = pd.read_csv(outputs["forward_metrics"])
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    assert set(region_matches["event_kind"]) == {
        "onset",
        "capped_confirmation",
    }
    assert set(region_metrics["aggregation"]) == {"index", "all_indices"}
    assert set(region_metrics["timing_slice"]) == {
        "all",
        "prediction",
        "confirmation",
    }
    assert set(forward_outcomes["horizon"]) == {5, 10, 20}
    assert "outcome_name" not in forward_outcomes
    assert set(forward_metrics["outcome_name"]) == {
        "terminal_return",
        "max_up",
        "max_down",
    }
    assert manifest["evaluation_version"] == "evaluation_v1"
    assert manifest["composite_score"] is None
    assert manifest["protocol"]["outcomes"]["max_up"].startswith("max(high")
    assert manifest["inputs"]["tdx_ohlc"][0]["sha256"] == _sha256(
        vipdoc / "sh/lday/test.day"
    )
    for record in manifest["outputs"]:
        assert record["sha256"] == _sha256(output_dir / record["path"])
    assert "不与信号后价格结果合成总分" in outputs["region_report"].read_text(
        encoding="utf-8"
    )
    assert "不构成交易回测" in outputs["forward_report"].read_text(encoding="utf-8")


def test_pipeline_rejects_nonempty_output_directory(tmp_path, monkeypatch):
    signal_daily, signal_episodes, ground_truth, vipdoc = _inputs(
        tmp_path, monkeypatch
    )
    output_dir = tmp_path / "evaluation"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not overwrite", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        pipeline.run_pipeline(
            signal_daily,
            signal_episodes,
            ground_truth,
            vipdoc,
            output_dir,
            evaluation_version="evaluation_v1",
        )


def test_pipeline_rejects_ohlc_changed_after_ground_truth(tmp_path, monkeypatch):
    signal_daily, signal_episodes, ground_truth, vipdoc = _inputs(
        tmp_path, monkeypatch
    )
    source = vipdoc / "sh/lday/test.day"
    source.write_bytes(source.read_bytes() + source.read_bytes()[-32:])

    with pytest.raises(ValueError, match="TDX source hash mismatch"):
        pipeline.run_pipeline(
            signal_daily,
            signal_episodes,
            ground_truth,
            vipdoc,
            tmp_path / "evaluation",
            evaluation_version="evaluation_v1",
        )


def test_pipeline_rejects_confirmation_dates_inconsistent_with_episode_file(
    tmp_path, monkeypatch
):
    signal_daily, signal_episodes, ground_truth, vipdoc = _inputs(
        tmp_path, monkeypatch
    )
    daily = pd.read_csv(signal_daily)
    first_confirmation = daily.index[daily["event_capped_confirmation"]].tolist()[0]
    daily.loc[first_confirmation, "event_capped_confirmation"] = False
    daily.to_csv(signal_daily, index=False)

    with pytest.raises(ValueError, match="capped confirmations do not match"):
        pipeline.run_pipeline(
            signal_daily,
            signal_episodes,
            ground_truth,
            vipdoc,
            tmp_path / "evaluation",
            evaluation_version="evaluation_v1",
        )
