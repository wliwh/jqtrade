import ast
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import types
import unittest
import zipfile
from unittest import mock

import numpy as np
import pandas as pd


def _load_module():
    jqdata = types.ModuleType("jqdata")
    jqdata.get_extras = lambda *args, **kwargs: None
    jqdata.get_index_stocks = lambda *args, **kwargs: None
    jqdata.get_industry = lambda *args, **kwargs: None
    jqdata.get_price = lambda *args, **kwargs: None
    jqdata.get_trade_days = lambda *args, **kwargs: None
    sys.modules["jqdata"] = jqdata

    module_name = (
        "research.index_turning_points.datas."
        "all_a_breadth_v1_20120101_20260814.jq_export_breadth"
    )
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class JQBreadthExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def test_source_is_compatible_with_jq_python36_and_old_pandas(self):
        source = Path(self.module.__file__).read_text(encoding="utf-8")
        ast.parse(source, feature_version=(3, 6))
        self.assertNotIn("from __future__ import", source)
        self.assertIn("from jqdata import *", source)
        self.assertNotIn(".to_numpy(", source)
        self.assertNotIn('astype("boolean")', source)
        self.assertNotIn("dropna=False", source)
        self.assertNotIn("lineterminator=", source)
        self.assertEqual(self.module.START_DATE, "2012-01-01")

    def test_aggregate_day_keeps_all_top1_ties_and_target_trigger(self):
        industries = {}
        universe = []
        definitions = (
            ("bank", "801780", "银行I", (9.0, 9.0)),
            ("tech", "801750", "计算机I", (9.0, 9.0)),
            ("coal", "801950", "煤炭I", (9.0, 11.0)),
            ("steel", "801040", "钢铁I", (9.0, 11.0)),
            ("metal", "801050", "有色金属I", (9.0, 11.0)),
        )
        ma20 = {}
        for prefix, industry_code, industry_name, values in definitions:
            for number, moving_average in enumerate(values, start=1):
                security = "%s%d" % (prefix, number)
                universe.append(security)
                industries[security] = (industry_code, industry_name)
                ma20[security] = moving_average

        close = pd.Series(10.0, index=universe)
        moving_averages = {
            20: pd.Series(ma20),
            60: pd.Series(9.0, index=universe),
            120: pd.Series(9.0, index=universe),
        }
        summary, industry = self.module.aggregate_day(
            date="2020-01-02",
            universe=universe,
            close=close,
            paused=pd.Series(0.0, index=universe),
            is_st=pd.Series(False, index=universe),
            moving_averages=moving_averages,
            industries=industries,
            min_industry_valid_count=2,
        )

        top1_names = set(industry.loc[industry["is_top1_ma20"], "industry_name"])
        self.assertEqual(top1_names, {"银行I", "计算机I"})
        self.assertEqual(summary["top1_tie_count_ma20"], 2)
        self.assertTrue(summary["four_industry_top1_triggered"])
        self.assertEqual(summary["four_industry_top1_ids"], "bank")
        self.assertEqual(summary["target_bank_rank_ma20"], 1.0)
        self.assertEqual(
            industry.loc[industry["industry_name"] == "煤炭I", "rank_ma20"].iloc[0],
            3.0,
        )

    def test_aggregate_day_reports_exclusions_and_minimum_industry_size(self):
        universe = ["valid", "paused", "missing", "st"]
        summary, industry = self.module.aggregate_day(
            date="2020-01-02",
            universe=universe,
            close=pd.Series(
                {"valid": 10.0, "paused": 10.0, "missing": np.nan, "st": 10.0}
            ),
            paused=pd.Series({"valid": 0.0, "paused": 1.0, "missing": 0.0, "st": 0.0}),
            is_st=pd.Series({"valid": False, "paused": False, "missing": False, "st": True}),
            moving_averages={
                window: pd.Series(9.0, index=universe)
                for window in self.module.MA_WINDOWS
            },
            industries={
                "valid": ("801780", "银行I"),
                "paused": ("801780", "银行I"),
                "missing": (None, None),
                "st": ("801040", "钢铁I"),
            },
            min_industry_valid_count=2,
        )

        self.assertEqual(summary["universe_size"], 4)
        self.assertEqual(summary["close_missing_count"], 1)
        self.assertEqual(summary["paused_count"], 1)
        self.assertEqual(summary["st_count"], 1)
        self.assertEqual(summary["base_valid_count"], 1)
        self.assertEqual(summary["valid_count_ma20"], 1)
        self.assertFalse(summary["four_industry_top1_triggered"])
        self.assertFalse(industry["rank_eligible_ma20"].any())

    def test_price_query_is_adjusted_daily_and_keeps_paused_rows(self):
        calls = []

        def fake_get_price(securities, **kwargs):
            calls.append((securities, kwargs))
            return pd.DataFrame(
                {
                    "time": ["2020-01-02", "2020-01-02"],
                    "code": securities,
                    "close": [10.0, 20.0],
                    "paused": [0, 1],
                }
            )

        with mock.patch.object(self.module, "get_price", fake_get_price):
            close, paused = self.module._query_price_matrices(
                ["a", "b"],
                "2020-01-02",
                "2020-01-02",
                ["2020-01-02"],
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["frequency"], "daily")
        self.assertEqual(calls[0][1]["fq"], "pre")
        self.assertFalse(calls[0][1]["skip_paused"])
        self.assertTrue(calls[0][1]["fill_paused"])
        self.assertEqual(list(close.columns), ["a", "b"])
        self.assertEqual(paused.loc[pd.Timestamp("2020-01-02"), "b"], 1.0)

    def test_archive_contains_only_manifest_and_two_processed_tables(self):
        daily = pd.DataFrame(
            {
                "date": ["2020-01-02"],
                "breadth_ma20": [0.5],
                "breadth_ma60": [0.4],
                "breadth_ma120": [0.3],
            }
        )
        industry = pd.DataFrame(
            {
                "date": ["2020-01-02"],
                "industry_code": ["801780"],
                "industry_name": ["银行I"],
                "breadth_ma20": [0.5],
                "breadth_ma60": [0.4],
                "breadth_ma120": [0.3],
            }
        )
        payload = self.module.build_archive(
            daily,
            industry,
            "2020-01-02",
            "2020-01-02",
        )

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(
                sorted(archive.namelist()),
                [
                    "data/daily_summary.csv",
                    "data/industry_breadth.csv",
                    "manifest.json",
                ],
            )
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertFalse(manifest["export_level"]["stock_level_rows_exported"])
            for record in manifest["files"]:
                content = archive.read(record["path"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
