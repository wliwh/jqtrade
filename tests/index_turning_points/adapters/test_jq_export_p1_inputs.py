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
    jqdata.get_valuation = lambda *args, **kwargs: None
    sys.modules["jqdata"] = jqdata

    module_name = "research.index_turning_points.adapters.jq.export_all_a_p1_inputs"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class JQP1InputsExportTests(unittest.TestCase):
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
        self.assertEqual(self.module.DATA_VERSION, "all_a_p1_inputs_v2")

    def test_price_features_are_causal_and_include_current_day(self):
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        close = pd.DataFrame({"a": [1.0, 2.0, 3.0, 2.0, 4.0]}, index=dates)
        high = close + 0.5
        low = close - 0.5

        with mock.patch.object(self.module, "MA_WINDOWS", (3,)), mock.patch.object(
            self.module,
            "HIGH_LOW_WINDOWS",
            (3,),
        ):
            full = self.module._price_feature_matrices(close, high, low)
            truncated = self.module._price_feature_matrices(
                close.iloc[:4],
                high.iloc[:4],
                low.iloc[:4],
            )

        cutoff = dates[3]
        for group in full:
            self.assertEqual(
                bool(full[group][3].loc[cutoff, "a"]),
                bool(truncated[group][3].loc[cutoff, "a"]),
            )
        self.assertTrue(full["new_high"][3].loc[dates[2], "a"])
        self.assertTrue(full["new_low"][3].loc[dates[3], "a"])

    def test_ma_breadth_is_invariant_to_extra_warmup_history(self):
        rng = np.random.RandomState(7)
        values = None
        for _ in range(10):
            changes = rng.choice(
                [-0.01, 0.0, 0.01],
                size=520,
                p=[0.2, 0.6, 0.2],
            )
            values = np.round(10.0 + np.cumsum(changes), 2)

        dates = pd.date_range("2018-01-01", periods=len(values), freq="D")
        close = pd.DataFrame({"a": values}, index=dates)
        high = close + 0.01
        low = close - 0.01
        long_history = self.module._price_feature_matrices(close, high, low)
        short_history = self.module._price_feature_matrices(
            close.iloc[130:],
            high.iloc[130:],
            low.iloc[130:],
        )

        long_signal = long_history["above_ma"][20].iloc[130:, 0]
        short_signal = short_history["above_ma"][20].iloc[:, 0]
        np.testing.assert_array_equal(long_signal.values, short_signal.values)

    def test_aggregate_day_covers_all_p1_feature_families(self):
        universe = ["up", "down", "paused", "st", "missing"]
        prices = {
            "close": pd.Series(
                {"up": 11.0, "down": 8.0, "paused": 10.0, "st": 10.0}
            ),
            "high": pd.Series(
                {"up": 11.0, "down": 9.0, "paused": 10.0, "st": 10.0}
            ),
            "low": pd.Series(
                {"up": 10.0, "down": 8.0, "paused": 10.0, "st": 10.0}
            ),
            "high_limit": pd.Series(11.0, index=universe),
            "low_limit": pd.Series(8.0, index=universe),
            "paused": pd.Series(
                {"up": 0.0, "down": 0.0, "paused": 1.0, "st": 0.0, "missing": 0.0}
            ),
        }
        price_features = {
            "ma_complete": {},
            "above_ma": {},
            "high_low_complete": {},
            "new_high": {},
            "new_low": {},
        }
        for window in self.module.MA_WINDOWS:
            price_features["ma_complete"][window] = pd.Series(True, index=universe)
            price_features["above_ma"][window] = pd.Series(
                {"up": True, "down": False},
                index=universe,
            )
        for window in self.module.HIGH_LOW_WINDOWS:
            price_features["high_low_complete"][window] = pd.Series(
                True,
                index=universe,
            )
            price_features["new_high"][window] = pd.Series(
                {"up": True, "down": False},
                index=universe,
            )
            price_features["new_low"][window] = pd.Series(
                {"up": False, "down": True},
                index=universe,
            )

        summary, industry = self.module.aggregate_day(
            date="2020-01-02",
            universe=universe,
            prices=prices,
            is_st=pd.Series(
                {
                    "up": False,
                    "down": False,
                    "paused": False,
                    "st": True,
                    "missing": False,
                }
            ),
            price_features=price_features,
            valuations={
                "turnover_ratio": pd.Series({"up": 10.0, "down": 20.0}),
                "circulating_market_cap": pd.Series({"up": 100.0, "down": 300.0}),
            },
            industries={
                "up": ("801780", "银行I"),
                "down": ("801750", "计算机I"),
            },
            min_industry_valid_count=1,
        )

        self.assertEqual(summary["base_valid_count"], 2)
        self.assertEqual(summary["valid_count_ma20"], 2)
        self.assertEqual(summary["above_count_ma20"], 1)
        self.assertEqual(summary["new_high_count_60"], 1)
        self.assertEqual(summary["new_low_count_60"], 1)
        self.assertEqual(summary["limit_up_hit_count"], 1)
        self.assertEqual(summary["limit_down_hit_count"], 1)
        self.assertEqual(summary["limit_up_close_count"], 1)
        self.assertEqual(summary["limit_down_close_count"], 1)
        self.assertEqual(summary["turnover_valid_count"], 2)
        self.assertEqual(summary["turnover_ratio_pct_p50"], 15.0)
        self.assertEqual(
            summary["turnover_ratio_pct_cap_weighted_mean"],
            17.5,
        )
        self.assertEqual(summary["turnover_ge_20pct_count"], 1)
        self.assertEqual(set(industry["industry_name"]), {"银行I", "计算机I"})

    def test_price_query_uses_one_adjusted_daily_request_for_all_fields(self):
        calls = []

        def fake_get_price(securities, **kwargs):
            calls.append((securities, kwargs))
            rows = []
            for number, security in enumerate(securities, start=1):
                rows.append(
                    {
                        "time": "2020-01-02",
                        "code": security,
                        "close": 10.0 + number,
                        "high": 11.0 + number,
                        "low": 9.0 + number,
                        "high_limit": 12.0 + number,
                        "low_limit": 8.0 + number,
                        "paused": 0,
                    }
                )
            return pd.DataFrame(rows)

        with mock.patch.object(self.module, "get_price", fake_get_price):
            matrices = self.module._query_price_matrices(
                ["a", "b"],
                "2020-01-02",
                "2020-01-02",
                ["2020-01-02"],
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["fields"], list(self.module.PRICE_FIELDS))
        self.assertEqual(calls[0][1]["fq"], "pre")
        self.assertEqual(calls[0][1]["frequency"], "daily")
        self.assertFalse(calls[0][1]["skip_paused"])
        self.assertTrue(calls[0][1]["fill_paused"])
        self.assertEqual(set(matrices), set(self.module.PRICE_FIELDS))

    def test_valuation_query_stays_below_documented_row_limit(self):
        calls = []
        trade_days = pd.to_datetime(["2020-01-02", "2020-01-03"])

        def fake_get_valuation(securities, **kwargs):
            calls.append((list(securities), kwargs))
            rows = []
            for day in trade_days:
                for security in securities:
                    rows.append(
                        {
                            "day": day,
                            "code": security,
                            "turnover_ratio": 2.0,
                            "circulating_market_cap": 100.0,
                        }
                    )
            return pd.DataFrame(rows)

        with mock.patch.object(
            self.module,
            "VALUATION_MAX_ROWS_PER_QUERY",
            4,
        ), mock.patch.object(
            self.module,
            "get_valuation",
            fake_get_valuation,
        ):
            matrices = self.module._query_valuation_matrices(
                ["a", "b", "c", "d", "e"],
                trade_days[0],
                trade_days[-1],
                trade_days,
            )

        self.assertEqual([len(call[0]) for call in calls], [2, 2, 1])
        for _, kwargs in calls:
            self.assertEqual(kwargs["fields"], list(self.module.VALUATION_FIELDS))
        self.assertEqual(matrices["turnover_ratio"].shape, (2, 5))

    def test_trade_day_chunk_runs_all_collectors_into_daily_and_industry_rows(self):
        output_days = pd.to_datetime(["2020-01-03", "2020-01-04"])
        price_days = pd.to_datetime(
            ["2020-01-02", "2020-01-03", "2020-01-04"]
        )
        securities = ["a", "b"]

        def fake_trade_days(**kwargs):
            if "count" in kwargs:
                return price_days[:2]
            return price_days

        def fake_price(requested, **kwargs):
            rows = []
            for day_number, day in enumerate(price_days, start=1):
                for security_number, security in enumerate(requested, start=1):
                    close = float(day_number + security_number + 8)
                    rows.append(
                        {
                            "time": day,
                            "code": security,
                            "close": close,
                            "high": close + 0.5,
                            "low": close - 0.5,
                            "high_limit": close + 1.0,
                            "low_limit": close - 1.0,
                            "paused": 0,
                        }
                    )
            return pd.DataFrame(rows)

        def fake_st(field, requested, **kwargs):
            self.assertEqual(field, "is_st")
            return pd.DataFrame(False, index=output_days, columns=requested)

        def fake_valuation(requested, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "day": day,
                        "code": security,
                        "turnover_ratio": 3.0,
                        "circulating_market_cap": 100.0,
                    }
                    for day in output_days
                    for security in requested
                ]
            )

        def fake_industry(requested, **kwargs):
            return {
                security: {
                    "sw_l1": {
                        "industry_code": "801780" if security == "a" else "801750",
                        "industry_name": "银行I" if security == "a" else "计算机I",
                    }
                }
                for security in requested
            }

        with mock.patch.object(
            self.module,
            "MA_WINDOWS",
            (2, 20),
        ), mock.patch.object(
            self.module,
            "HIGH_LOW_WINDOWS",
            (2,),
        ), mock.patch.object(
            self.module,
            "get_trade_days",
            fake_trade_days,
        ), mock.patch.object(
            self.module,
            "get_index_stocks",
            lambda *args, **kwargs: securities,
        ), mock.patch.object(
            self.module,
            "get_price",
            fake_price,
        ), mock.patch.object(
            self.module,
            "get_extras",
            fake_st,
        ), mock.patch.object(
            self.module,
            "get_valuation",
            fake_valuation,
        ), mock.patch.object(
            self.module,
            "get_industry",
            fake_industry,
        ):
            daily, industry = self.module._process_trade_day_chunk(output_days)
            self.module._validate_outputs(daily, industry, output_days)

        self.assertEqual(list(daily["date"]), ["2020-01-03", "2020-01-04"])
        self.assertEqual(list(daily["base_valid_count"]), [2, 2])
        self.assertEqual(list(daily["turnover_valid_count"]), [2, 2])
        self.assertEqual(len(industry), 4)

    def test_archive_contains_manifest_and_two_aggregated_tables(self):
        daily = pd.DataFrame({"date": ["2020-01-02"], "breadth_ma20": [0.5]})
        industry = pd.DataFrame(
            {
                "date": ["2020-01-02"],
                "industry_code": ["801780"],
                "industry_name": ["银行I"],
                "breadth_ma20": [0.5],
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
                    "data/daily_market_features.csv",
                    "data/industry_breadth.csv",
                    "manifest.json",
                ],
            )
            manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertEqual(manifest["data_version"], "all_a_p1_inputs_v2")
            self.assertFalse(manifest["export_level"]["stock_level_rows_exported"])
            self.assertEqual(manifest["query"]["high_low_windows"], [60, 120, 250])
            self.assertEqual(
                manifest["query"]["ma_comparison_relative_tolerance"],
                1e-12,
            )
            for record in manifest["files"]:
                content = archive.read(record["path"])
                self.assertEqual(hashlib.sha256(content).hexdigest(), record["sha256"])


if __name__ == "__main__":
    unittest.main()
