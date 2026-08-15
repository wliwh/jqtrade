import importlib
import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from unittest import mock


def _load_module():
    jqdata = types.ModuleType("jqdata")
    jqdata.get_all_securities = lambda **kwargs: None
    jqdata.get_price = lambda **kwargs: None
    sys.modules["jqdata"] = jqdata

    module_name = "tools.jq_data_export.src.download_fund_minute_data_batches"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _universe():
    return [
        {
            "security": "159001.XSHE",
            "display_name": "基金一",
            "name": "JJY",
            "fund_type": "etf",
            "start_date": "2010-01-01",
            "end_date": "2200-01-01",
        },
        {
            "security": "160001.XSHE",
            "display_name": "基金二",
            "name": "JJE",
            "fund_type": "lof",
            "start_date": "2011-01-01",
            "end_date": "2200-01-01",
        },
        {
            "security": "510001.XSHG",
            "display_name": "基金三",
            "name": "JJS",
            "fund_type": "etf",
            "start_date": "2012-01-01",
            "end_date": "2200-01-01",
        },
    ]


class BatchDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temporary_directory.name)
        self.module = _load_module()

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary_directory.cleanup()

    def _read_state(self, output_dir):
        with open(os.path.join(output_dir, "state.json"), encoding="utf-8") as file:
            return json.load(file)

    def test_batch_archive_pause_cleanup_and_resume(self):
        universe_calls = []

        def fake_universe(fund_types):
            universe_calls.append(tuple(fund_types))
            return _universe()

        download_calls = []

        def fake_download(**kwargs):
            download_calls.append(kwargs["security"])
            os.makedirs(kwargs["output_dir"], exist_ok=True)
            filename = "%s.csv" % kwargs["security"].replace(".", "_")
            path = os.path.join(kwargs["output_dir"], filename)
            with open(path, "wb") as file:
                file.write((kwargs["security"] + "\n").encode("utf-8") * 70)
            return path, 70

        run_kwargs = {
            "start_date": None,
            "end_date": "2020-12-31",
            "output_dir": "out",
            "batch_target_mb": 0.001,
            "chunk_days": 60,
            "fund_types": ("etf", "lof"),
        }

        with mock.patch.object(self.module, "_build_universe", fake_universe), mock.patch.object(
            self.module, "_download_security_minute_data", fake_download
        ):
            self.assertEqual(self.module.run_batch_download(**run_kwargs), "archive_created")
            state = self._read_state("out")
            first_archive = state["pending_archive"]["path"]
            self.assertTrue(os.path.exists(first_archive))
            self.assertEqual(state["next_index"], 2)
            self.assertEqual(download_calls, ["159001.XSHE", "160001.XSHE"])
            self.assertFalse(os.path.exists("out/work/batch_0001"))

            with zipfile.ZipFile(first_archive) as archive:
                self.assertIsNone(archive.testzip())
                self.assertEqual(
                    sorted(archive.namelist()),
                    [
                        "data/159001_XSHE.csv",
                        "data/160001_XSHE.csv",
                        "manifest.json",
                    ],
                )
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
            self.assertIsNone(manifest["query"]["fq"])
            self.assertEqual(manifest["query"]["end_date"], "2020-12-31")
            self.assertEqual(
                [item["fund_type"] for item in manifest["files"]],
                ["etf", "lof"],
            )

            self.assertEqual(
                self.module.run_batch_download(**run_kwargs),
                "awaiting_archive_cleanup",
            )
            self.assertEqual(download_calls, ["159001.XSHE", "160001.XSHE"])

            os.remove(first_archive)
            self.assertEqual(self.module.run_batch_download(**run_kwargs), "archive_created")
            state = self._read_state("out")
            final_archive = state["pending_archive"]["path"]
            self.assertTrue(os.path.exists(final_archive))
            self.assertTrue(state["pending_archive"]["final_batch"])
            self.assertEqual(state["next_index"], 3)
            self.assertEqual(download_calls[-1], "510001.XSHG")

            os.remove(final_archive)
            self.assertEqual(self.module.run_batch_download(**run_kwargs), "completed")
            state = self._read_state("out")
            self.assertEqual(state["status"], "completed")
            self.assertIsNone(state["pending_archive"])
            self.assertEqual(len(state["released_archives"]), 2)
            self.assertEqual(universe_calls, [("etf", "lof")])

    def test_failed_security_is_retried_without_advancing_state(self):
        attempts = []

        def failing_download(**kwargs):
            attempts.append(kwargs["security"])
            raise OSError("simulated JQ error")

        run_kwargs = {
            "end_date": "2020-12-31",
            "output_dir": "retry",
            "batch_target_mb": 1,
        }
        with mock.patch.object(self.module, "_build_universe", lambda fund_types: _universe()[:1]), mock.patch.object(
            self.module, "_download_security_minute_data", failing_download
        ):
            with self.assertRaisesRegex(RuntimeError, "159001.XSHE"):
                self.module.run_batch_download(**run_kwargs)

        state = self._read_state("retry")
        self.assertEqual(state["next_index"], 0)
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["failures"]["159001.XSHE"]["attempts"], 1)

        def successful_download(**kwargs):
            attempts.append(kwargs["security"])
            os.makedirs(kwargs["output_dir"], exist_ok=True)
            path = os.path.join(kwargs["output_dir"], "159001_XSHE.csv")
            with open(path, "wb") as file:
                file.write(b"minute-data")
            return path, 1

        with mock.patch.object(self.module, "_build_universe", lambda fund_types: _universe()[:1]), mock.patch.object(
            self.module, "_download_security_minute_data", successful_download
        ):
            self.assertEqual(self.module.run_batch_download(**run_kwargs), "archive_created")
        state = self._read_state("retry")
        self.assertEqual(state["next_index"], 1)
        self.assertNotIn("159001.XSHE", state["failures"])
        self.assertEqual(attempts, ["159001.XSHE", "159001.XSHE"])

    def test_resume_rejects_changed_query_settings(self):
        def fake_download(**kwargs):
            os.makedirs(kwargs["output_dir"], exist_ok=True)
            path = os.path.join(kwargs["output_dir"], "data.csv")
            with open(path, "wb") as file:
                file.write(b"x")
            return path, 1

        with mock.patch.object(self.module, "_build_universe", lambda fund_types: _universe()[:1]), mock.patch.object(
            self.module, "_download_security_minute_data", fake_download
        ):
            self.module.run_batch_download(
                end_date="2020-12-31",
                output_dir="fixed",
                batch_target_mb=1,
            )
            with self.assertRaisesRegex(ValueError, "状态文件不一致"):
                self.module.run_batch_download(
                    end_date="2020-12-30",
                    output_dir="fixed",
                    batch_target_mb=1,
                )

    def test_build_universe_filters_etf_and_lof_and_sorts_codes(self):
        rows = {
            "160001.XSHE": {
                "display_name": "LOF",
                "name": "LOF",
                "start_date": "2010-01-01",
                "end_date": "2200-01-01",
                "type": "lof",
            },
            "150001.XSHE": {
                "display_name": "分级A",
                "name": "FJA",
                "start_date": "2010-01-01",
                "end_date": "2020-01-01",
                "type": "fja",
            },
            "510001.XSHG": {
                "display_name": "ETF",
                "name": "ETF",
                "start_date": "2011-01-01",
                "end_date": "2200-01-01",
                "type": "etf",
            },
        }

        class FakeSeries:
            def __init__(self, values):
                self.values = values

            def isin(self, allowed):
                return [value in allowed for value in self.values]

        class FakeFrame:
            columns = ["display_name", "name", "start_date", "end_date", "type"]

            def __init__(self, values):
                self.values = values
                self.empty = not bool(values)

            def __getitem__(self, key):
                if isinstance(key, str):
                    return FakeSeries([row[key] for row in self.values.values()])
                selected = {
                    code: row
                    for (code, row), include in zip(self.values.items(), key)
                    if include
                }
                return FakeFrame(selected)

            def sort_index(self):
                return FakeFrame(dict(sorted(self.values.items())))

            def iterrows(self):
                return iter(self.values.items())

        calls = []

        def fake_get_all_securities(**kwargs):
            calls.append(kwargs)
            return FakeFrame(rows)

        with mock.patch.object(self.module, "get_all_securities", fake_get_all_securities):
            universe = self.module._build_universe(["etf", "lof"])
        self.assertEqual(calls, [{"types": ["fund"], "date": None}])
        self.assertEqual(
            [item["security"] for item in universe],
            ["160001.XSHE", "510001.XSHG"],
        )
        self.assertEqual([item["fund_type"] for item in universe], ["lof", "etf"])

    def test_chunk_query_is_unadjusted_one_minute_data(self):
        calls = []

        class EmptyData:
            empty = True

        def fake_get_price(**kwargs):
            calls.append(kwargs)
            return EmptyData()

        with mock.patch.object(self.module, "get_price", fake_get_price):
            result = self.module._download_one_chunk(
                "159001.XSHE",
                self.module.dt.date(2020, 1, 1),
                self.module.dt.date(2020, 1, 31),
            )
        self.assertIsNone(result)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["frequency"], "1m")
        self.assertIsNone(calls[0]["fq"])
        self.assertFalse(calls[0]["skip_paused"])


if __name__ == "__main__":
    unittest.main()
