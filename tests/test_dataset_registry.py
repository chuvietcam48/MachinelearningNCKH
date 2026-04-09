"""
tests/test_dataset_registry.py
================================
Unit tests for src/dataset_registry.py:
  - register_dataset / get_dataset round-trip
  - KeyError for unknown dataset
  - list_datasets() structure
  - get_currency_symbol() / get_currency_code() fallbacks
  - Default snapshot function (+1 day)
"""
import unittest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset_registry import (
    register_dataset, get_dataset, list_datasets,
    get_currency_symbol, get_currency_code,
    _default_snapshot, _REGISTRY,
)


def _dummy_loader(path: str) -> pd.DataFrame:
    return pd.DataFrame({"CustomerID": ["X"], "InvoiceDate": [pd.Timestamp("2023-01-01")]})


class TestRegisterAndGet(unittest.TestCase):

    def setUp(self):
        """Remove any leftover test dataset from registry."""
        _REGISTRY.pop("__test_ds__", None)

    def tearDown(self):
        _REGISTRY.pop("__test_ds__", None)

    def test_register_and_retrieve(self):
        """Registered dataset must be retrievable by name."""
        register_dataset(
            name="__test_ds__",
            display="Test Dataset",
            data_path="/tmp/fake.csv",
            loader_fn=_dummy_loader,
        )
        info = get_dataset("__test_ds__")
        self.assertEqual(info.name, "__test_ds__")
        self.assertEqual(info.display, "Test Dataset")
        self.assertEqual(info.data_path, "/tmp/fake.csv")

    def test_loader_fn_callable(self):
        """loader_fn must be callable and return a DataFrame."""
        register_dataset(
            name="__test_ds__",
            display="Test Dataset",
            data_path="/tmp/fake.csv",
            loader_fn=_dummy_loader,
        )
        info = get_dataset("__test_ds__")
        result = info.loader_fn(info.data_path)
        self.assertIsInstance(result, pd.DataFrame)

    def test_default_snapshot_fn_assigned(self):
        """When snapshot_fn is None, default (+1 day) should be used."""
        register_dataset(
            name="__test_ds__",
            display="Test Dataset",
            data_path="/tmp/fake.csv",
            loader_fn=_dummy_loader,
            snapshot_fn=None,
        )
        info = get_dataset("__test_ds__")
        self.assertEqual(info.snapshot_fn, _default_snapshot)


class TestGetUnknownRaises(unittest.TestCase):

    def test_keyerror_for_unknown_name(self):
        """get_dataset with unknown name must raise KeyError."""
        with self.assertRaises(KeyError):
            get_dataset("__does_not_exist__")

    def test_error_message_lists_available(self):
        """Error message must mention available datasets."""
        try:
            get_dataset("__does_not_exist__")
        except KeyError as e:
            self.assertIn("Available", str(e))


class TestListDatasets(unittest.TestCase):

    def test_returns_list(self):
        """list_datasets() must return a list."""
        result = list_datasets()
        self.assertIsInstance(result, list)

    def test_each_item_is_tuple_of_two_strings(self):
        """Each item must be (name: str, display: str)."""
        result = list_datasets()
        for item in result:
            self.assertEqual(len(item), 2)
            self.assertIsInstance(item[0], str)
            self.assertIsInstance(item[1], str)

    def test_builtin_datasets_registered(self):
        """UCI, TaFeng, CDNOW must be in the registry."""
        names = [name for name, _ in list_datasets()]
        for expected in ("uci", "tafeng", "cdnow", "x5retail"):
            self.assertIn(expected, names)


class TestCurrencyHelpers(unittest.TestCase):

    def test_currency_symbol_returns_string(self):
        """get_currency_symbol() must return a non-empty string."""
        symbol = get_currency_symbol()
        self.assertIsInstance(symbol, str)
        self.assertGreater(len(symbol), 0)

    def test_currency_code_returns_string(self):
        """get_currency_code() must return a non-empty string."""
        code = get_currency_code()
        self.assertIsInstance(code, str)
        self.assertGreater(len(code), 0)

    def test_currency_code_is_uppercase(self):
        """Currency code should be uppercase (e.g. 'GBP', 'USD')."""
        code = get_currency_code()
        self.assertEqual(code, code.upper())


class TestDefaultSnapshot(unittest.TestCase):

    def test_snapshot_is_max_plus_one_day(self):
        """Default snapshot = max(InvoiceDate) + 1 day."""
        df = pd.DataFrame({
            "InvoiceDate": pd.to_datetime(["2023-01-01", "2023-06-15", "2022-12-31"])
        })
        snap = _default_snapshot(df)
        self.assertEqual(snap, pd.Timestamp("2023-06-16"))

    def test_snapshot_returns_timestamp(self):
        """Snapshot must be a pd.Timestamp."""
        df = pd.DataFrame({"InvoiceDate": pd.to_datetime(["2023-01-01"])})
        snap = _default_snapshot(df)
        self.assertIsInstance(snap, pd.Timestamp)


if __name__ == "__main__":
    unittest.main()
