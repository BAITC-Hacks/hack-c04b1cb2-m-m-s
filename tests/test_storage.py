"""A failed write must not destroy the previous forecast artifact."""
import io
import json
import pathlib
import tempfile
import unittest
from contextlib import redirect_stderr
from unittest import mock

import run_forecast
import storage

ROOT = pathlib.Path(__file__).resolve().parents[1]


class StorageTests(unittest.TestCase):
    def test_partial_write_preserves_previous_file_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            destination = root / "forecast.json"
            destination.write_bytes(b"previous valid forecast")
            original = storage.tempfile.NamedTemporaryFile

            def failing_stream(*args, **kwargs):
                stream = original(*args, **kwargs)
                class PartialWrite:
                    name = stream.name
                    def __enter__(self):
                        return self
                    def __exit__(self, *exc):
                        stream.close()
                    def write(self, text):
                        stream.write(text[:80])
                        raise OSError(28, "simulated disk full")
                return PartialWrite()

            with mock.patch.object(storage.tempfile, "NamedTemporaryFile", side_effect=failing_stream):
                with self.assertRaises(OSError):
                    storage.atomic_write(destination, "new forecast" * 30)
            self.assertEqual(destination.read_bytes(), b"previous valid forecast")
            self.assertEqual(list(root.iterdir()), [destination])
            storage.atomic_write(destination, "recovered forecast")
            self.assertEqual(destination.read_text(), "recovered forecast")

    def test_cli_failed_replace_keeps_previous_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            expected = (ROOT / "results/2026-02-01-forecast.json").read_bytes()
            destination = root / "2026-02-01-forecast.json"
            destination.write_bytes(expected)
            with mock.patch("sys.argv", ["run_forecast.py", "2026-02-01", "--model",
                                         str(ROOT / "models/power-curve-2026-01-31.json"),
                                         "--output-dir", str(root)]), \
                    mock.patch.object(run_forecast, "retrieve", return_value={}), \
                    mock.patch.object(run_forecast, "combine", return_value=json.loads(expected)), \
                    mock.patch.object(storage.os, "replace", side_effect=OSError(28, "disk full")), \
                    redirect_stderr(io.StringIO()):
                self.assertEqual(run_forecast.main(), 1)
            self.assertEqual(destination.read_bytes(), expected)
            self.assertEqual(list(root.iterdir()), [destination])


if __name__ == "__main__":
    unittest.main()
