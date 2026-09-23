"""Offline profile validation and station-specific weather provenance tests."""
from __future__ import annotations

import copy
import datetime as dt
import io
import json
import math
import pathlib
import tempfile
import unittest
import urllib.parse
from unittest import mock

import forecast_weather
from station_profile import DEFAULT_STATION, load_station, station_digest, validate_station

DAY = dt.date(2026, 2, 1)
ROOT = pathlib.Path(__file__).resolve().parents[1]


def body_for(day: dt.date) -> dict:
    decision, _ = forecast_weather.run_for(day)
    times = [(decision - dt.timedelta(hours=6) + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
             for i in range(60)]
    return {"hourly_units": {"wind_speed_10m": "m/s", "wind_speed_100m": "m/s",
                             "temperature_2m": "°C"},
            "hourly": {"time": times, "wind_speed_10m": [6.0] * 60,
                       "wind_speed_100m": [8.0] * 60, "temperature_2m": [10.0] * 60}}


class StationProfileTests(unittest.TestCase):
    def test_default_and_example_are_explicit_and_digest_is_canonical(self):
        profile = load_station(ROOT / "stations/example.json")
        self.assertEqual(profile["id"], "example-wind-farm")
        self.assertEqual(profile["locations"], DEFAULT_STATION["locations"])
        self.assertEqual(set(profile), {"id", "name", "locations"})
        reordered = {"locations": {"turbine-2": tuple(profile["locations"]["turbine-2"]),
                                   "turbine-1": tuple(profile["locations"]["turbine-1"])},
                     "name": profile["name"], "id": profile["id"]}
        self.assertEqual(station_digest(profile), station_digest(reordered))
        changed = copy.deepcopy(profile)
        changed["locations"]["turbine-2"][1] += 0.001
        self.assertNotEqual(station_digest(profile), station_digest(changed))
        self.assertEqual(profile["locations"]["turbine-1"], list(forecast_weather.LOCATIONS["turbine-1"]))

    def test_rejects_extra_fields_missing_turbines_and_invalid_coordinates(self):
        invalid = []
        base = copy.deepcopy(DEFAULT_STATION)
        invalid.append({**base, "url": "https://example.invalid"})
        invalid.append({"id": "x", "name": "x"})
        for station_id in ("A", "", "-bad", "x_1", "x" * 65):
            item = copy.deepcopy(base)
            item["id"] = station_id
            invalid.append(item)
        for name in (" ", "x" * 121):
            item = copy.deepcopy(base)
            item["name"] = name
            invalid.append(item)
        for coordinates in ([float("nan"), 1], [float("inf"), 1], [True, 1],
                            [91, 1], [-91, 1], [1, 181], [1, -181], [1], "43,78"):
            item = copy.deepcopy(base)
            item["locations"]["turbine-1"] = coordinates
            invalid.append(item)
        missing = copy.deepcopy(base)
        del missing["locations"]["turbine-2"]
        invalid.append(missing)
        extra = copy.deepcopy(base)
        extra["locations"]["turbine-3"] = [43, 78]
        invalid.append(extra)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_station(item)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "station.json"
            path.write_text('{"id":')
            with self.assertRaises(ValueError):
                load_station(path)

    def test_name_and_numeric_values_normalize_without_mutating_input(self):
        value = copy.deepcopy(DEFAULT_STATION)
        value["name"] = "  Пример  "
        value["locations"]["turbine-1"] = [43, 78]
        normalized = validate_station(value)
        self.assertEqual(normalized["name"], "Пример")
        self.assertEqual(normalized["locations"]["turbine-1"], [43.0, 78.0])
        self.assertEqual(value["name"], "  Пример  ")
        self.assertEqual(value["locations"]["turbine-1"], [43, 78])


class StationWeatherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cache = pathlib.Path(self.temp.name)
        self.body = body_for(DAY)
        self.urls = []

        def response(url, timeout):
            self.urls.append(url)
            self.assertEqual(timeout, 30)
            return io.BytesIO(json.dumps(self.body).encode())

        self.fetch = mock.patch.object(forecast_weather.urllib.request, "urlopen", side_effect=response)
        self.fetch.start()
        self.addCleanup(self.fetch.stop)

    def test_custom_coordinates_drive_url_and_cache_validation(self):
        custom = {"turbine-1": [44.125, 79.875], "turbine-2": [44.126, 79.876]}
        first = forecast_weather.retrieve(DAY, "turbine-1", self.cache, locations=custom)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.urls[0]).query)
        self.assertEqual(query["latitude"], ["44.125"])
        self.assertEqual(query["longitude"], ["79.875"])
        self.assertEqual(first["coordinates"], {"latitude": 44.125, "longitude": 79.875})
        self.assertEqual(len(first["forecast"]), 48)
        self.assertEqual(forecast_weather.retrieve(DAY, "turbine-1", self.cache, locations=custom), first)
        self.assertEqual(len(self.urls), 1)  # cached, no network
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            forecast_weather.retrieve(DAY, "turbine-1", self.cache)
        self.assertEqual(len(self.urls), 1)
        altered = copy.deepcopy(first)
        altered["coordinates"]["latitude"] = 44.0
        cache_path = self.cache / "2026-02-01-turbine-1.json"
        cache_path.write_text(json.dumps(altered))
        with self.assertRaisesRegex(ValueError, "coordinates mismatch"):
            forecast_weather.retrieve(DAY, "turbine-1", self.cache, locations=custom)
        self.assertEqual(len(self.urls), 1)

    def test_default_behavior_matches_explicit_global_locations(self):
        default = forecast_weather.retrieve(DAY, "turbine-1", self.cache / "default")
        explicit = forecast_weather.retrieve(DAY, "turbine-1", self.cache / "explicit",
                                             locations=forecast_weather.LOCATIONS)
        self.assertEqual(default, explicit)
        self.assertEqual(self.urls[0], self.urls[1])
        self.assertEqual((self.cache / "default/2026-02-01-turbine-1.json").read_bytes(),
                         (self.cache / "explicit/2026-02-01-turbine-1.json").read_bytes())
        self.assertEqual(forecast_weather.LOCATIONS["turbine-1"], (43.645150, 78.535604))


if __name__ == "__main__":
    unittest.main()
