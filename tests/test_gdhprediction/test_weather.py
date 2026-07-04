import math
from datetime import datetime, timedelta
from typing import Any
from unittest import TestCase
from zoneinfo import ZoneInfo

import numpy as np

from gdhprediction.utils import BrightSkyDType, rh_from
from gdhprediction.weather import (
    BrightskyDataRequest,
    WeatherRequestData,
    _coerce_to_float,
)
from tests._base import ArrayTestCase

# ruff: noqa: SLF001

# Berlin — a valid coordinate so timezonefinder resolves a tz.
LAT, LON = 52.52, 13.41
BERLIN = ZoneInfo("Europe/Berlin")
TOKYO = ZoneInfo("Asia/Tokyo")


def _make(start: datetime | None = None, duration: int = 10) -> WeatherRequestData:
    return WeatherRequestData(LAT, LON, start, duration)


def _request() -> BrightskyDataRequest:
    """A request instance; its interpolation/parsing methods hit no network."""
    return BrightskyDataRequest(WeatherRequestData(LAT, LON))


def _bsky_array(
    n: int,
    *,
    t2m: Any = None,
    rh2m: Any = None,
    allsky: Any = None,
) -> np.ndarray:
    """Structured BrightSky array; per-field defaults are all finite (NaN-free)."""
    arr = np.zeros(n, dtype=BrightSkyDType)
    arr["MO"] = 6
    arr["DY"] = 15
    arr["HR"] = (np.arange(n) % 24).astype(np.uint8)
    arr["T2M"] = np.arange(n, dtype=np.float32) if t2m is None else t2m
    arr["RH2M"] = 50.0 if rh2m is None else rh2m
    arr["ALLSKY_SFC_SW_DWN"] = 10.0 if allsky is None else allsky
    return arr


class TestWeatherRequestDataSnapping(TestCase):
    """Tests :class:`WeatherRequestData` construction.

    WeatherRequestData.__post_init__ snaps the start hour down to the
    nearest SNAPPED=6h boundary in the GPS-derived timezone, then writes
    `date` and `last_date` as ISO strings spanning `duration` days.
    """

    def test_hour_snaps_down_to_nearest_six(self) -> None:
        # (hour // 6) * 6 produces buckets at 0, 6, 12, 18.
        for hour, expected in [
            (0, 0),
            (5, 0),
            (6, 6),
            (11, 6),
            (12, 12),
            (14, 12),  # the off-boundary case used elsewhere in the suite
            (17, 12),
            (18, 18),
            (23, 18),
        ]:
            with self.subTest(hour=hour):
                rd = _make(datetime(2024, 6, 15, hour, 30))
                start = datetime.fromisoformat(rd.date)
                self.assertEqual(start.hour, expected)
                # Sub-hour fields are zeroed.
                self.assertEqual(
                    (start.minute, start.second, start.microsecond),
                    (0, 0, 0),
                )

    def test_naive_input_is_stamped_with_gps_tz(self) -> None:
        # 14:30 naive → 12:00 Berlin (snapped). Naive input must NOT be
        # converted, only stamped. Note: ISO round-trip yields a fixed-
        # offset tzinfo, so we compare utcoffset() rather than identity.
        rd = _make(datetime(2024, 6, 15, 14, 30))
        start = datetime.fromisoformat(rd.date)
        expected_offset = datetime(2024, 6, 15, 12, tzinfo=BERLIN).utcoffset()
        self.assertEqual(start.utcoffset(), expected_offset)
        self.assertEqual(
            (start.year, start.month, start.day, start.hour),
            (2024, 6, 15, 12),
        )

    def test_aware_input_is_converted_then_snapped(self) -> None:
        # 14:30 Tokyo (UTC+9) → 07:30 Berlin (CEST, UTC+2) → snapped to 06:00.
        aware = datetime(2024, 6, 15, 14, 30, tzinfo=TOKYO)
        rd = _make(aware)
        start = datetime.fromisoformat(rd.date)
        # Local hour after conversion + snapping.
        self.assertEqual(start.hour, 6)
        # Same instant on the timeline as the snapped Berlin equivalent.
        expected = aware.astimezone(BERLIN).replace(
            hour=6,
            minute=0,
            second=0,
            microsecond=0,
        )
        self.assertEqual(start, expected)

    def test_last_date_is_exactly_duration_days_after_start(self) -> None:
        rd = _make(datetime(2024, 6, 15, 14, 30), duration=7)
        start = datetime.fromisoformat(rd.date)
        end = datetime.fromisoformat(rd.last_date)
        self.assertEqual(end - start, timedelta(days=7))

    def test_default_duration_is_ten_days(self) -> None:
        rd = _make(datetime(2024, 6, 15, 12, 0))
        start = datetime.fromisoformat(rd.date)
        end = datetime.fromisoformat(rd.last_date)
        self.assertEqual(end - start, timedelta(days=10))

    def test_none_start_uses_now_in_gps_tz(self) -> None:
        # Hard to assert exact "now", but the result must be tz-aware in
        # the GPS tz (offset-wise) and snapped to a 6h boundary within
        # the current bucket.
        before = datetime.now(BERLIN)
        rd = _make(None)
        after = datetime.now(BERLIN)
        start = datetime.fromisoformat(rd.date)
        # The ISO round-trip yields a fixed-offset tzinfo, so compare
        # utcoffset() against Berlin's offset at this moment.
        self.assertEqual(start.utcoffset(), before.utcoffset())
        # `start` should sit in the snapped-down bucket of `before` (or
        # `after` if the clock crossed a SNAPPED boundary during the call).
        snap = WeatherRequestData.SNAPPED
        floor_before = before.replace(
            hour=(before.hour // snap) * snap,
            minute=0,
            second=0,
            microsecond=0,
        )
        floor_after = after.replace(
            hour=(after.hour // snap) * snap,
            minute=0,
            second=0,
            microsecond=0,
        )
        self.assertGreaterEqual(start, floor_before)
        self.assertLessEqual(start, floor_after)


class TestCoerceToFloat(TestCase):
    """``_coerce_to_float`` maps None → NaN and rejects non-numeric values."""

    def test_none_becomes_nan(self) -> None:
        out = _coerce_to_float({"a": None, "b": 3})
        self.assertTrue(math.isnan(out["a"]))
        self.assertEqual(out["b"], 3.0)

    def test_drop_excludes_keys(self) -> None:
        out = _coerce_to_float({"a": 1, "ts": "skip me"}, drop={"ts"})
        self.assertEqual(out, {"a": 1.0})

    def test_non_numeric_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            _coerce_to_float({"a": "not a number"})


class TestFilterRecords(TestCase):
    """``_filter_records`` validates, parses timestamps, and backfills RH."""

    @staticmethod
    def _raw(**overrides: Any) -> dict[str, Any]:
        rec: dict[str, Any] = {
            "timestamp": "2024-06-15T00:00:00+02:00",
            "temperature": 20.0,
            "relative_humidity": 55.0,
            "sunshine": 30.0,
            "dew_point": 10.0,
        }
        rec.update(overrides)
        return rec

    def setUp(self) -> None:
        self.req = _request()

    def test_missing_required_field_raises(self) -> None:
        rec = self._raw()
        del rec["sunshine"]
        with self.assertRaises(ValueError):
            self.req._filter_records([rec])

    def test_non_str_timestamp_raises(self) -> None:
        with self.assertRaises(TypeError):
            self.req._filter_records([self._raw(timestamp=1234)])

    def test_bad_timestamp_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.req._filter_records([self._raw(timestamp="not-a-date")])

    def test_relative_humidity_computed_from_dew_point(self) -> None:
        # None RH is coerced to NaN, then recomputed from temperature + dew point.
        out = self.req._filter_records([self._raw(relative_humidity=None)])
        self.assertAlmostEqual(out[0]["relative_humidity"], rh_from(20.0, 10.0))

    def test_dew_point_dropped_and_timestamp_parsed(self) -> None:
        out = self.req._filter_records([self._raw()])
        self.assertNotIn("dew_point", out[0])
        self.assertIsInstance(out[0]["timestamp"], datetime)


class TestInterpolateNans(ArrayTestCase):
    """``_interpolate_nans`` backfills leading gaps from +24h then interpolates."""

    def setUp(self) -> None:
        self.req = _request()

    def test_leading_nans_backfilled_from_next_day(self) -> None:
        t2m = np.arange(48, dtype=np.float32)
        t2m[:3] = np.nan
        arr = _bsky_array(48, t2m=t2m)
        self.req._interpolate_nans(arr)
        # Same hour one day later: indices 24, 25, 26.
        self.assertAllClose(arr["T2M"][:3], [24.0, 25.0, 26.0])
        self.assertAllClose(arr["T2M"][3:], np.arange(48, dtype=np.float32)[3:])

    def test_all_nan_column_raises(self) -> None:
        arr = _bsky_array(48, t2m=np.full(48, np.nan, dtype=np.float32))
        with self.assertRaises(ValueError):
            self.req._interpolate_nans(arr)

    def test_leading_gap_longer_than_a_day_raises(self) -> None:
        t2m = np.arange(60, dtype=np.float32)
        t2m[:25] = np.nan
        arr = _bsky_array(60, t2m=t2m)
        with self.assertRaises(ValueError):
            self.req._interpolate_nans(arr)

    def test_leading_gap_without_a_full_next_day_raises(self) -> None:
        # first_valid=3 but size=26 < 3 + 24, so the +24 source window overruns.
        t2m = np.arange(26, dtype=np.float32)
        t2m[:3] = np.nan
        arr = _bsky_array(26, t2m=t2m)
        with self.assertRaises(ValueError):
            self.req._interpolate_nans(arr)

    def test_intermittent_nan_linearly_interpolated(self) -> None:
        arr = _bsky_array(6, t2m=np.array([0, 1, 2, np.nan, 4, 5], dtype=np.float32))
        self.req._interpolate_nans(arr)
        self.assertAlmostEqual(float(arr["T2M"][3]), 3.0, places=5)

    def test_intermittent_gap_longer_than_a_day_raises(self) -> None:
        t2m = np.arange(60, dtype=np.float32)
        t2m[10:37] = np.nan  # 27-hour interior gap
        arr = _bsky_array(60, t2m=t2m)
        with self.assertRaises(ValueError):
            self.req._interpolate_nans(arr)

    def test_too_few_valid_points_raises(self) -> None:
        # One valid anchor at index 0; the 24-hour tail passes the gap check but
        # leaves < 2 points to interpolate from.
        t2m = np.full(25, np.nan, dtype=np.float32)
        t2m[0] = 5.0
        arr = _bsky_array(25, t2m=t2m)
        with self.assertRaises(ValueError):
            self.req._interpolate_nans(arr)

    def test_returns_input_array_unchanged_when_clean(self) -> None:
        arr = _bsky_array(48)
        result = self.req._interpolate_nans(arr)
        self.assertIs(result, arr)

    def test_error_message_uses_source_field_label(self) -> None:
        # Diagnostics name the BrightSky source field beside the internal one.
        allsky = np.arange(60, dtype=np.float32)
        allsky[10:37] = np.nan
        arr = _bsky_array(60, allsky=allsky)
        with self.assertRaises(ValueError) as ctx:
            self.req._interpolate_nans(arr)
        self.assertIn("sunshine (ALLSKY_SFC_SW_DWN)", str(ctx.exception))


class TestProcessBrightskyData(ArrayTestCase):
    """End-to-end: interpolate raw sunshine, then vectorised radiation conversion."""

    def setUp(self) -> None:
        self.req = _request()

    @staticmethod
    def _records(n: int, sunshine: list[float] | None = None) -> list[dict[str, Any]]:
        base = datetime(2024, 6, 15, 0, tzinfo=BERLIN)
        return [
            {
                "timestamp": base + timedelta(hours=i),
                "temperature": 20.0 + i,
                "relative_humidity": 55.0,
                "sunshine": float(i % 24) if sunshine is None else sunshine[i],
            }
            for i in range(n)
        ]

    def test_leading_sunshine_nan_is_filled_and_converted(self) -> None:
        # Regression: the conversion is called on whole columns; leading-NaN
        # sunshine must be filled before the (vectorised) radiation conversion.
        sun = [float(i % 24) for i in range(48)]
        sun[0] = sun[1] = sun[2] = math.nan
        out = self.req._process_brightsky_data(self._records(48, sun))
        radiation = out["ALLSKY_SFC_SW_DWN"]
        self.assertFalse(np.any(np.isnan(radiation)))
        self.assertTrue(np.all(radiation >= 0.0))

    def test_temperature_and_humidity_pass_through(self) -> None:
        out = self.req._process_brightsky_data(self._records(48))
        self.assertAllClose(out["T2M"], 20.0 + np.arange(48, dtype=np.float32))
        self.assertAllClose(out["RH2M"], np.full(48, 55.0, dtype=np.float32))
