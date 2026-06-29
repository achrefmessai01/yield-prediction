from datetime import date, datetime, timedelta
from typing import Final
from unittest import TestCase
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np

from gdhprediction.apis import (
    Frequency,
    _coerce_tz_midnight,
    get_current_brightsky_forecast,
    get_gdh_prediction,
)
from gdhprediction.config import GdhLocalizedPredictionConfig
from gdhprediction.utils import Float32, StructuredDataArray, coerce_tz_datetime
from tests._base import ArrayTestCase

WEATHER_DTYPE = np.dtype([("HR", "<u1"), ("GDH", "<f4")])
CLIMATE_DTYPE = np.dtype([("GDH_p50", "<f4")])

# Berlin: a valid coordinate for validate_gps and the GPS timezone lookup
LAT: Final = 52.52
LON: Final = 13.41
DEFAULT_DURATION: Final = 28


def _weather(snapped_hour: int, length: int = 240) -> StructuredDataArray:
    arr = np.empty(length, dtype=WEATHER_DTYPE)
    arr["HR"] = snapped_hour
    arr["GDH"] = np.arange(length, dtype=np.float32)
    return arr


def _climate(length: int, step: float = 0.5) -> StructuredDataArray:
    arr = np.empty(length, dtype=CLIMATE_DTYPE)
    arr["GDH_p50"] = np.arange(length, dtype=np.float32) * step
    return arr


def _run(
    start: datetime,
    snapped_hour: int,
    frequency: Frequency,
    duration: int = DEFAULT_DURATION,
    weather_length: int = 240,
) -> tuple[Float32, StructuredDataArray, StructuredDataArray, int]:
    """Run get_gdh_prediction with mocked weather/climate.

    Returns the prediction array, the mocked weather array, the mocked
    climate array, and the ``forecast=`` value passed to climate so each
    test can assert the horizon explicitly. ``weather_length`` controls how
    many forecast hours BrightSky is mocked to return (0 simulates a window
    entirely past the ~10-day horizon; a small value simulates truncation).
    """
    weather_arr = _weather(snapped_hour, weather_length)
    # apis.get_gdh_prediction asks climate for `duration + 1` days in hourly
    # mode so the final array can be truncated cleanly to `duration * 24`.
    expected_forecast = duration + 1 if frequency == "hourly" else duration
    climate_arr = _climate(expected_forecast * 24)
    with (
        patch("gdhprediction.apis.ClimateGdhData") as climate_cls,
        patch("gdhprediction.apis.WeatherGdhData") as weather_cls,
    ):
        climate_cls.from_params.return_value.forecast_gdh.return_value = climate_arr
        weather_cls.from_params.return_value.forecast_gdh.return_value = weather_arr
        prediction = get_gdh_prediction(
            GdhLocalizedPredictionConfig(latitude=LAT, longitude=LON),
            start,
            frequency=frequency,
            duration=duration,
        )
        forecast_arg = int(
            climate_cls.from_params.return_value.forecast_gdh.call_args.kwargs[
                "forecast"
            ],
        )
    return prediction, weather_arr, climate_arr, forecast_arg


class TestStitchContinuity(ArrayTestCase):
    """Verify the seam between weather and climate is continuous and aligned
    to the requested hour. Weather is cropped from the snapped hour to the
    requested hour, re-anchored so its first GDH is zero, and the prediction
    is truncated to exactly `duration * 24` hours in hourly mode.
    """

    def assertHourlyInvariants(
        self,
        prediction: Float32,
        weather: StructuredDataArray,
        climate: StructuredDataArray,
        snapped_hour: int,
        datetime_hour: int,
        duration: int = DEFAULT_DURATION,
    ) -> None:
        """Properties that must hold for any successful hourly stitch."""
        weather_start = datetime_hour - snapped_hour
        n_w_cropped = len(weather) - weather_start
        anchor = float(weather["GDH"][weather_start])
        duration_hours = duration * 24

        # Final shape is exactly `duration * 24`.
        self.assertEqual(prediction.shape, (duration_hours,))

        # Weather portion: cropped from `weather_start`, anchored so the
        # first value is zero.
        expected_weather = weather["GDH"][weather_start:] - anchor
        self.assertAllClose(
            prediction[:n_w_cropped],
            expected_weather,
            atol=1.5e-6,
        )

        # Seam is continuous: climate_part[0] anchors on the anchored
        # weather tail, so the first climate sample equals it.
        self.assertAlmostEqual(
            float(prediction[n_w_cropped]),
            float(expected_weather[-1]),
            places=4,
        )

        # Slope past the seam equals the climate array's intrinsic step.
        climate_step = float(climate["GDH_p50"][1] - climate["GDH_p50"][0])
        self.assertAlmostEqual(
            float(prediction[n_w_cropped + 1] - prediction[n_w_cropped]),
            climate_step,
            places=4,
        )

    def test_daily_at_midnight(self) -> None:
        # snapped_hour=0 since 0 % 6 == 0; daily forces midnight, so
        # weather_start=0 and the anchor (weather["GDH"][0]==0) is a no-op.
        prediction, weather, climate, forecast = _run(
            start=datetime(2024, 6, 15, 0, 0),
            snapped_hour=0,
            frequency="daily",
        )
        # Daily mode asks climate for exactly `duration` days.
        self.assertEqual(forecast, DEFAULT_DURATION)
        # Daily output has exactly `duration` samples.
        self.assertEqual(prediction.shape, (DEFAULT_DURATION,))
        # Day 0 last hour is hour 23 in the hourly stream, still inside
        # the weather window.
        self.assertAlmostEqual(
            float(prediction[0]),
            float(weather["GDH"][23]),
            places=4,
        )
        # Day (len(weather)//24 - 1) last hour is the final weather sample.
        last_weather_day = len(weather) // 24 - 1
        self.assertAlmostEqual(
            float(prediction[last_weather_day]),
            float(weather["GDH"][-1]),
            places=4,
        )
        # First fully-climate day: hourly index = len(weather) + 23.
        # => climate_part[k] = climate[seam+k] - climate[seam] + weather[-1]
        seam = len(weather)
        expected_first_climate_day = float(
            climate["GDH_p50"][seam + 23]
            - climate["GDH_p50"][seam]
            + weather["GDH"][-1],
        )
        self.assertAlmostEqual(
            float(prediction[last_weather_day + 1]),
            expected_first_climate_day,
            places=4,
        )

    def test_hourly_on_six_hour_boundary(self) -> None:
        # 12:00 is on a 6h boundary, so snapped_hour == datetime_hour == 12;
        # weather_start=0 and the anchor is zero (no shift).
        prediction, weather, climate, forecast = _run(
            start=datetime(2024, 6, 15, 12, 0),
            snapped_hour=12,
            frequency="hourly",
        )
        # Hourly mode asks climate for an extra day so truncation has slack.
        self.assertEqual(forecast, DEFAULT_DURATION + 1)
        self.assertHourlyInvariants(
            prediction,
            weather,
            climate,
            snapped_hour=12,
            datetime_hour=12,
        )

    def test_hourly_off_six_hour_boundary(self) -> None:
        """At 14:30 the request snaps back to 12:00 but the prediction is
        aligned to 14:00, so the first two cumsum hours are cropped and
        weather GDH is re-anchored to zero at hour 14.
        """
        prediction, weather, climate, forecast = _run(
            start=datetime(2024, 6, 15, 14, 30),
            snapped_hour=12,
            frequency="hourly",
        )
        self.assertEqual(forecast, DEFAULT_DURATION + 1)
        self.assertHourlyInvariants(
            prediction,
            weather,
            climate,
            snapped_hour=12,
            datetime_hour=14,
        )

    def test_hourly_short_weather_is_stitched_to_full_duration(self) -> None:
        """A future-dated start can make BrightSky truncate to far fewer than
        240 hours. The seam anchors on len(weather_data), so the climate leg
        backfills and the output is still exactly `duration * 24`.
        """
        prediction, weather, climate, forecast = _run(
            start=datetime(2024, 6, 15, 12, 0),
            snapped_hour=12,
            frequency="hourly",
            weather_length=95,  # < duration_hours; forces a climate extension
        )
        self.assertEqual(len(weather), 95)
        self.assertEqual(forecast, DEFAULT_DURATION + 1)
        self.assertHourlyInvariants(
            prediction,
            weather,
            climate,
            snapped_hour=12,
            datetime_hour=12,
        )


BERLIN: Final = ZoneInfo("Europe/Berlin")
TOKYO: Final = ZoneInfo("Asia/Tokyo")


class TestDstSeamCorrection(ArrayTestCase):
    """BrightSky stamps every record with the offset present in the request's
    `date` field (no ``tz=`` param is sent). NASA POWER LST is local
    *standard* time year-round. So when ``start_datetime`` is in DST, the
    weather frame runs one civil hour ahead of climate; ``climate_start``
    must be pulled back by one to keep the seam aligned. When the start is
    in standard time, no correction is needed.
    """

    @staticmethod
    def _run_with_climate(start: datetime, climate_arr: StructuredDataArray) -> Float32:
        weather_arr = _weather(snapped_hour=12)
        with (
            patch("gdhprediction.apis.ClimateGdhData") as climate_cls,
            patch("gdhprediction.apis.WeatherGdhData") as weather_cls,
        ):
            climate_cls.from_params.return_value.forecast_gdh.return_value = climate_arr
            weather_cls.from_params.return_value.forecast_gdh.return_value = weather_arr
            return get_gdh_prediction(
                GdhLocalizedPredictionConfig(latitude=LAT, longitude=LON),
                start,
                frequency="hourly",
                duration=DEFAULT_DURATION,
            )

    def test_summer_start_pulls_climate_seam_back_by_one(self) -> None:
        """A quadratic climate array makes consecutive differences distinct,
        so the seam index is observable from the prediction.
        """
        length = (DEFAULT_DURATION + 1) * 24
        climate_arr = np.empty(length, dtype=CLIMATE_DTYPE)
        # quadratic so climate[k+1]-climate[k] = 2k+1, distinct per index
        climate_arr["GDH_p50"] = np.arange(length, dtype=np.float32) ** 2

        # 12:00 is a 6h snap boundary -> weather_start = 0 -> weather portion
        # is the full 240 records and the anchor is weather["GDH"][0] == 0.
        summer_pred = self._run_with_climate(
            datetime(2024, 6, 15, 12, 0),
            climate_arr.copy(),
        )
        winter_pred = self._run_with_climate(
            datetime(2024, 12, 15, 12, 0),
            climate_arr.copy(),
        )

        # Sanity: weather portion is identical in both seasons (mock returns
        # the same array; anchor and crop are season-independent).
        n_weather = 240
        self.assertAllClose(
            summer_pred[:n_weather],
            winter_pred[:n_weather],
            atol=1.5e-6,
        )

        # Seam value is always the weather anchor regardless of DST, because
        # climate_part is re-anchored with `- climate[climate_start]`.
        weather_last = 239.0  # from _weather(snapped_hour=12)["GDH"][-1]
        self.assertAlmostEqual(float(winter_pred[n_weather]), weather_last)
        self.assertAlmostEqual(float(summer_pred[n_weather]), weather_last)

        # The first climate STEP reveals climate_start.
        # No-DST path:    climate_start = 240 + 12      = 252
        # In-DST path:    climate_start = 240 + 12 - 1  = 251
        gdh = climate_arr["GDH_p50"]
        expected_winter_step = float(gdh[253] - gdh[252])  # 2*252 + 1 = 505
        expected_summer_step = float(gdh[252] - gdh[251])  # 2*251 + 1 = 503

        self.assertAlmostEqual(
            float(winter_pred[n_weather + 1] - winter_pred[n_weather]),
            expected_winter_step,
            places=3,
        )
        self.assertAlmostEqual(
            float(summer_pred[n_weather + 1] - summer_pred[n_weather]),
            expected_summer_step,
            places=3,
        )
        # And the two seasons must disagree -- if they don't, the correction
        # is silently a no-op.
        self.assertNotAlmostEqual(
            float(summer_pred[n_weather + 1]),
            float(winter_pred[n_weather + 1]),
            places=1,
        )


class TestClimatologyOnlyFallback(ArrayTestCase):
    """When BrightSky returns no usable forecast hours -- either an empty
    response (window entirely past the horizon) or a short response that the
    snap-crop empties -- the prediction must fall back to climatology only,
    keep its full length, and never index into an empty weather array.

    These pin three regressions found while hardening the seam:
      * IndexError on the re-anchor / seam tail access for empty weather,
      * the same crash re-hidden inside the non-empty branch by a refactor,
      * a negative ``climate_start`` wrap when datetime_hour == 0 in DST.
    """

    def _assert_climatology_only(
        self,
        prediction: Float32,
        climate: StructuredDataArray,
        datetime_hour: int,
        duration_hours: int,
    ) -> None:
        # Full length preserved and the leg starts at zero (re-anchored).
        self.assertEqual(prediction.shape, (duration_hours,))
        self.assertAlmostEqual(float(prediction[0]), 0.0, places=4)
        # With no weather, climate_start == datetime_hour (no DST correction),
        # so prediction[k] == climate[datetime_hour + k] - climate[datetime_hour].
        expected = (
            climate["GDH_p50"][datetime_hour : datetime_hour + duration_hours]
            - climate["GDH_p50"][datetime_hour]
        )
        self.assertAllClose(prediction, expected, atol=1.5e-6)

    def test_empty_weather_hourly_falls_back_to_climatology(self) -> None:
        with self.assertLogs("gdhprediction.apis", level="WARNING"):
            prediction, _weather_arr, climate, _forecast = _run(
                start=datetime(2024, 6, 15, 14, 30),
                snapped_hour=0,
                frequency="hourly",
                weather_length=0,
            )
        self._assert_climatology_only(
            prediction,
            climate,
            datetime_hour=14,
            duration_hours=DEFAULT_DURATION * 24,
        )

    def test_empty_weather_daily_in_dst_returns_full_duration(self) -> None:
        """Regression: in DST with datetime_hour == 0 the old code computed
        ``climate_start = 0 + 0 - 1 == -1``, wrapping to the climate tail and
        collapsing the daily output to length 0. The output must be `duration`.
        """
        # July midnight in Berlin is CEST (DST); daily coerces to midnight.
        prediction, _weather_arr, _climate, _forecast = _run(
            start=datetime(2024, 7, 15, 0, 0),
            snapped_hour=0,
            frequency="daily",
            weather_length=0,
        )
        self.assertEqual(prediction.shape, (DEFAULT_DURATION,))

    def test_short_weather_cropped_to_empty_falls_back_to_climatology(self) -> None:
        """Regression: a near-horizon start returns only a few rows; the snap
        crop (weather_start in [0, 5]) can empty them. The re-anchor and seam
        accesses must not run against the emptied array.
        """
        # snapped_hour=18, requested hour 23 -> weather_start = 5; only 3 rows
        # returned -> _weather_data[5:] is empty.
        prediction, _weather_arr, climate, _forecast = _run(
            start=datetime(2024, 6, 15, 23, 0),
            snapped_hour=18,
            frequency="hourly",
            weather_length=3,
        )
        self._assert_climatology_only(
            prediction,
            climate,
            datetime_hour=23,
            duration_hours=DEFAULT_DURATION * 24,
        )


class TestCoerceTzDatetime(TestCase):
    """Tests the normalization point for the start datetime.

    The :func:`coerce_tz_datetime` must
    #. stamp naive inputs without shifting wall-clock fields,
    #. convert tz-aware inputs to the GPS tz,
    #. lift bare dates to midnight in the GPS tz, and
    #. default to "now" when no input is given.
    """

    def test_none_returns_now_in_target_tz(self) -> None:
        before = datetime.now(BERLIN)
        result = coerce_tz_datetime(None, BERLIN)
        after = datetime.now(BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        # Result was sampled between before and after.
        self.assertGreaterEqual(result, before - timedelta(seconds=1))
        self.assertLessEqual(result, after + timedelta(seconds=1))

    def test_naive_datetime_is_stamped_not_shifted(self) -> None:
        naive = datetime(2024, 6, 15, 14, 30)
        result = coerce_tz_datetime(naive, BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        self.assertEqual(
            (result.year, result.month, result.day, result.hour, result.minute),
            (2024, 6, 15, 14, 30),
        )

    def test_aware_datetime_in_other_tz_is_converted(self) -> None:
        # 14:30 Tokyo (UTC+9) → 07:30 Berlin (CEST, UTC+2) on a June day.
        aware_tokyo = datetime(2024, 6, 15, 14, 30, tzinfo=TOKYO)
        result = coerce_tz_datetime(aware_tokyo, BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        self.assertEqual(result.hour, 7)
        self.assertEqual(result.minute, 30)
        # Same instant in time, just expressed in Berlin's local clock.
        self.assertEqual(result, aware_tokyo)

    def test_bare_date_lifts_to_local_midnight(self) -> None:
        result = coerce_tz_datetime(date(2024, 6, 15), BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        self.assertEqual(
            (result.year, result.month, result.day, result.hour, result.minute),
            (2024, 6, 15, 0, 0),
        )


class TestCoerceTzMidnight(TestCase):
    """Test normalization point for the date midnight."""

    def test_naive_datetime_with_time_is_truncated_to_midnight(self) -> None:
        result = _coerce_tz_midnight(datetime(2024, 6, 15, 14, 30, 45, 123), BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        self.assertEqual(
            (result.hour, result.minute, result.second, result.microsecond),
            (0, 0, 0, 0),
        )
        self.assertEqual((result.year, result.month, result.day), (2024, 6, 15))

    def test_aware_datetime_is_converted_then_truncated(self) -> None:
        # 23:30 Tokyo (Jun 15) → 16:30 Berlin (Jun 15) → midnight Jun 15 Berlin.
        # Important: truncation happens *after* conversion, so a same-day
        # Tokyo evening doesn't get pushed back to Jun 14 by accident.
        aware = datetime(2024, 6, 15, 23, 30, tzinfo=TOKYO)
        result = _coerce_tz_midnight(aware, BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        self.assertEqual((result.year, result.month, result.day), (2024, 6, 15))
        self.assertEqual(result.hour, 0)

    def test_bare_date_is_already_midnight(self) -> None:
        result = _coerce_tz_midnight(date(2024, 6, 15), BERLIN)
        self.assertEqual(result.tzinfo, BERLIN)
        self.assertEqual((result.hour, result.minute), (0, 0))


# A fixed "now" in Berlin: 14:30 snaps down to the 12:00 boundary (drop = 2 hours).
FIXED_NOW: Final = datetime(2024, 6, 15, 14, 30, tzinfo=BERLIN)


class TestGetCurrentBrightskyForecast(TestCase):
    """Covers the now-anchored forecast: snapped-hour drop, fixed length, caching.

    BrightSky's network call is mocked and ``WeatherRequestData`` runs for real
    (it is pure). The mocked provider response is a structured array whose first
    row carries the snapped hour, so the drop arithmetic is exercised end-to-end.
    """

    def _call(
        self,
        *,
        snapped_hour: int = 12,
        length: int = 240,
        cache: bool = True,
        now: datetime = FIXED_NOW,
    ) -> tuple[StructuredDataArray, StructuredDataArray, MagicMock, MagicMock]:
        raw = _weather(snapped_hour=snapped_hour, length=length)
        with (
            patch("gdhprediction.apis.datetime") as dt,
            patch("gdhprediction.apis.BrightskyDataRequest") as brightsky_cls,
            patch("gdhprediction.apis.CacheManager") as cache_cls,
        ):
            dt.now.return_value = now
            brightsky_cls.return_value.request_brightsky_data.return_value = raw
            result = get_current_brightsky_forecast(LAT, LON, cache=cache)
        return result, raw, brightsky_cls, cache_cls

    def test_drops_hours_between_now_and_snapped_boundary(self) -> None:
        # now=14:30 -> snapped 12:00 -> drop the leading 2 hours, cap to 234.
        result, raw, _brightsky_cls, _cache_cls = self._call()
        self.assertEqual(len(result), 234)
        # GDH is arange, so the first kept row is raw[2] (value 2.0).
        self.assertEqual(result["GDH"][0], raw["GDH"][2])
        self.assertEqual(result["GDH"][0], 2.0)

    def test_boundary_now_drops_nothing(self) -> None:
        # now exactly on a 6h boundary -> drop 0, still capped to 234.
        on_boundary = datetime(2024, 6, 15, 12, 0, tzinfo=BERLIN)
        result, raw, _brightsky_cls, _cache_cls = self._call(now=on_boundary)
        self.assertEqual(len(result), 234)
        self.assertEqual(result["GDH"][0], raw["GDH"][0])

    def test_empty_response_raises(self) -> None:
        with self.assertRaises(ValueError):
            self._call(length=0)

    def test_short_response_warns_and_returns_best_effort(self) -> None:
        # fewer than the full horizon -> warning, still returns the trimmed slice.
        with self.assertLogs("gdhprediction.apis", level="WARNING"):
            result, _raw, _brightsky_cls, _cache_cls = self._call(length=100)
        self.assertEqual(len(result), 98)  # 100 - drop(2), below the 234 cap

    def test_cache_true_writes_untrimmed_raw(self) -> None:
        result, raw, _brightsky_cls, cache_cls = self._call(cache=True)
        cache_cls.assert_called_once()
        # written to the volatile (non-persistent) cache as the raw format.
        self.assertFalse(cache_cls.call_args.kwargs["persistent"])
        write = cache_cls.return_value.write_to_cache
        write.assert_called_once()
        self.assertEqual(write.call_args.kwargs["cache_fmt"], "raw")
        # the cached array is the full, untrimmed provider response (not the
        # now-aligned slice) so the GDH path's raw cache stays uncontaminated.
        cached = write.call_args.args[0]
        self.assertEqual(len(cached), len(raw))
        self.assertEqual(len(cached), 240)
        self.assertEqual(len(result), 234)

    def test_cache_false_skips_write(self) -> None:
        _result, _raw, _brightsky_cls, cache_cls = self._call(cache=False)
        cache_cls.assert_not_called()
