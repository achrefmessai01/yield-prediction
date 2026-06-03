from datetime import date, datetime, timedelta
from typing import Final
from unittest import TestCase
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

from gdhprediction.apis import (
    Frequency,
    _coerce_tz_midnight,
    get_gdh_prediction,
)
from gdhprediction.config import GdhLocalizedPredictionConfig
from gdhprediction.utils import coerce_tz_datetime

WEATHER_DTYPE = np.dtype([("HR", "<u1"), ("GDH", "<f4")])
CLIMATE_DTYPE = np.dtype([("GDH_p50", "<f4")])

# Berlin: a valid coordinate for validate_gps and the GPS timezone lookup
LAT: Final = 52.52
LON: Final = 13.41
DEFAULT_DURATION: Final = 28


def _weather(snapped_hour: int, length: int = 240) -> np.ndarray:
    arr = np.empty(length, dtype=WEATHER_DTYPE)
    arr["HR"] = snapped_hour
    arr["GDH"] = np.arange(length, dtype=np.float32)
    return arr


def _climate(length: int, step: float = 0.5) -> np.ndarray:
    arr = np.empty(length, dtype=CLIMATE_DTYPE)
    arr["GDH_p50"] = np.arange(length, dtype=np.float32) * step
    return arr


def _run(
    start: datetime,
    snapped_hour: int,
    frequency: Frequency,
    duration: int = DEFAULT_DURATION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Run get_gdh_prediction with mocked weather/climate.

    Returns the prediction array, the mocked weather array, the mocked
    climate array, and the ``forecast=`` value passed to climate so each
    test can assert the horizon explicitly.
    """
    weather_arr = _weather(snapped_hour)
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
            GdhLocalizedPredictionConfig(LAT, LON),
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


class TestStitchContinuity(TestCase):
    """Verify the seam between weather and climate is continuous and aligned
    to the requested hour. Weather is cropped from the snapped hour to the
    requested hour, re-anchored so its first GDH is zero, and the prediction
    is truncated to exactly `duration * 24` hours in hourly mode.
    """

    def assertHourlyInvariants(
        self,
        prediction: np.ndarray,
        weather: np.ndarray,
        climate: np.ndarray,
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
        np.testing.assert_array_almost_equal(
            prediction[:n_w_cropped],
            expected_weather,
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


BERLIN: Final = ZoneInfo("Europe/Berlin")
TOKYO: Final = ZoneInfo("Asia/Tokyo")


class TestDstSeamCorrection(TestCase):
    """BrightSky stamps every record with the offset present in the request's
    `date` field (no ``tz=`` param is sent). NASA POWER LST is local
    *standard* time year-round. So when ``start_datetime`` is in DST, the
    weather frame runs one civil hour ahead of climate; ``climate_start``
    must be pulled back by one to keep the seam aligned. When the start is
    in standard time, no correction is needed.
    """

    @staticmethod
    def _run_with_climate(start: datetime, climate_arr: np.ndarray) -> np.ndarray:
        weather_arr = _weather(snapped_hour=12)
        with (
            patch("gdhprediction.apis.ClimateGdhData") as climate_cls,
            patch("gdhprediction.apis.WeatherGdhData") as weather_cls,
        ):
            climate_cls.from_params.return_value.forecast_gdh.return_value = climate_arr
            weather_cls.from_params.return_value.forecast_gdh.return_value = weather_arr
            return get_gdh_prediction(
                GdhLocalizedPredictionConfig(LAT, LON),
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
        np.testing.assert_array_almost_equal(
            summer_pred[:n_weather],
            winter_pred[:n_weather],
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
