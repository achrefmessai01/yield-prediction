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
from gdhprediction.config import GdhPredictionConfig
from gdhprediction.utils import coerce_tz_datetime

WEATHER_DTYPE = np.dtype([("HR", "<u1"), ("GDH", "<f4")])
CLIMATE_DTYPE = np.dtype([("GDH_p50", "<f4")])

# Berlin: a valid coordinate for validate_gps and the GPS timezone lookup
LAT: Final = 52.52
LON: Final = 13.41


def _weather(snapped_hour: int, length: int = 240) -> np.ndarray:
    arr = np.empty(length, dtype=WEATHER_DTYPE)
    arr["HR"] = snapped_hour
    arr["GDH"] = np.arange(length, dtype=np.float32)
    return arr


def _climate(length: int = 672, step: float = 0.5) -> np.ndarray:
    arr = np.empty(length, dtype=CLIMATE_DTYPE)
    arr["GDH_p50"] = np.arange(length, dtype=np.float32) * step
    return arr


def _run(
    start: datetime,
    snapped_hour: int,
    frequency: Frequency,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weather_arr = _weather(snapped_hour)
    climate_arr = _climate()
    with (
        patch("gdhprediction.apis.ClimateGdhData") as climate_cls,
        patch("gdhprediction.apis.WeatherGdhData") as weather_cls,
    ):
        climate_cls.from_params.return_value.forecast_gdh.return_value = climate_arr
        weather_cls.from_params.return_value.forecast_gdh.return_value = weather_arr
        prediction = get_gdh_prediction(
            LAT,
            LON,
            GdhPredictionConfig(),
            start,
            frequency=frequency,
        )
    return prediction, weather_arr, climate_arr


class TestStitchContinuity(TestCase):
    """Verify the seam between weather and climate is continuous and uses
    the snapped hour to index into the climate array.
    """

    def assertSeamInvariants(
        self,
        prediction: np.ndarray,
        weather: np.ndarray,
        climate: np.ndarray,
        snapped_hour: int,
    ) -> None:
        """Properties that must hold for any successful hourly stitch."""
        n_w = len(weather)
        # Climate is indexed from today-midnight; the slice starts at
        # `n_w + snapped_hour`, so the hourly prediction has that many
        # leading climate samples skipped.
        self.assertEqual(
            prediction.shape,
            (n_w + len(climate) - (n_w + snapped_hour),),
        )
        # Weather portion is included verbatim.
        np.testing.assert_array_almost_equal(prediction[:n_w], weather["GDH"])
        # Seam is continuous: climate_part[0] anchors on weather["GDH"][-1].
        self.assertAlmostEqual(
            float(prediction[n_w]),
            float(weather["GDH"][-1]),
            places=4,
        )
        # Slope past the seam equals the climate array's intrinsic step.
        climate_step = float(climate["GDH_p50"][1] - climate["GDH_p50"][0])
        self.assertAlmostEqual(
            float(prediction[n_w + 1] - prediction[n_w]),
            climate_step,
            places=4,
        )

    def test_daily_at_midnight(self) -> None:
        # snapped_hour=0 since 0 % 6 == 0
        prediction, weather, climate = _run(
            start=datetime(2024, 6, 15, 0, 0),
            snapped_hour=0,
            frequency="daily",
        )
        # Daily sampling is hourly_prediction[23::24]. With snapped_hour=0
        # the hourly length equals len(climate) (weather contributes its
        # full length and the climate slice contributes len(climate) - n_w).
        self.assertEqual(prediction.shape, (len(climate) // 24,))
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
        # 12:00 is on a 6h boundary, so snapped_hour == 12.
        prediction, weather, climate = _run(
            start=datetime(2024, 6, 15, 12, 0),
            snapped_hour=12,
            frequency="hourly",
        )
        self.assertSeamInvariants(prediction, weather, climate, snapped_hour=12)

    def test_hourly_off_six_hour_boundary(self) -> None:
        """Test regression from snapped hour to rounded hour."""
        # 14:30 → snapped to 12:00, so snapped_hour=12
        prediction, weather, climate = _run(
            start=datetime(2024, 6, 15, 14, 30),
            snapped_hour=12,
            frequency="hourly",
        )
        self.assertSeamInvariants(prediction, weather, climate, snapped_hour=12)


BERLIN: Final = ZoneInfo("Europe/Berlin")
TOKYO: Final = ZoneInfo("Asia/Tokyo")


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
