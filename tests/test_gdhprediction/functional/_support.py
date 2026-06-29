"""Shared scaffolding for the gdhprediction functional tests.

These helpers inject synthetic raw weather/climate arrays at the
``_request_raw_data`` seam (with the disk cache forced to miss) so the real
``transform -> max(0, T - base) -> cumsum -> crop -> stitch`` pipeline runs end
to end. Each transform's test supplies its own temperature/irradiance signal and
the inner-temperature relation it expects; the GDH accumulation and
prediction-array math live here so every transform checks it the same way.
"""

from datetime import datetime
from typing import Final
from unittest.mock import patch

import numpy as np

from gdhprediction.apis import Frequency, get_gdh_prediction
from gdhprediction.climate import ClimateGdhData
from gdhprediction.config import GdhLocalizedPredictionConfig
from gdhprediction.request import CacheManager
from gdhprediction.utils import (
    HOURS_PER_YEAR,
    BrightSkyDataArray,
    BrightSkyDType,
    Float32,
    Float64,
    NasaDataArray,
    hourly_cache_dtype,
)
from gdhprediction.weather import WeatherGdhData, WeatherRequestData
from tests._base import ArrayTestCase

# Berlin: valid for validate_gps and the GPS timezone lookup.
LAT: Final = 52.52
LON: Final = 13.41
# Default GdhPredictionConfig.gdh_base, stated so expected increments are explicit.
GDH_BASE: Final = 4.5
# Cumulative GDH must never decrease; allow only float round-off below zero.
MONOTONIC_TOL: Final = -1e-6

# Standard start on a 6-hour snap boundary: the snapped hour equals the datetime
# hour, so the weather crop starts at index 0 (no leading hours dropped).
START: Final = datetime(2024, 6, 15, 12, 0)
SNAPPED_HOUR: Final = 12  # (12 // 6) * 6
WEATHER_HOURS: Final = WeatherRequestData.MAX_DURATION * 24  # 240
# Horizon short enough to stay inside the forecast window: 216 h < WEATHER_HOURS,
# so the climate leg is never reached and the output is purely transform-derived.
IN_FORECAST_DAYS: Final = 9


def step_series(low: float, high: float, *, at: int, length: int) -> Float64:
    """Return a length-``length`` array stepping from ``low`` to ``high`` at ``at``."""
    idx = np.arange(length)
    return np.where(idx < at, low, high).astype(np.float64)


def make_weather_raw(
    *,
    temperature: Float64,
    irradiance: Float64 | float = 0.0,
    snapped_hour: int = SNAPPED_HOUR,
) -> BrightSkyDataArray:
    """Build a BrightSky raw array from a per-hour temperature (and irradiance).

    ``HR[0]`` carries the snapped hour the apis crop reads back; RH is a benign
    constant and irradiance defaults to zero (inert under temperature-only
    transforms). ``temperature`` sets the array length.
    """
    length = temperature.shape[0]
    arr = np.zeros(length, dtype=BrightSkyDType)
    arr["HR"] = (np.arange(length) + snapped_hour) % 24
    arr["MO"] = 6
    arr["DY"] = 15
    arr["RH2M"] = 70.0
    arr["T2M"] = temperature
    arr["ALLSKY_SFC_SW_DWN"] = irradiance
    return arr


def make_climate_raw_constant(t2m: float, *, num_years: int = 3) -> NasaDataArray:
    """Build a NASA raw array (8760 h x years) at a constant temperature.

    Identical years collapse the per-hour percentile to that single value, so
    the climate leg contributes a constant, exactly predictable GDH slope of
    ``max(0, t2m - GDH_BASE)`` per hour.
    """
    arr = np.zeros(HOURS_PER_YEAR, dtype=hourly_cache_dtype(num_years))
    arr["HR"] = np.tile(np.arange(24), HOURS_PER_YEAR // 24)
    arr["T2M"] = t2m
    arr["RH2M"] = 70.0
    arr["ALLSKY_SFC_SW_DWN"] = 0.0
    return arr


def predict(
    config: GdhLocalizedPredictionConfig,
    *,
    weather_raw: BrightSkyDataArray,
    climate_raw: NasaDataArray,
    duration: int,
    frequency: Frequency = "hourly",
) -> Float32:
    """Run get_gdh_prediction from ``START`` with raw arrays injected, cache missed.

    The disk cache load returns None and writes are no-ops, so the synthetic raw
    arrays always flow through ``_process``.
    """
    with (
        patch.object(WeatherGdhData, "_request_raw_data", return_value=weather_raw),
        patch.object(ClimateGdhData, "_request_raw_data", return_value=climate_raw),
        patch.object(CacheManager, "load_from_cache", return_value=None),
        patch.object(CacheManager, "write_to_cache", return_value=None),
    ):
        return get_gdh_prediction(config, START, frequency=frequency, duration=duration)


def per_hour_gdh(inner_temp: Float32, *, base: float = GDH_BASE) -> Float32:
    """Return the per-hour GDH the pipeline produces from an inner-temperature series.

    Mirrors ``_process``: ``max(0, inner - base)`` rounded to 2 dp.
    """
    return np.maximum(0, inner_temp.astype(np.float32) - base).round(2)


def expected_prediction(
    inner_temp: Float32,
    *,
    duration_hours: int,
    base: float = GDH_BASE,
) -> Float32:
    """Return the GDH prediction the pipeline produces from an inner-temp series.

    Mirrors ``_process`` plus the apis crop/anchor: per-hour GDH is cumulatively
    summed, truncated to ``duration_hours`` and re-anchored so the first value
    is zero (valid when the weather crop starts at index 0, i.e. a snap-boundary
    start, and the horizon fits inside the forecast window).
    """
    cumulative = np.cumsum(per_hour_gdh(inner_temp, base=base))
    return (cumulative[:duration_hours] - cumulative[0]).astype(np.float32)


class VariabilityTestCase(ArrayTestCase):
    """Base for the transform-variability tests; shares the pipeline check."""

    def assert_pipeline_matches(
        self,
        config: GdhLocalizedPredictionConfig,
        *,
        weather_raw: BrightSkyDataArray,
        indoor: Float32,
        duration: int,
        climate_t2m: float = 12.0,
        atol: float = 1.5e-4,
    ) -> tuple[Float32, Float32]:
        """Predict from ``START`` and assert the output reproduces the GDH pipeline.

        Checks the four invariants every transform shares for a horizon that fits
        inside the forecast window: the output has the full hourly shape, equals
        ``expected_prediction(indoor)``, its hourly diffs equal
        ``per_hour_gdh(indoor)[1:]``, and the cumulative GDH never decreases.
        Returns ``(prediction, increments)`` for the transform-specific asserts.
        """
        duration_hours = duration * 24
        prediction = predict(
            config,
            weather_raw=weather_raw,
            climate_raw=make_climate_raw_constant(climate_t2m),
            duration=duration,
        )
        expected = expected_prediction(indoor, duration_hours=duration_hours)
        self.assertEqual(prediction.shape, (duration_hours,))
        self.assertAllClose(prediction, expected, atol=atol)
        increments = np.diff(prediction)
        self.assertAllClose(
            increments,
            per_hour_gdh(indoor)[1:duration_hours],
            atol=atol,
        )
        self.assertTrue(np.all(increments >= MONOTONIC_TOL))
        return prediction, increments
