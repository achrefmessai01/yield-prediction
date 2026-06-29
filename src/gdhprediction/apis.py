"""Public GDH-prediction entry points: stitch BrightSky forecast onto climatology."""

import logging
from datetime import date, datetime
from typing import Annotated, Any, Final, Literal
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import typer

from gdhprediction.climate import ClimateGdhData
from gdhprediction.config import APP_PATHS, GdhLocalizedPredictionConfig
from gdhprediction.request import CacheManager, CacheScope
from gdhprediction.utils import (
    CLIMATE_YEARS,
    LAST_YEAR,
    BrightSkyDataArray,
    TzDatetime,
    coerce_tz_datetime,
    get_gps_tzinfo,
    validate_gps,
)
from gdhprediction.weather import (
    BrightskyDataRequest,
    WeatherGdhData,
    WeatherRequestData,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

Frequency = Literal["hourly", "daily"]
FORECAST_HORIZON: Final[int] = WeatherRequestData.MAX_DURATION * 24
"""Expected weather forecast horizon in hours"""
_FIXED_HORIZON: Final[int] = FORECAST_HORIZON - WeatherRequestData.SNAPPED
"""Returned horizon in hours after dropping hours between now and the snapped hour."""

cli = typer.Typer(no_args_is_help=True, add_completion=False)


def get_gdh_prediction(
    localized_config: GdhLocalizedPredictionConfig,
    maybe_datetime: date | datetime | None = None,
    frequency: Frequency = "hourly",
    duration: int = 28,
    **kwargs: Any,
) -> npt.NDArray[np.float32]:
    """Get GDH prediction for the given location and configuration.

    For "hourly" frequency, the prediction starts from the hour of `maybe_datetime`
    (or current hour if None) and includes `duration * 24` hourly values.
    For "daily" frequency, the prediction starts from the midnight of `maybe_datetime`
    (or current day if None) and includes `duration` daily values
    (last value of each day).
    If the weather forecast does not provide enough data for the requested duration,
    the prediction is extended with climate data. If the weather forecast provides no
    data, the prediction is entirely based on climate data.

    :param localized_config: the GPS-localized configuration for the prediction.
    :param maybe_datetime: optional date or datetime to start the prediction from.
        If None, defaults to the current date and time in the GPS timezone.
    :param frequency: "hourly" or "daily" output frequency.
    :param duration: number of days to predict.
    :param kwargs: forwarded to the weather/climate data builders;
        unknown keys are ignored.
    :return: a copy of the GDH prediction array, with shape (duration,) for daily
        and (duration * 24,) for hourly frequency.
    """
    # unpack the GPS-localized config into its components
    latitude, longitude, config = localized_config.unpack()
    # validate GPS
    validate_gps(latitude, longitude)
    # daily output requires start_datetime to start at midnight in the local timezone
    start_datetime: TzDatetime
    gps_timezone = get_gps_tzinfo(latitude, longitude)
    if frequency == "daily":
        start_datetime = _coerce_tz_midnight(maybe_datetime, gps_timezone)
    else:  # "hourly" is default and does not require adjustment
        start_datetime = coerce_tz_datetime(maybe_datetime, gps_timezone)

    # construct data objects
    # always request the full WeatherRequestData.MAX_DURATION weather window
    wgd = WeatherGdhData.from_params(
        latitude,
        longitude,
        config,
        start_datetime=start_datetime,
        **kwargs,
    )
    _weather_data = wgd.forecast_gdh()  # start_datetime passed through constructor
    datetime_hour: int = start_datetime.hour
    duration_hours: int = duration * 24
    # crop the weather data to the requested hour
    if len(_weather_data) == 0:  # returned no forecast hours
        weather_data = _weather_data
        LOGGER.warning(
            "No weather records within horizon for %s; prediction is climatology-only.",
            start_datetime,
        )
    else:  # returned at least some forecast hours
        _snapped_hour: int = int(_weather_data["HR"][0])
        weather_start: int = datetime_hour - _snapped_hour  # >= 0
        weather_data = _weather_data[weather_start:].copy()

    if len(weather_data) > 0:  # can be empty
        weather_data["GDH"] -= weather_data["GDH"][0]  # set first value to 0
        if len(weather_data) >= duration_hours:
            # requested duration covered by weather data range
            if frequency == "daily":
                return weather_data["GDH"][23::24][:duration].copy()
            return weather_data["GDH"][:duration_hours].copy()

    # need to extend weather data with climate data
    cgd = ClimateGdhData.from_params(
        latitude,
        longitude,
        config,
        start_year=LAST_YEAR - CLIMATE_YEARS,
        **kwargs,
    )
    # hourly forecast needs extra day to meet the expected number of hours
    _duration = duration + 1 if frequency == "hourly" else duration
    climate_data = cgd.forecast_gdh(start_date=start_datetime, forecast=_duration)

    # Connect weather and climate at the proper hour.
    # BrightSky labels every record with the offset at start_datetime.
    # NASA POWER LST is local standard time year-round.
    # So when start_datetime is in DST, the weather frame runs one civil hour ahead
    # of climate and the seam must be pulled back by one climate hour to stay aligned.
    dst_correction: int = -1 if (len(weather_data) > 0 and start_datetime.dst()) else 0
    climate_start: int = len(weather_data) + datetime_hour + dst_correction
    weather_gdh_end = weather_data["GDH"][-1] if len(weather_data) > 0 else 0.0
    climate_part = (
        climate_data["GDH_p50"][climate_start:]
        - climate_data["GDH_p50"][climate_start]
        + weather_gdh_end
    )
    prediction = np.concatenate((weather_data["GDH"], climate_part))

    if frequency == "daily":
        # aggregate hourly to daily by taking the last value of each day
        # for daily, the `duration` matches the number of days
        return prediction[23::24].copy()

    return prediction[:duration_hours].copy()


def get_current_brightsky_forecast(
    latitude: float,
    longitude: float,
    *,
    cache: bool = True,
) -> BrightSkyDataArray:
    """Get the BrightSky forecast for the given GPS coordinates as of call time.

    The horizon of the forecast determined by the BrightSky API is 10 days.
    For the returned data structure see `BrightSkyDType` in :mod:`gdhprediction.utils`.

    :param latitude: GPS latitude of the location.
    :param longitude: GPS longitude of the location.
    :param cache: whether to cache the raw BrightSky data in the XDG cache directory.
    :return: the `BrightSkyDType` dtype named array for the requested forecast,
        the array has 234 rows (10 days * 24 hours - 6 hours snapping),
        fewer if the provider returns a short horizon (a warning is logged)
    :raises ValueError: if the GPS coordinates are invalid, or provider returns no data.
    """
    # validate GPS
    validate_gps(latitude, longitude)
    tzinfo = get_gps_tzinfo(latitude, longitude)
    forecast_start: TzDatetime = TzDatetime(datetime.now(tzinfo))
    # request the forecast; don't read from cache
    wrd = WeatherRequestData(latitude, longitude, forecast_start)
    bdr = BrightskyDataRequest(wrd)
    raw_brightsky_data = bdr.request_brightsky_data()
    # check the returned data for expected length
    if len(raw_brightsky_data) == 0:
        msg = "Forecast request returned no data from now."
        raise ValueError(msg)
    if len(raw_brightsky_data) < FORECAST_HORIZON:
        msg = (
            f"Forecast request returned only {len(raw_brightsky_data)} hours of data, "
            f"which is less than the expected {FORECAST_HORIZON} "
            f"hours for a full forecast horizon."
        )
        LOGGER.warning(msg)
    # drop hours between now and the snapped hour
    snapped_hour: int = int(raw_brightsky_data["HR"][0])
    now_hour: int = forecast_start.hour
    hours_to_drop: int = now_hour - snapped_hour  # >= 0
    # cache the untrimmed raw array (keeps the GDH path's raw cache consistent)
    if cache:
        cache_mgr = CacheManager(wrd.compute_hash(), "", APP_PATHS, persistent=False)
        cache_mgr.write_to_cache(raw_brightsky_data, cache_fmt="raw")
    return raw_brightsky_data[hours_to_drop:][:_FIXED_HORIZON].copy()


@cli.command("invalidate-cache")
def _cli_invalidate_cache(
    scope: Annotated[
        CacheScope,
        typer.Option(
            help=(
                "Which cache to clear: 'cache' (volatile weather), "
                "'data' (persistent climate), or 'all'."
            ),
        ),
    ] = "all",
) -> None:
    """Remove cached gdhprediction files from the XDG cache/data directories."""
    CacheManager.invalidate(scope=scope)


def _coerce_tz_midnight(
    start_datetime: date | datetime | None,
    tzinfo: ZoneInfo,
) -> TzDatetime:
    """Coerce to a tz-aware midnight in tzinfo."""
    dt: TzDatetime = coerce_tz_datetime(start_datetime, tzinfo)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)
