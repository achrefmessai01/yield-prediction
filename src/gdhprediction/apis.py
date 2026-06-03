from datetime import date, datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import typer

from gdhprediction.climate import ClimateGdhData
from gdhprediction.config import GdhLocalizedPredictionConfig
from gdhprediction.request import CacheManager, CacheScope
from gdhprediction.utils import (
    CLIMATE_YEARS,
    LAST_YEAR,
    TzDatetime,
    coerce_tz_datetime,
    get_gps_tzinfo,
    validate_gps,
)
from gdhprediction.weather import WeatherGdhData

Frequency = Literal["hourly", "daily"]

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

    :param localized_config: the GPS-localized configuration for the prediction.
    :param maybe_datetime: optional date or datetime to start the prediction from.
        If None, defaults to the current date and time in the GPS timezone.
    :param frequency: "hourly" or "daily" output frequency.
    :param duration: number of `days` to predict.
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
    # crop the weather data to the requested hour
    datetime_hour: int = start_datetime.hour
    _snapped_hour: int = int(_weather_data["HR"][0])
    duration_hours: int = duration * 24
    weather_start: int = datetime_hour - _snapped_hour  # >= 0
    weather_data = _weather_data[weather_start:].copy()
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

    # Connect weather and climate at the proper hour
    climate_start: int = len(weather_data) + datetime_hour
    climate_part = (
        climate_data["GDH_p50"][climate_start:]
        - climate_data["GDH_p50"][climate_start]
        + weather_data["GDH"][-1]
    )
    prediction = np.concatenate((weather_data["GDH"], climate_part))

    if frequency == "daily":
        # aggregate hourly to daily by taking the last value of each day
        # for daily, the `duration` matches the number of days
        return prediction[23::24].copy()

    return prediction[:duration_hours].copy()


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
    dt: TzDatetime = coerce_tz_datetime(start_datetime, tzinfo)
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)
