from datetime import date, datetime
from typing import Annotated, Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import typer

from gdhprediction.climate import ClimateGdhData
from gdhprediction.config import GdhPredictionConfig
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
    latitude: float,
    longitude: float,
    config: GdhPredictionConfig,
    maybe_datetime: date | datetime | None = None,
    frequency: Frequency = "hourly",
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """Get GDH prediction for the given location and configuration."""
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
    cgd = ClimateGdhData.from_params(
        latitude,
        longitude,
        config,
        start_year=LAST_YEAR - CLIMATE_YEARS,
        **kwargs,
    )
    climate_data = cgd.forecast_gdh(start_date=start_datetime)
    wgd = WeatherGdhData.from_params(
        latitude,
        longitude,
        config,
        start_datetime=start_datetime,
        **kwargs,
    )
    weather_data = wgd.forecast_gdh()  # start_datetime passed through constructor

    # Connect weather and climate at the proper hour
    _snapped_hour: int = int(weather_data["HR"][0])
    climate_start: int = len(weather_data) + _snapped_hour
    climate_part = (
        climate_data["GDH_p50"][climate_start:]
        - climate_data["GDH_p50"][climate_start]
        + weather_data["GDH"][-1]
    )
    prediction = np.concatenate((weather_data["GDH"], climate_part))

    if frequency == "daily":
        # aggregate hourly to daily by taking the last value of each day
        prediction = prediction[23::24]

    return prediction


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
