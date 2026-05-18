import logging
from collections.abc import Set
from dataclasses import InitVar, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, ClassVar, Self, TypedDict, cast
from zoneinfo import ZoneInfo

import numpy as np
from requests import RequestException

from gdhprediction.config import GdhPredictionConfig
from gdhprediction.request import CacheManager, DataRequest, GdhData, RequestApiError
from gdhprediction.utils import (
    BRIGHTSKY_API,
    BRIGHTSKY_FIELDS,
    BrightSkyDataArray,
    BrightSkyDType,
    BrightSkyFields,
    FieldsHashMxn,
    HourlyForecastGhdDType,
    NamedDataArray,
    coerce_tz_datetime,
    get_gps_tzinfo,
    rh_from,
    sunshine_to_radiation,
    to_month_day_hour,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class BrightSkyRecord(TypedDict):

    timestamp: str
    temperature: float
    relative_humidity: float
    sunshine: float


@dataclass(frozen=True, slots=True)
class WeatherRequestData(FieldsHashMxn):

    _MAX_DURATION: ClassVar[int] = 10  # days
    SNAPPED: ClassVar[int] = 6  # hours

    lat: float
    lon: float
    start_date: InitVar[datetime | None] = field(default=None, repr=False)
    duration: InitVar[int] = field(default=_MAX_DURATION, repr=False)
    date: str = field(init=False)
    last_date: str = field(init=False)
    gps_tzinfo: ZoneInfo = field(init=False)

    def __post_init__(self, start_date: datetime | None, duration: int) -> None:
        # derive correct timezone from lat/lon
        _tzinfo = get_gps_tzinfo(self.lat, self.lon)
        start_dt: datetime = coerce_tz_datetime(start_date, _tzinfo)
        # snap hour to nearest previous multiple of _SNAPPED
        _snapped_hour = (start_dt.hour // self.SNAPPED) * self.SNAPPED
        _start = start_dt.replace(hour=_snapped_hour, minute=0, second=0, microsecond=0)
        iso_start_datetime = _start.isoformat()
        iso_end_datetime = (_start + timedelta(days=duration)).isoformat()
        object.__setattr__(self, "date", iso_start_datetime)
        object.__setattr__(self, "last_date", iso_end_datetime)
        object.__setattr__(self, "gps_tzinfo", _tzinfo)

    @property
    def latitude(self) -> float:
        return self.lat

    @property
    def longitude(self) -> float:
        return self.lon


@dataclass(frozen=True, slots=True)
class BrightskyDataRequest(DataRequest):

    # API configuration
    _API_NAME = "Brightsky"
    _API_URL = BRIGHTSKY_API
    _DEFAULT_API_VERSION = "v2"
    _REQUIRED: ClassVar[Set[BrightSkyFields]] = BRIGHTSKY_FIELDS
    _TOP_LEVEL: ClassVar[str] = "weather"

    request_data: WeatherRequestData

    def _get_user_agent(self, api_version: str) -> str:
        return f"GdhPrediction/1.0 (brightsky-client/{api_version})"

    def request_brightsky_data(self) -> BrightSkyDataArray:
        response = self._perform_request(self._api_version)

        # process response
        _records = response.json()[self._TOP_LEVEL]
        required: list[BrightSkyRecord] = self._filter_records(_records)
        LOGGER.info(
            "Bright Sky fetch succeeded: %d records.",
            len(required),
        )

        processed = self._process_brightsky_data(required)

        return processed

    def _filter_records(self, records: list[dict[str, Any]]) -> list[BrightSkyRecord]:
        filtered: list[BrightSkyRecord] = []
        for record in records:
            if missing := self._REQUIRED - record.keys():
                msg = f"Record missing required fields: {missing}. Record: {record}"
                LOGGER.warning(msg)
                raise ValueError(msg)
            reduced: dict[BrightSkyFields, Any] = {k: record[k] for k in self._REQUIRED}
            # check timestamp is valid
            if not isinstance(reduced["timestamp"], str):
                msg = (
                    f"Invalid timestamp type: {type(reduced['timestamp'])}. "
                    f"Record: {reduced}"
                )
                LOGGER.warning(msg)
                raise TypeError(msg)
            # check remaining numeric fields; compute RH if not directly reported
            if reduced["relative_humidity"] is None:
                reduced["relative_humidity"] = rh_from(
                    reduced["temperature"],
                    reduced["dew_point"],
                )
            reduced.pop("dew_point")  # not needed anymore
            numeric_fields = reduced.keys() - {"timestamp"}
            if any(not isinstance(reduced[nf], (int, float)) for nf in numeric_fields):
                msg = (
                    f"Invalid field type in record: {reduced}. "
                    f"Expected numeric types for {list(numeric_fields)}."
                )
                LOGGER.warning(msg)
                raise TypeError(msg)
            filtered.append(cast("BrightSkyRecord", reduced))
        return filtered

    def _process_brightsky_data(
        self,
        records: list[BrightSkyRecord],
    ) -> BrightSkyDataArray:
        lat = self.request_data.latitude
        sorted_records = sorted(records, key=lambda r: r["timestamp"])
        brightsky_array = np.empty(len(sorted_records), dtype=BrightSkyDType)
        for row, record in enumerate(sorted_records):
            try:
                date = datetime.fromisoformat(record["timestamp"])
            except ValueError as e:
                msg = f"Invalid timestamp format: {record['timestamp']}."
                LOGGER.warning(msg)
                raise ValueError(msg) from e
            mo, day, hr = to_month_day_hour(date)
            toy = date.timetuple().tm_yday
            radiation = sunshine_to_radiation(record["sunshine"], lat, toy, hr)
            brightsky_array[row] = np.array(
                (
                    mo,
                    day,
                    hr,
                    record["temperature"],
                    record["relative_humidity"],
                    radiation,
                ),
                dtype=BrightSkyDType,
            )
        return brightsky_array


class WeatherGdhData(GdhData[WeatherRequestData]):

    PARAMETERS: ClassVar[tuple[BrightSkyFields, ...]] = tuple(
        BRIGHTSKY_FIELDS - {"dew_point"},
    )
    _STALE_LOOKBACK: ClassVar[int] = 48  # hours

    @classmethod
    def from_params(
        cls,
        latitude: float,
        longitude: float,
        config: GdhPredictionConfig,
        *,
        start_datetime: datetime | None = None,
        duration: int = 10,
        **kwargs: Any,
    ) -> Self:
        request_data = WeatherRequestData(latitude, longitude, start_datetime, duration)
        return cls(request_data, config, cache_persistent=False)

    def _request_raw_data(self) -> BrightSkyDataArray:
        cdr = BrightskyDataRequest(self._request_data)
        return cdr.request_brightsky_data()

    def _process(self, raw_data: BrightSkyDataArray) -> BrightSkyDataArray:
        inner_t2m = self.transform(raw_data)
        gdh_base = self._config.gdh_base
        ghd = np.maximum(0, inner_t2m - gdh_base).round(2)
        hourly_gdh = np.empty(ghd.shape[0], dtype=HourlyForecastGhdDType)
        hourly_gdh["MO"] = raw_data["MO"]
        hourly_gdh["DY"] = raw_data["DY"]
        hourly_gdh["HR"] = raw_data["HR"]
        hourly_gdh["GDH"] = np.cumsum(ghd)
        return hourly_gdh

    def forecast_gdh(self, /, **kwargs: Any) -> BrightSkyDataArray:
        return self.data_array if self.data_array is not None else self.get_gdh_data()

    def _refresh_caches(self) -> NamedDataArray:
        try:
            data_array = super()._refresh_caches()
        except (RequestException, RequestApiError):
            LOGGER.exception("Network error refreshing weather cache")
        except OSError:
            LOGGER.exception("Cache I/O error")
        except (ValueError, TypeError, KeyError):
            LOGGER.exception("Unexpected data format from Brightsky")
        else:
            return data_array
        # if we get here, the refresh failed; try to find stale cache
        LOGGER.critical("Failed to refresh weather cache. Checking stale cache.")
        looks = max(1, self._STALE_LOOKBACK // WeatherRequestData.SNAPPED)
        # the snapped request time
        now = datetime.fromisoformat(self._request_data.date)
        step = timedelta(hours=WeatherRequestData.SNAPPED)
        for look in range(1, looks + 1):
            look_time = now - look * step
            # replace the field; the `duration` defaults to _MAX_DURATION
            stale_request_data = replace(self._request_data, start_date=look_time)
            # recreate CacheManager with stale hash
            _cache_manager = CacheManager(
                stale_request_data.compute_hash(),
                self._config.compute_hash(),
                self._cache_paths,
                self._cache_manager.persistent,
            )
            # try only processed; it's unlikely 'raw' would be available
            stale_array = _cache_manager.load_from_cache("processed")
            if stale_array is not None:
                LOGGER.critical(
                    "Using stale weather data from %s (lookback: %d hours).",
                    look_time,
                    look * WeatherRequestData.SNAPPED,
                )
                return stale_array
        msg = "Could not fetch weather data and no stale data cache available."
        LOGGER.error(msg)
        raise RuntimeError(msg)
