"""Weather leg: the BrightSky/DWD short-range forecast reduced to cumulative GDH."""

import logging
from collections.abc import Container, Mapping
from collections.abc import Set as AbstractSet
from dataclasses import InitVar, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, ClassVar, Self, TypedDict, cast

import numpy as np
from requests import RequestException

from gdhprediction.config import GdhPredictionConfig
from gdhprediction.request import CacheManager, DataRequest, GdhData, RequestApiError
from gdhprediction.utils import (
    BRIGHTSKY_API,
    BRIGHTSKY_FIELDS,
    HOURS_PER_DAY,
    PARAMETERS,
    BrightSkyDataArray,
    BrightSkyDType,
    BrightSkyFields,
    FieldsHashMxn,
    HourlyForecastGhdDType,
    Int16,
    NasaParameters,
    StructuredDataArray,
    TzDatetime,
    coerce_tz_datetime,
    get_gps_tzinfo,
    or_else,
    rh_from,
    sunshine_to_radiation,
    to_month_day_hour,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

_MIN_INTERP_POINTS = 2  # np.interp needs at least two anchors


class BrightSkyRecord(TypedDict):
    """The BrightSky fields consumed from each hourly observation."""

    timestamp: TzDatetime
    temperature: float
    relative_humidity: float
    sunshine: float


@dataclass(frozen=True, slots=True)
class WeatherRequestData(FieldsHashMxn):
    """Frozen BrightSky request params; snaps the start hour to a 6-hour boundary."""

    MAX_DURATION: ClassVar[int] = 10  # days
    SNAPPED: ClassVar[int] = 6  # hours

    lat: float
    lon: float
    start_date: InitVar[datetime | None] = field(default=None, repr=False)
    duration: InitVar[int] = field(default=MAX_DURATION, repr=False)
    date: str = field(init=False)
    last_date: str = field(init=False)

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

    @property
    def latitude(self) -> float:
        return self.lat

    @property
    def longitude(self) -> float:
        return self.lon


def _coerce_to_float[S: str](
    value: Mapping[S, Any],
    drop: Container[S] = (),
) -> dict[S, float]:
    """Coerce mapping values to float, raising TypeError on non-numerics.

    ``None`` becomes ``NaN``; keys listed in ``drop`` are omitted from the result.
    """
    coerced: dict[S, float] = {}
    for k, v in value.items():
        if k in drop:
            continue
        try:
            coerced[k] = float(or_else(v, np.nan))
        except (TypeError, ValueError) as terr:
            msg = f"Cannot coerce value to float: {v} (key: {k})"
            LOGGER.warning(msg)
            raise TypeError(msg) from terr
    return coerced


@dataclass(frozen=True, slots=True)
class BrightskyDataRequest(DataRequest):
    """DataRequest for the BrightSky/DWD weather endpoint."""

    # API configuration
    _API_NAME = "Brightsky"
    _API_URL = BRIGHTSKY_API
    _DEFAULT_API_VERSION = "v2"
    _REQUIRED: ClassVar[AbstractSet[BrightSkyFields]] = BRIGHTSKY_FIELDS
    _INTERPOLATED: ClassVar[AbstractSet[NasaParameters]] = PARAMETERS
    _FIELD_LABELS: ClassVar[Mapping[str, str]] = {
        "T2M": "temperature",
        "RH2M": "relative_humidity",
        "ALLSKY_SFC_SW_DWN": "sunshine",
    }
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

        return self._process_brightsky_data(required)

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
            # convert timestamp to datetime
            try:
                timestamp = TzDatetime(datetime.fromisoformat(reduced["timestamp"]))
            except ValueError as verr:
                msg = (
                    f"Invalid timestamp format: {reduced['timestamp']}. "
                    f"Record: {reduced}"
                )
                LOGGER.warning(msg)
                raise ValueError(msg) from verr
            # check numeric fields; tolerate None but transform to np.nan <: float
            try:
                # numerics does not include timestamp
                numerics: dict[BrightSkyFields, float] = _coerce_to_float(
                    reduced,
                    drop={"timestamp"},
                )
            except TypeError as terr:
                msg = (
                    f"Invalid field type in record: {reduced}. "
                    f"Expected numeric types for {set(reduced) - {"timestamp"}}."
                )
                LOGGER.warning(msg)
                raise TypeError(msg) from terr
            # try to compute RH if not reported; if dew_point is missing, RH will be NaN
            if np.isnan(numerics["relative_humidity"]):
                numerics["relative_humidity"] = rh_from(
                    numerics["temperature"],
                    numerics["dew_point"],
                )
            numerics.pop("dew_point")  # not needed anymore
            # append to filtered list
            filtered.append(
                cast("BrightSkyRecord", numerics | {"timestamp": timestamp}),
            )
        return filtered

    def _interpolate_nans(self, data_array: BrightSkyDataArray) -> BrightSkyDataArray:
        """Interpolate NaN values in the BrightSky data array.

        The interpolation happens in two steps:

        #. Backfill leading NaNs from the same hour one day later.
        #. Linearly interpolate short intermittent NaN runs, with a maximum allowed gap
            of one day (24 hours). If a gap exceeds this limit, a ValueError is raised.

        .. warning::

            This method modifies the input data_array in-place.
            If you need to preserve the original data,
            make a copy before calling this method.

        """
        # Backfill leading NaNs from the same hour one day later.
        for afield in self._INTERPOLATED:
            label = f"{self._FIELD_LABELS.get(afield, afield)} ({afield})"
            field_data = data_array[afield]
            if not np.isnan(field_data[0]):
                continue
            # field_data[0] is NaN here, so argmax==0 means the column is all NaN.
            first_valid = int(np.argmax(~np.isnan(field_data)))
            if (
                not first_valid
                or first_valid > HOURS_PER_DAY
                or field_data.size < first_valid + HOURS_PER_DAY
            ):
                msg = (
                    f"Leading NaN run is {first_valid} h of {field_data.size} h "
                    f"for field {label}; expected under a day."
                )
                raise ValueError(msg)
            next_day = field_data[HOURS_PER_DAY : HOURS_PER_DAY + first_valid]
            field_data[:first_valid] = next_day
        # Linearly interpolate short intermittent NaN runs.
        for afield in self._INTERPOLATED:
            label = f"{self._FIELD_LABELS.get(afield, afield)} ({afield})"
            field_data = data_array[afield]
            nan_mask = np.isnan(field_data)
            if not nan_mask.any():
                continue
            idx = np.arange(nan_mask.size)
            # running max of the last valid index carries across NaN gaps, so
            # idx - prev_valid is the length of the NaN run ending at each position.
            prev_valid = np.maximum.accumulate(np.where(nan_mask, -1, idx))
            worst = int((idx - prev_valid).max())
            nan_indices = np.flatnonzero(nan_mask)
            LOGGER.warning(
                "Field %s has %d NaNs (longest run %d h) at %s. Interpolating.",
                label,
                int(nan_mask.sum()),
                worst,
                nan_indices.tolist(),
            )
            if worst > HOURS_PER_DAY:
                msg = f"Field {label} NaN gap is {worst} h; too wide to interpolate."
                raise ValueError(msg)
            valid_indices = np.flatnonzero(~nan_mask)
            if valid_indices.size < _MIN_INTERP_POINTS:
                msg = f"Not enough valid data to interpolate field {label}."
                raise ValueError(msg)
            field_data[nan_indices] = np.interp(
                nan_indices,
                valid_indices,
                field_data[valid_indices],
            )
        return data_array

    def _process_brightsky_data(
        self,
        records: list[BrightSkyRecord],
    ) -> BrightSkyDataArray:
        lat = self.request_data.latitude
        sorted_records = sorted(records, key=lambda r: r["timestamp"])
        brightsky_array = np.empty(len(sorted_records), dtype=BrightSkyDType)
        toys: Int16 = np.empty(len(sorted_records), dtype=np.int16)
        for row, record in enumerate(sorted_records):
            date = record["timestamp"]
            mo, day, hr = to_month_day_hour(date)
            toy = date.timetuple().tm_yday
            toys[row] = toy
            brightsky_array[row] = np.array(
                (
                    mo,
                    day,
                    hr,
                    record["temperature"],
                    record["relative_humidity"],
                    record["sunshine"],  # interpolate sunshine
                ),
                dtype=BrightSkyDType,
            )
        brightsky_array = self._interpolate_nans(brightsky_array)
        brightsky_array["ALLSKY_SFC_SW_DWN"] = sunshine_to_radiation(
            brightsky_array["ALLSKY_SFC_SW_DWN"],
            lat,
            toys,
            brightsky_array["HR"],
        )
        return brightsky_array


class WeatherGdhData(GdhData[WeatherRequestData]):
    """Weather leg: BrightSky forecast reduced to cumulative GDH."""

    __slots__ = ()

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
        duration: int = WeatherRequestData.MAX_DURATION,
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

    def _refresh_caches(self) -> StructuredDataArray:
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
