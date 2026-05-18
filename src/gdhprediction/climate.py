import logging
from calendar import isleap
from collections.abc import Iterable
from dataclasses import InitVar, dataclass, field
from datetime import date, datetime
from io import StringIO
from typing import Any, ClassVar, Final, Self
from zoneinfo import ZoneInfo

import numpy as np

from gdhprediction.config import GdhPredictionConfig
from gdhprediction.request import DataRequest, GdhData
from gdhprediction.utils import (
    DATETIME_UNITS,
    FEB,
    FIRST_YEAR,
    HOURS_PER_YEAR,
    LAST_YEAR,
    LAT_STEP,
    LON_STEP,
    MISSING_DATA,
    NASA_HOURLY_API,
    PERCENTILES,
    Communities,
    FieldsHashMxn,
    Formats,
    HourlyAnnualizedGdhDType,
    NasaDataArray,
    NasaDataDType,
    NasaParameters,
    TimeStandards,
    coerce_tz_datetime,
    get_gps_tzinfo,
    hourly_cache_dtype,
    is_datetime_unit,
    is_parameter,
    remove_leap,
)

LOGGER: Final = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # avoid "No handler found" warnings


class NasaApiError(Exception):
    """Custom exception for NASA API errors."""


@dataclass(frozen=True, slots=True)
class RequestData(FieldsHashMxn):
    """Permanent attributes correspond exactly to NASA API parameters."""

    # human friendly input parameters
    lat: InitVar[float]
    lon: InitVar[float]
    parameter_set: InitVar[Iterable[NasaParameters]]
    start_year: InitVar[int]
    end_year: InitVar[int]
    # actual representation to send to NASA API
    start: str = field(init=False)
    end: str = field(init=False)
    parameters: str = field(init=False)
    latitude: float = field(init=False)
    longitude: float = field(init=False)
    gps_tzinfo: ZoneInfo = field(init=False)
    community: Communities
    format: Formats
    time_standard: TimeStandards

    def __post_init__(
        self,
        lat: float,
        lon: float,
        parameter_set: Iterable[NasaParameters],
        start_year: int,
        end_year: int,
    ) -> None:
        """Validate and transform input parameters for NASA API."""
        start, end = self._valid_years(start_year, end_year)
        lat, lon = self._snap_coordinates(lat, lon)
        gps_tz = get_gps_tzinfo(lat, lon)
        # prepare for NASA API
        object.__setattr__(self, "start", f"{start}0101")
        object.__setattr__(self, "end", f"{end}1231")
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "longitude", lon)
        object.__setattr__(self, "parameters", ",".join(sorted(parameter_set)))
        object.__setattr__(self, "gps_tzinfo", gps_tz)

    def _valid_years(
        self,
        start_year: int | None,
        end_year: int | None,
    ) -> tuple[int, int]:
        last_year: int = date.today().year - 1
        start: int = start_year or FIRST_YEAR
        end: int = end_year or last_year
        if start > end:
            LOGGER.error(
                "Start year %d must be less than or equal to end year %d.",
                start,
                end,
            )
            start = FIRST_YEAR
        if start < FIRST_YEAR:
            LOGGER.warning("Start year %d must be at least %d.", start, FIRST_YEAR)
            start = FIRST_YEAR
        if end > last_year:
            LOGGER.warning("End year %d must be at most %d.", end, last_year)
            end = last_year
        return start, end

    def _snap_coordinates(self, lat: float, lon: float) -> tuple[float, float]:
        snap_lat = round(lat / LAT_STEP) * LAT_STEP
        snap_lon = round(lon / LON_STEP) * LON_STEP
        return snap_lat, snap_lon


class ClimateGdhData(GdhData[RequestData]):

    # default parameters for NASA API request
    COMMUNITY: ClassVar[Communities] = "ag"
    FORMAT: ClassVar[Formats] = "csv"
    TIME_STANDARD: ClassVar[TimeStandards] = "lst"
    PARAMETERS: ClassVar[tuple[NasaParameters, ...]] = (
        "T2M",
        "RH2M",
        "ALLSKY_SFC_SW_DWN",
    )

    @classmethod
    def from_params(
        cls,
        latitude: float,
        longitude: float,
        config: GdhPredictionConfig,
        *,
        start_year: int = FIRST_YEAR,
        end_year: int = LAST_YEAR,
        **kwargs: Any,
    ) -> Self:
        request_data = RequestData(
            latitude,
            longitude,
            cls.PARAMETERS,
            start_year,
            end_year,
            cls.COMMUNITY,
            cls.FORMAT,
            cls.TIME_STANDARD,
        )
        return cls(request_data, config, cache_persistent=True)

    def _request_raw_data(self) -> NasaDataArray:
        cdr = ClimateDataRequest(self._request_data)
        return cdr.request_nasa_data()

    def forecast_gdh(
        self,
        *,
        start_date: date | datetime | None = None,
        forecast: int = 28,
        **kwargs,
    ) -> NasaDataArray:
        """Forecast GDH for the next 28 days based on historical data."""
        _tzinfo = self._request_data.gps_tzinfo
        # make sure time zone is properly handled
        _today: date = coerce_tz_datetime(start_date, _tzinfo).date()
        # ensure climate data available
        data_array: NasaDataArray
        if self.data_array is None:
            data_array = self.get_gdh_data()
        else:
            data_array = self.data_array
        # find data between today and 28 days from now
        today_hour = (_today.timetuple().tm_yday - 1) * 24
        if isleap(_today.year) and _today.month > FEB:
            today_hour -= 24  # ignore leap day
            # Feb 29 behaves as Mar 1 in non-leap year
        last_hour = np.mod(today_hour + forecast * 24, HOURS_PER_YEAR)
        if last_hour > today_hour:
            forecast_cumgdh = data_array[today_hour:last_hour]
        else:
            forecast_cumgdh = np.concatenate(
                (data_array[today_hour:], data_array[:last_hour]),
            )
        ghd_cols = [f"CUMGDH_p{q:02d}" for q in PERCENTILES]
        anchor = data_array[today_hour]
        forecast_gdh = forecast_cumgdh[ghd_cols].copy()
        for ghd_col in ghd_cols:
            forecast_gdh[ghd_col] -= anchor[ghd_col]
        # rename columns to GDH_pXX
        forecast_gdh.dtype.names = tuple(f"GDH_p{q:02d}" for q in PERCENTILES)
        return forecast_gdh

    def _process(self, raw_data: NasaDataArray) -> NasaDataArray:
        """Process raw NASA data into GDH data format."""
        inner_t2m = self.transform(raw_data)
        gdh_base = self._config.gdh_base
        ghd = np.maximum(0, inner_t2m - gdh_base).round(2)
        # calculate GDH percentiles across years for each hour
        gdh_percentiles = np.percentile(ghd, PERCENTILES, axis=1)
        # create HourlyAnnualizedGdhDType array to hold the results
        hourly_gdh = np.empty(ghd.shape[0], dtype=HourlyAnnualizedGdhDType)
        hourly_gdh["MO"] = raw_data["MO"]
        hourly_gdh["DY"] = raw_data["DY"]
        hourly_gdh["HR"] = raw_data["HR"]
        for i, q in enumerate(PERCENTILES):
            hourly_gdh[f"CUMGDH_p{q:02d}"] = np.cumsum(gdh_percentiles[i])
        return hourly_gdh


@dataclass(frozen=True, slots=True)
class ClimateDataRequest(DataRequest):

    # class variables
    _API_URL = NASA_HOURLY_API
    _API_NAME = "NASA"
    _DEFAULT_API_VERSION: ClassVar[str] = "v2.5.9"

    # init fields
    request_data: RequestData

    # convenience fields for validation and processing
    parameters: set[NasaParameters] = field(init=False, repr=False)
    _start_year: int = field(init=False, repr=False)
    _end_year: int = field(init=False, repr=False)

    def __post_init__(self, api_version: str | None) -> None:
        # create session
        super(ClimateDataRequest, self).__post_init__(api_version)
        # extract validated parameters
        _start_year = int(self.request_data.start[:4])
        _end_year = int(self.request_data.end[:4])
        _parameter_set = set(self.request_data.parameters.split(","))
        object.__setattr__(self, "_start_year", _start_year)
        object.__setattr__(self, "_end_year", _end_year)
        object.__setattr__(self, "parameters", _parameter_set)

    def _get_user_agent(self, api_version: str) -> str:
        return f"GdhPrediction/1.0 (NASA-POWER-Client/{api_version})"

    def request_nasa_data(self) -> NasaDataArray:

        response = self._perform_request(self._api_version)
        csv_text: str = response.text

        # validate response content
        if not csv_text:
            msg = "Received empty response from NASA API."
            LOGGER.error(msg)
            raise NasaApiError(msg)

        line_count: int = csv_text.count("\n")
        exp_line_count: int = (self._end_year - self._start_year + 1) * HOURS_PER_YEAR

        if line_count < exp_line_count:
            msg = (
                f"Received incomplete data: expected at least {exp_line_count} lines, "
                f"got {line_count}."
            )
            LOGGER.error(msg)
            raise NasaApiError(msg)

        return self._process_nasa_data(csv_text)

    def _process_nasa_data(self, csv_text: str) -> NasaDataArray:
        # Split into lines
        lines = csv_text.strip().split("\n")
        data_start_idx = self._get_header_length(lines)

        # Extract data portion (headers + data rows)
        data_lines = lines[data_start_idx:]
        data_csv = "\n".join(data_lines)

        LOGGER.debug("Parsing %d data rows.", len(data_lines) - 1)

        # Parse with NumPy
        try:
            data: NasaDataArray = np.genfromtxt(
                StringIO(data_csv),
                delimiter=",",
                names=True,  # First line is column names
                dtype=NasaDataDType,  # Should match this type
                encoding="utf-8",
                filling_values=MISSING_DATA,  # NASA uses -999 for missing data
            )
        except Exception as e:
            msg = "Failed to parse CSV data returned by NASA."
            LOGGER.exception(msg)
            raise ValueError(msg) from e

        LOGGER.info("Parsed %d records with columns: %s", len(data), data.dtype.names)

        # Validate proper time units were returned
        missing_time_fields = DATETIME_UNITS - set(
            filter(is_datetime_unit, data.dtype.names or ()),
        )
        if missing_time_fields:
            msg = f"Missing required time fields: {missing_time_fields}"
            LOGGER.error(msg)
            raise ValueError(msg)

        # Validate we got expected parameter columns
        actual: set[NasaParameters] = set(
            filter(is_parameter, (data.dtype.names or ())),
        )

        if not self.parameters.issubset(actual):
            missing: set[NasaParameters] = self.parameters - actual
            msg = f"Missing expected columns: {missing}"
            raise ValueError(msg)

        # process the actual content

        # remove leap days; reuse the same reference
        data = remove_leap(data)
        # replace -999 with np.nan; no copy
        for param in self.parameters:
            data[param][data[param] == MISSING_DATA] = np.nan

        # reindex by MO, DY, HR triplets and reshape data to (8760, years)
        final = self._reindex(data)

        # validate final shape is 8760
        if final.shape[0] != HOURS_PER_YEAR:
            LOGGER.warning(
                "Unexpected number of records after processing: expected %d, got %d.",
                HOURS_PER_YEAR,
                final.shape[0],
            )

        return final

    def _reindex(self, data: NasaDataArray) -> NasaDataArray:
        """Reindex data by MO, DY, HR triplets."""
        # Sort by date fields
        sort_idx = np.lexsort((data["HR"], data["DY"], data["MO"]))
        sorted_data = data[sort_idx]

        # Find unique dates and their positions
        sorted_dates = sorted_data[["MO", "DY", "HR"]]
        unique_dates, _, counts = np.unique(
            sorted_dates,
            return_index=True,
            return_counts=True,
        )

        # Check all counts are the same (np.nan evaluate to True)
        years: int = len(np.unique(data["YEAR"]))
        equal = counts == years
        if not np.all(equal):
            msg = (
                "Inconsistent number of records per hour. "
                f"Should match number of years {years}. "
                f"Inconsistent for dates: {unique_dates[~equal]}."
            )
            raise ValueError(msg)

        # Prepare output array
        dates: int = len(unique_dates)  # 8760 for non-leap years
        cache_dtype: np.dtype[Any] = hourly_cache_dtype(years)
        hourly_stats: NasaDataArray = np.empty(dates, dtype=cache_dtype)
        hourly_stats["HR"] = unique_dates["HR"]
        hourly_stats["DY"] = unique_dates["DY"]
        hourly_stats["MO"] = unique_dates["MO"]
        hourly_stats["RH2M"] = sorted_data["RH2M"].reshape(dates, years).round(2)
        hourly_stats["T2M"] = sorted_data["T2M"].reshape(dates, years).round(2)
        hourly_stats["ALLSKY_SFC_SW_DWN"] = (
            sorted_data["ALLSKY_SFC_SW_DWN"]
            .reshape(
                dates,
                years,
            )
            .round()
        )  # round to whole W/m^2

        return hourly_stats

    def _get_header_length(self, lines: list[str]) -> int:
        first_param, *_ = self.parameters
        for i, line in enumerate(lines):

            # Skip empty and metadata lines (start with '-' or '#')
            if not line.strip() or line.startswith(("-", "#")):
                continue

            # First non-metadata line should be column headers
            # Check if it looks like headers (contains expected columns)
            upper: str = line.upper()
            if "YEAR" in upper and first_param in upper:
                LOGGER.debug("Found column headers at line %d: %s", i, {line[:100]})
                return i

        msg = f"No column headers found in CSV after checking first {len(lines)} lines."
        LOGGER.error(msg)
        raise ValueError(msg)
