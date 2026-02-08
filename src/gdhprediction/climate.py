import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import InitVar, asdict, dataclass, field
from datetime import date
from http.client import (
    BAD_REQUEST,
    INTERNAL_SERVER_ERROR,
    TOO_MANY_REQUESTS,
    UNPROCESSABLE_ENTITY,
)
from io import StringIO
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Self

import numpy as np
import requests
from numpy import typing as npt
from requests import RequestException, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from gdhprediction.config import GdhPredictionConfig
from gdhprediction.paths import APP_PATHS, GdhPredictionPaths
from gdhprediction.utils import (
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
    HourlyDataDType,
    NasaDataArray,
    NasaDataDType,
    Parameters,
    Shift,
    TimeStandards,
    is_parameter,
    remove_leap,
)

if TYPE_CHECKING:
    from pathlib import Path


LOGGER: Final = logging.getLogger(__name__)


class NasaApiError(Exception):
    """Custom exception for NASA API errors."""


@dataclass(frozen=True, slots=True)
class RequestData(FieldsHashMxn):
    """Permanent attributes correspond exactly to NASA API parameters."""

    # human friendly input parameters
    lat: InitVar[float]
    lon: InitVar[float]
    parameter_set: InitVar[Iterable[Parameters]]
    start_year: InitVar[int]
    end_year: InitVar[int]
    # actual representation to send to NASA API
    start: str = field(init=False)
    end: str = field(init=False)
    parameters: str = field(init=False)
    latitude: float = field(init=False)
    longitude: float = field(init=False)
    community: Communities
    format: Formats
    time_standard: TimeStandards

    def __post_init__(
        self,
        lat: float,
        lon: float,
        parameter_set: Iterable[Parameters],
        start_year: int,
        end_year: int,
    ) -> None:
        """Validate and transform input parameters for NASA API."""
        start, end = self._valid_years(start_year, end_year)
        lat, lon = self._snap_coordinates(lat, lon)
        # prepare for NASA API
        object.__setattr__(self, "start", f"{start}0101")
        object.__setattr__(self, "end", f"{end}1231")
        object.__setattr__(self, "latitude", lat)
        object.__setattr__(self, "longitude", lon)
        object.__setattr__(self, "parameters", ",".join(sorted(parameter_set)))

    def _valid_years(
        self, start_year: int | None, end_year: int | None,
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


@dataclass(frozen=True, slots=True)
class CacheManager:

    data_request_hash: InitVar[str]
    config_hash: InitVar[str]
    cache_hash: str = field(init=False)
    _cache_paths: GdhPredictionPaths

    def __post_init__(self, data_request_hash: str, config_hash: str) -> None:
        # Collision-free combination
        concatenated = f"{data_request_hash}_{config_hash}"
        cache_hash = hashlib.md5(concatenated.encode(encoding="utf-8")).hexdigest()[:8]
        object.__setattr__(self, "cache_hash", cache_hash)

    def check_cache_integrity(self) -> bool:
        """Check if cached data exists for the current request parameters."""
        cache_file = self._cache_paths.annual_data(self.cache_hash)
        exists: bool = cache_file.is_file()
        if exists:
            LOGGER.info("Cache hit: found cached data at %s.", cache_file)
            checksum_file = self._cache_paths.annual_data_checksum(self.cache_hash)
            if checksum_file.is_file():
                LOGGER.debug("Found checksum file at %s.", checksum_file)
                # verify checksum
                expected_checksum = checksum_file.read_text().strip()
                actual_checksum = hashlib.sha256(cache_file.read_bytes()).hexdigest()
                if actual_checksum != expected_checksum:
                    LOGGER.warning(
                        "Checksum mismatch! File may be corrupted. "
                        "Expected: %s, Actual: %s",
                        expected_checksum,
                        actual_checksum,
                    )
                    return False
            else:
                LOGGER.warning("Missing checksum file at %s. ", checksum_file)
                return False
        else:
            LOGGER.info("Cache miss: no cached data found at %s.", cache_file)
        return exists

    def write_to_cache(self, data_array: NasaDataArray) -> None:
        """Write current data array to cache with checksum."""
        if data_array is None:
            LOGGER.warning("No data to write to cache.")
            return
        data_path: Path = self._cache_paths.annual_data(self.cache_hash)
        checksum_path: Path = self._cache_paths.annual_data_checksum(self.cache_hash)
        np.save(data_path, data_array)
        checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
        checksum_path.write_text(checksum)
        LOGGER.info(
            "Saved data and checksum to cache at %s and %s.", data_path, checksum_path,
        )

    def load_from_cache(self) -> NasaDataArray | None:
        """Load cached data into memory if it exists and is uncorrupted."""
        if self.check_cache_integrity():
            data_array = np.load(self._cache_paths.annual_data(self.cache_hash))
            LOGGER.info("Successfully loaded data from cache.")
            return data_array
        LOGGER.warning("Cannot load from cache: data is missing or corrupted.")
        return None


class ClimateData:

    __slots__ = (
        "_cache_manager",
        "_cache_paths",
        "_config",
        "_request_data",
        "data_array",
    )

    # default parameters for NASA API request
    COMMUNITY: ClassVar[Communities] = "ag"
    FORMAT: ClassVar[Formats] = "csv"
    TIME_STANDARD: ClassVar[TimeStandards] = "lst"
    PARAMETERS: ClassVar[tuple[Parameters, ...]] = ("T2M", "RH2M")
    MISSING_DATA: Final[Literal[-999]] = -999
    EXPECTED_COLS: ClassVar[frozenset[str]] = frozenset(
        {"YEAR", "MO", "DY", "HR"} | set(PARAMETERS),
    )

    # instance attributes
    data_array: NasaDataArray | None
    _config: GdhPredictionConfig
    _cache_manager: CacheManager
    _request_data: RequestData
    _start_year: int
    _end_year: int
    _parameter_set: set[str]
    _cache_paths: GdhPredictionPaths

    def __init__(
        self,
        request_data: RequestData,
        config: GdhPredictionConfig,
        cache_paths: GdhPredictionPaths = APP_PATHS,
    ) -> None:
        self._request_data = request_data
        self._config = config
        self._cache_manager = CacheManager(
            request_data.compute_hash(),
            config.compute_hash(),
            cache_paths,
        )
        # compute cache hash based on request parameters
        self._cache_paths = cache_paths
        self.data_array = None
        LOGGER.info("Created climate data management object %s.", self)

    @classmethod
    def from_params(
        cls,
        latitude: float,
        longitude: float,
        config: GdhPredictionConfig,
        *,
        start_year: int = FIRST_YEAR,
        end_year: int = LAST_YEAR,
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
        return cls(request_data, config)

    @property
    def request_data(self) -> RequestData:
        return self._request_data

    @property
    def latitude(self) -> float:
        return self._request_data.latitude

    @property
    def longitude(self) -> float:
        return self._request_data.longitude

    def cache_hash(self) -> str:
        return self._cache_manager.cache_hash

    def __repr__(self) -> str:
        return (
            f"ClimateData(request_data='{self._request_data!r}', "
            f"cache_hash='{self.cache_hash}', "
            f"shape={self.data_array.shape if self.data_array is not None else None})"
        )


@dataclass(frozen=True, slots=True)
class ClimateDataRequest:

    # api call parameters
    _DEFAULT_API_VERSION: ClassVar[str] = "v2.5.9"
    _TIMEOUT: ClassVar[tuple[float, float]] = (5.0, 15.0)
    _RETRY: ClassVar[Retry] = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    request_data: RequestData
    gdh_base: float
    temperature_transform: Callable[[NasaDataArray], npt.NDArray[np.float32]]
    api_version: str = field(default=_DEFAULT_API_VERSION, repr=False, kw_only=True)
    _session: Session = field(init=False, repr=False)
    # convenience fields for validation and processing
    parameters: set[Parameters] = field(init=False, repr=False)
    _start_year: int = field(init=False, repr=False)
    _end_year: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        session = self._create_session()
        object.__setattr__(self, "_session", session)
        # extract validated parameters
        _start_year = int(self.request_data.start[:4])
        _end_year = int(self.request_data.end[:4])
        _parameter_set = set(self.request_data.parameters.split(","))
        object.__setattr__(self, "_start_year", _start_year)
        object.__setattr__(self, "_end_year", _end_year)
        object.__setattr__(self, "parameters", _parameter_set)

    def _create_session(self) -> Session:
        """Initialize a requests Session with retry logic."""
        session = Session()
        adapter = HTTPAdapter(max_retries=self._RETRY)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update(
            {
                "User-Agent": self._get_user_agent(self.api_version),
                "Accept": "text/csv",
            },
        )
        return session

    def _get_user_agent(self, api_version: str) -> str:
        return f"GdhPrediction/1.0 (NASA-POWER-Client/{api_version})"

    def __del__(self) -> None:
        """Close session on cleanup."""
        self._session.close()

    def request_nasa_data(self, api_version: str | None = None) -> NasaDataArray:
        params: dict[str, str | float] = asdict(self.request_data)
        if api_version:
            self._session.headers.update(
                {"User-Agent": self._get_user_agent(api_version)},
            )
        LOGGER.info("Requesting NASA data: %s.", self.request_data)
        LOGGER.debug("NASA API version: %s", api_version or self._DEFAULT_API_VERSION)

        # perform the api call
        try:
            response = self._session.get(
                NASA_HOURLY_API, params=params, timeout=self._TIMEOUT,
            )
        # timeout, connection, or other request-related error
        except RequestException as rerr:
            LOGGER.exception("Connection failed.")
            raise NasaApiError("Failed to connect to NASA API.") from rerr

        LOGGER.debug("NASA API response status: %d", response.status_code)

        # check for HTTP errors
        try:
            response.raise_for_status()  # Raise for 4xx/5xx errors
        except requests.HTTPError as http_err:
            status = response.status_code
            LOGGER.exception("HTTP error %d.", status)

            detail: str
            if status == BAD_REQUEST:
                detail = "Bad request. Check parameters."
            elif status == TOO_MANY_REQUESTS:
                detail = "Rate limit exceeded. Try again later."
            elif status == UNPROCESSABLE_ENTITY:
                detail = "Unprocessable entity. Check parameters."
            elif status >= INTERNAL_SERVER_ERROR:
                detail = "Server error. Try again later."
            else:
                detail = ""

            raise NasaApiError(
                f"NASA API returned HTTP {status}. {detail}",
            ) from http_err

        LOGGER.info("Received data from the server.")
        LOGGER.debug("Received data with encoding: %s", response.encoding)

        csv_text: str = response.text

        # validate response content
        if not csv_text:
            LOGGER.error("Received empty response from NASA API.")
            raise NasaApiError("NASA API returned empty data.")

        line_count: int = csv_text.count("\n")
        exp_line_count: int = (self._end_year - self._start_year + 1) * HOURS_PER_YEAR

        if line_count < exp_line_count:
            LOGGER.error(
                "Received incomplete data: expected at least %d lines, got %d.",
                exp_line_count,
                line_count,
            )
            raise NasaApiError(
                f"NASA API returned incomplete data: "
                f"expected at least {exp_line_count} lines, got {line_count}.",
            )

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
            LOGGER.exception("Failed to parse CSV data returned by NASA.")
            raise ValueError("CSV parsing failed.") from e

        LOGGER.info("Parsed %d records with columns: %s", len(data), data.dtype.names)

        # Validate we got expected parameter columns
        actual: set[Parameters] = set(filter(is_parameter, (data.dtype.names or ())))

        if not self.parameters.issubset(actual):
            missing: set[Parameters] = self.parameters - actual
            msg = f"Missing expected columns: {missing}"
            raise ValueError(msg)

        # process the actual content

        # remove leap days; reuse the same reference
        data = remove_leap(data)
        # replace -999 with np.nan; no copy
        for param in self.parameters:
            data[param][data[param] == MISSING_DATA] = np.nan

        # calculate annual means for each parameter
        final = self._calculate_hourly_stats(data, gdh_base=4.5, transform=Shift(2.0))

        # validate final shape is 8760
        if final.shape[0] != HOURS_PER_YEAR:
            LOGGER.warning(
                "Unexpected number of records after processing: expected %d, got %d.",
                HOURS_PER_YEAR,
                final.shape[0],
            )

        return final

    def _calculate_hourly_stats(self, data: NasaDataArray) -> NasaDataArray:
        # Sort by date fields
        sort_idx = np.lexsort((data["HR"], data["DY"], data["MO"]))
        sorted_data = data[sort_idx]

        # Find unique dates and their positions
        sorted_dates = sorted_data[["MO", "DY", "HR"]]
        unique_dates, unique_idx, counts = np.unique(
            sorted_dates, return_index=True, return_counts=True,
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
        dates: int = len(unique_dates)
        hourly_stats: NasaDataArray = np.empty(dates, dtype=HourlyDataDType)
        hourly_stats["MO"] = unique_dates["MO"]
        hourly_stats["DY"] = unique_dates["DY"]
        hourly_stats["HR"] = unique_dates["HR"]

        inner_t2m = self.temperature_transform(sorted_data)
        gdh_2d = np.maximum(inner_t2m - self.gdh_base, 0).reshape(dates, years)

        # generate GDH percentiles
        for p in PERCENTILES:
            k = f"GDH_p{p:02d}"
            hourly_stats[k] = np.percentile(gdh_2d, p, axis=1).round(2)

        # Use reduceat to sum within each group
        t2m_sums = np.add.reduceat(sorted_data["T2M"], unique_idx, dtype=np.float64)
        rh2m_sums = np.add.reduceat(sorted_data["RH2M"], unique_idx, dtype=np.float64)

        hourly_stats["T2M"] = (t2m_sums / years).round(2)
        hourly_stats["RH2M"] = (rh2m_sums / years).round(2)
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

        LOGGER.error("No column headers in CSV after checking %d lines.", len(lines))
        raise ValueError("No column headers in CSV")
