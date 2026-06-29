"""HTTP request base, the two-tier .npy cache manager, and the GdhData fetch ABC."""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import suppress
from dataclasses import InitVar, asdict, dataclass, field
from http.client import (
    BAD_REQUEST,
    INTERNAL_SERVER_ERROR,
    TOO_MANY_REQUESTS,
    UNPROCESSABLE_ENTITY,
)
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, assert_never

import numpy as np
import requests
from numpy import typing as npt
from requests import RequestException, Response, Session
from requests.adapters import HTTPAdapter
from urllib3 import Retry

from gdhprediction.config import APP_PATHS, GdhPredictionConfig, GdhPredictionPaths
from gdhprediction.utils import (
    CacheFormat,
    DataclassInstance,
    FieldsHashCoor,
    StructuredDataArray,
)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


class RequestApiError(Exception):
    """Custom exception for API request errors."""


@dataclass(frozen=True, slots=True)
class DataRequest(ABC):
    """Abstract HTTP data request backed by a retrying session."""

    # API configuration
    _API_URL: ClassVar[str]
    _API_NAME: ClassVar[str]
    _DEFAULT_API_VERSION: ClassVar[str]

    # api call parameters
    _TIMEOUT: ClassVar[tuple[float, float]] = (5.0, 15.0)
    _RETRY: ClassVar[Retry] = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    request_data: DataclassInstance

    # session parameters
    api_version: InitVar[str | None] = field(default=None, repr=False, kw_only=True)
    _session: Session = field(init=False, repr=False)
    _api_version: str = field(init=False, repr=False)

    def __post_init__(self, api_version: str | None) -> None:
        # set API version
        version = api_version or self._DEFAULT_API_VERSION
        object.__setattr__(self, "_api_version", version)
        # create active session
        session = self._create_session()
        object.__setattr__(self, "_session", session)

    @abstractmethod
    def _get_user_agent(self, api_version: str) -> str: ...

    def _create_session(self) -> Session:
        """Initialize a requests Session with retry logic."""
        session = Session()
        adapter = HTTPAdapter(max_retries=self._RETRY)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        session.headers.update(
            {
                "User-Agent": self._get_user_agent(self._api_version),
                "Accept": "text/csv",
            },
        )
        return session

    def __del__(self) -> None:
        """Close session on cleanup."""
        with suppress(AttributeError):
            self._session.close()

    def _perform_request(self, api_version: str | None = None) -> Response:
        params: dict[str, str | float] = asdict(self.request_data)
        if api_version:
            self._session.headers.update(
                {"User-Agent": self._get_user_agent(api_version)},
            )
        LOGGER.info("Requesting %s data: %s.", self._API_NAME, self.request_data)
        LOGGER.debug(
            "%s API version: %s",
            self._API_NAME,
            api_version or self._DEFAULT_API_VERSION,
        )

        # perform the api call
        try:
            response = self._session.get(
                self._API_URL,
                params=params,
                timeout=self._TIMEOUT,
            )
        # timeout, connection, or other request-related error
        except RequestException as rerr:
            msg = f"Connection error while requesting {self._API_NAME} data."
            LOGGER.exception(msg)
            raise RequestApiError(msg) from rerr

        LOGGER.debug("%s API response status: %d", self._API_NAME, response.status_code)

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

            msg = f"{self._API_NAME} API returned HTTP {status}. {detail}"
            raise RequestApiError(msg) from http_err

        LOGGER.info("Received data from the server.")
        LOGGER.debug("Received data with encoding: %s", response.encoding)

        return response


CacheScope = Literal["data", "cache", "all"]


@dataclass(frozen=True, slots=True)
class CacheManager:
    """Two-tier (raw/processed) .npy cache with sha256 integrity checks."""

    raw_data_hash: str
    config_hash: InitVar[str]
    processed_hash: str = field(init=False)
    _cache_paths: GdhPredictionPaths
    persistent: bool = field(default=True)

    def __post_init__(self, config_hash: str) -> None:
        # Collision-free combination
        data_request_hash = self.raw_data_hash
        concatenated = f"{data_request_hash}_{config_hash}"
        combined = hashlib.md5(
            concatenated.encode(encoding="utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:8]
        object.__setattr__(self, "processed_hash", combined)

    def _hash(self, data_format: CacheFormat) -> str:
        if data_format == "processed":
            return self.processed_hash
        if data_format == "raw":
            return self.raw_data_hash
        assert_never(data_format)

    def _file(self, hashkey: str) -> Path:
        if self.persistent:
            return self._cache_paths.data_file(hashkey)
        return self._cache_paths.cache_file(hashkey)

    def _checksum(self, hashkey: str) -> Path:
        if self.persistent:
            return self._cache_paths.data_file_checksum(hashkey)
        return self._cache_paths.cache_file_checksum(hashkey)

    def _check_cache(self, hashkey: str) -> bool:
        """Check if cached data exists for the current request parameters."""
        cache_file = self._file(hashkey)
        exists: bool = cache_file.is_file()
        if exists:
            LOGGER.info("Cache hit: found cached data at %s.", cache_file)
            checksum_file = self._checksum(hashkey)
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

    def write_to_cache(
        self,
        data_array: StructuredDataArray,
        cache_fmt: CacheFormat,
    ) -> None:
        """Write current data array to cache with checksum."""
        if data_array is None:
            LOGGER.warning("No data to write to cache.")
            return
        hashkey = self._hash(cache_fmt)
        data_path: Path = self._file(hashkey)
        checksum_path: Path = self._checksum(hashkey)
        np.save(data_path, data_array)
        checksum = hashlib.sha256(data_path.read_bytes()).hexdigest()
        checksum_path.write_text(checksum)
        LOGGER.info(
            "Saved %s data and checksum to cache at %s and %s.",
            cache_fmt,
            data_path,
            checksum_path,
        )

    def load_from_cache(self, data_fmt: CacheFormat) -> StructuredDataArray | None:
        """Load cached data into memory if it exists and is uncorrupted."""
        hashkey = self._hash(data_fmt)
        if self._check_cache(hashkey):
            data_array = np.load(self._file(hashkey))
            LOGGER.info("Successfully loaded %s data from cache.", data_fmt)
            return data_array
        LOGGER.warning(
            "Cannot load from cache: %s data is missing or corrupted.",
            data_fmt,
        )
        return None

    @classmethod
    def invalidate(
        cls,
        cache_paths: GdhPredictionPaths = APP_PATHS,
        scope: CacheScope = "all",
    ) -> None:
        """Remove cached files."""
        targets: list[Path] = []
        if scope in ("cache", "all"):
            targets.append(cache_paths.cache)
        if scope in ("data", "all"):
            targets.append(cache_paths.data)
        for directory in targets:
            # checksums first so a partial failure leaves the cache invalid
            for pattern in ("*.sha256", "*.npy"):
                for file in directory.glob(pattern):
                    try:
                        file.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.exception("Failed to remove %s", file)
            LOGGER.info("Invalidated cache directory %s.", directory)


class GdhData[FHD: FieldsHashCoor](ABC):
    """Abstract fetch/cache/process pipeline producing GDH arrays for a request type."""

    __slots__ = (
        "_cache_manager",
        "_cache_paths",
        "_config",
        "_request_data",
        "data_array",
        "transform",
    )

    PARAMETERS: ClassVar[tuple[str, ...]]  # allows covariance in subclasses

    # instance attributes
    data_array: StructuredDataArray | None
    transform: Callable[[StructuredDataArray], npt.NDArray[np.float32]]
    _config: GdhPredictionConfig
    _cache_manager: CacheManager
    _request_data: FHD
    _cache_paths: GdhPredictionPaths

    def __init__(
        self,
        request_data: FHD,
        config: GdhPredictionConfig,
        *,
        cache_paths: GdhPredictionPaths = APP_PATHS,
        cache_persistent: bool = True,
    ) -> None:
        self._request_data = request_data
        self._config = config
        self.transform = self._config.temperature_transform
        # compute cache hash based on request parameters
        self._cache_manager = CacheManager(
            request_data.compute_hash(),
            config.compute_hash(),
            cache_paths,
            cache_persistent,
        )
        self._cache_paths = cache_paths
        self.data_array = None
        LOGGER.info("Created climate data management object %s.", self)

    @classmethod
    @abstractmethod
    def from_params(
        cls,
        latitude: float,
        longitude: float,
        config: GdhPredictionConfig,
        **kwargs: Any,
    ) -> Self: ...

    def get_gdh_data(self) -> StructuredDataArray:
        """Get climate data, using cache if available and valid."""
        if self.data_array is not None:
            LOGGER.info("Using in-memory data array.")
            return self.data_array

        # Try to load processed data from cache
        cached = self._cache_manager.load_from_cache("processed")
        if cached is not None:
            self.data_array = cached
            return self.data_array

        # Try to load raw data from cache and process it
        raw_cached = self._cache_manager.load_from_cache("raw")
        if raw_cached is not None:
            processed = self._process(raw_cached)
            self.data_array = processed
            self._cache_manager.write_to_cache(processed, "processed")
            return self.data_array

        # If no cache, request from API
        return self._refresh_caches()

    @abstractmethod
    def _process(self, raw_data: StructuredDataArray) -> StructuredDataArray: ...

    @abstractmethod
    def _request_raw_data(self) -> StructuredDataArray: ...

    def _refresh_caches(self) -> StructuredDataArray:
        """Recreate raw and processed caches by requesting and processing new data."""
        raw_data = self._request_raw_data()
        self._cache_manager.write_to_cache(raw_data, "raw")
        processed = self._process(raw_data)
        self.data_array = processed
        self._cache_manager.write_to_cache(processed, "processed")
        return self.data_array

    @abstractmethod
    def forecast_gdh(self, /, **kwargs: Any) -> StructuredDataArray:
        """Compute GDH forecast."""

    @property
    def request_data(self) -> FHD:
        return self._request_data

    @property
    def latitude(self) -> float:
        return self._request_data.latitude

    @property
    def longitude(self) -> float:
        return self._request_data.longitude

    @property
    def cache_hash(self) -> str:
        return self._cache_manager.processed_hash

    def __repr__(self) -> str:
        return (
            f"GdhData(request_data='{self._request_data!r}', "
            f"cache_hash='{self.cache_hash}', "
            f"shape={self.data_array.shape if self.data_array is not None else None})"
        )
