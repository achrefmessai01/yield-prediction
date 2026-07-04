"""Shared dtypes, constants, GPS/timezone, FAO-56 radiation, and field hashing."""

import hashlib
import math
from abc import ABC
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from datetime import date, datetime, time
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    NewType,
    Protocol,
    assert_never,
    get_args,
)
from zoneinfo import ZoneInfo

import numpy as np
from numpy import typing as npt
from timezonefinder import TimezoneFinderL  # type: ignore [import-untyped]
from typing_extensions import TypeIs

TzDatetime = NewType("TzDatetime", datetime)  # datetime with tzinfo

Communities = Literal["ag", "sb", "re"]
Formats = Literal["csv", "json", "ascii", "netcdf", "icasa", "xarray"]
TimeStandards = Literal["lst", "utc"]
NasaParameters = Literal[
    "RH2M",  # relative humidity [0-100]
    "T2M",  # degree Celsius
    "ALLSKY_SFC_SW_DWN",  # W/m^2 (on hourly basis)
]
BrightSkyFields = Literal[
    "timestamp",  # ISO 8601 format
    "temperature",  # degree Celsius
    "relative_humidity",  # %
    "sunshine",  # min / hour
    "dew_point",  # degree Celsius
]
BRIGHTSKY_FIELDS: Final[frozenset[BrightSkyFields]] = frozenset(
    get_args(BrightSkyFields),
)
PARAMETERS: Final[frozenset[NasaParameters]] = frozenset(get_args(NasaParameters))
DateTimeUnits = Literal["YEAR", "MO", "DY", "HR"]
DATETIME_UNITS: Final[frozenset[DateTimeUnits]] = frozenset(get_args(DateTimeUnits))
CacheFormat = Literal["raw", "processed"]
MISSING_DATA: Final[Literal[-999]] = -999
FIRST_YEAR: Final = 2001  # must by later than 2001
LAST_YEAR: Final = 2025
CLIMATE_YEARS: Final = 20  # default number of years to use for climate data
HOURS_PER_YEAR: Final = 8760
HOURS_PER_DAY = 24
FEB: Final = 2
LEAP_DAY: Final = 29
LAT_STEP: Final = 0.5
LON_STEP: Final = 0.625
LAT_MAX: Final = 90.0
LON_MAX: Final = 180.0
NASA_HOURLY_API: Final = "https://power.larc.nasa.gov/api/temporal/hourly/point"
BRIGHTSKY_API: Final = "https://api.brightsky.dev/weather"
# this is the type returned by the nasa api
NasaDataDType = np.dtype(
    [
        ("YEAR", "<i8"),
        ("MO", "<i8"),
        ("DY", "<i8"),
        ("HR", "<i8"),
        ("RH2M", "<f8"),
        ("T2M", "<f8"),
        ("ALLSKY_SFC_SW_DWN", "<f8"),
    ],
)
# this is the type written to cache after processing
HourlyAnnualizedGdhDType = np.dtype(
    [
        ("MO", "<u1"),  # uint8 is sufficient for month (1-12)
        ("DY", "<u1"),  # uint8 is sufficient for day (1-31)
        ("HR", "<u1"),  # uint8 is sufficient for hour (0-23)
        ("CUMGDH_p05", "<f4"),  # float32 for percentiles
        ("CUMGDH_p25", "<f4"),
        ("CUMGDH_p50", "<f4"),
        ("CUMGDH_p75", "<f4"),
        ("CUMGDH_p95", "<f4"),
    ],
)
# this is the type returned by the BrightSky API call, preprocessed
BrightSkyDType = np.dtype(
    [
        ("MO", "<u1"),  # uint8 is sufficient for month (1-12)
        ("DY", "<u1"),  # uint8 is sufficient for day (1-31)
        ("HR", "<u1"),  # uint8 is sufficient for hour (0-23)
        ("T2M", "<f4"),  # float32 for temperature
        ("RH2M", "<f4"),  # float32 for relative humidity
        ("ALLSKY_SFC_SW_DWN", "<f4"),  # float32 for sunshine duration
    ],
)
# this is the type written to cache after processing
HourlyForecastGhdDType = np.dtype(
    [
        ("MO", "<u1"),  # uint8 is sufficient for month (1-12)
        ("DY", "<u1"),  # uint8 is sufficient for day (1-31)
        ("HR", "<u1"),  # uint8 is sufficient for hour (0-23)
        ("GDH", "<f4"),  # float32 for GDH forecast
    ],
)

PERCENTILES = (5, 25, 50, 75, 95)


# this is the type written to cache before processing
def hourly_cache_dtype(num_years: int) -> np.dtype[Any]:
    return np.dtype(
        [
            ("MO", "<u1"),  # uint8 is sufficient for month (1-12)
            ("DY", "<u1"),  # uint8 is sufficient for day (1-31)
            ("HR", "<u1"),  # uint8 is sufficient for hour (0-23)
            ("RH2M", "<f4", num_years),  # float32 for means to save memory
            ("T2M", "<f4", num_years),  # float32
            ("ALLSKY_SFC_SW_DWN", "<f4", num_years),  # float32
        ],
    )


type StructuredDataArray = npt.NDArray[np.void]
"""Structured (named-field) array at the cache + abstract pipeline layer;
supertype of NasaDataArray and BrightSkyDataArray."""
type NasaDataArray = npt.NDArray[np.void]
"""Structured array carrying NASA POWER's hourly fields (T2M, ALLSKY_SFC_SW_DWN)."""
type BrightSkyDataArray = npt.NDArray[np.void]
"""Structured array carrying BrightSky/DWD's hourly forecast fields."""
type Float64 = npt.NDArray[np.float64]
"""Plain float64 array (raw temperature/irradiance series)."""
type Float32 = npt.NDArray[np.float32]
"""Plain float32 array (processed GDH output series)."""
type UInt8 = npt.NDArray[np.uint8]
"""Plain uint8 array (up to 255, e.g. month/day/hour)."""
type Int16 = npt.NDArray[np.int16]
"""Plain int16 array (up to 32767, e.g. day-of-year)."""


# ABC used as a mixin base: subclasses supply dataclass fields, not abstract methods.
class FieldsHashMxn(ABC):  # noqa: B024
    """Mixin giving a frozen dataclass a stable 8-char hash over its fields."""

    __slots__ = ()

    hash_version: ClassVar[int] = 1  # increment this if the hash_fields logic changes

    @property
    def hash_fields(self) -> Iterable[str]:
        if is_dataclass(self):
            for f in sorted(fields(self), key=lambda fld: fld.name):
                val = getattr(self, f.name)
                if isinstance(val, float):
                    # round floats to 5 decimal places for hashing
                    yield f"{val:.5f}"
                else:
                    yield str(val)
        else:
            msg = f"Default {type(self)} can only be mixed with dataclasses."
            raise TypeError(msg)

    def compute_hash(self) -> str:
        key = "_".join(self.hash_fields) + f"_v{self.hash_version}"
        digest = hashlib.md5(key.encode(encoding="utf-8"), usedforsecurity=False)
        return digest.hexdigest()[:8]


def is_parameter(value: str) -> TypeIs[NasaParameters]:
    return value in PARAMETERS


def is_datetime_unit(value: str) -> TypeIs[DateTimeUnits]:
    return value in DATETIME_UNITS


def remove_leap(data_array: NasaDataArray) -> NasaDataArray:
    # Remove rows where MO=2 and DY=29 (leap day)
    return data_array[~((data_array["MO"] == FEB) & (data_array["DY"] == LEAP_DAY))]


class DataclassInstance(Protocol):
    """Structural type matching any dataclass instance."""

    __dataclass_fields__: ClassVar[dict[str, Any]]


class FieldsHashCoor(DataclassInstance, Protocol):
    """Read-only dataclass with fields that can be hashed for caching purposes."""

    @property
    def hash_fields(self) -> Iterable[str]: ...

    def compute_hash(self) -> str: ...

    @property
    def latitude(self) -> float: ...

    @property
    def longitude(self) -> float: ...


def to_month_day_hour(timestamp: datetime) -> tuple[int, int, int]:
    return timestamp.month, timestamp.day, timestamp.hour


def sunshine_to_radiation(  # noqa: PLR0913
    sunshine_min: Float32,  # DWD sunshine duration (min/h)
    lat_deg: float,
    day_of_year: Int16,
    hour: UInt8,
    a_s: float = 0.25,
    b_s: float = 0.50,
) -> Float32:
    """Estimate surface shortwave irradiance (W/m²) from sunshine duration.

    Vectorised: operates elementwise over the input arrays and returns float32.
    Uses the Angstrom-Prescott equation (FAO-56 parameterisation).

    :param sunshine_min: Sunshine duration in minutes per hour (0-60).
    :param lat_deg: Latitude in degrees.
    :param day_of_year: Day of the year (1-365), one entry per hour.
    :param hour: Hour of the day (0-23), one entry per hour.
    :param a_s: Angström-Prescott coefficient for clear-sky radiation (default 0.25).
    :param b_s: Angström-Prescott coefficient for sunshine duration (default 0.50).
    :return: Surface shortwave irradiance (W/m²) as a float32 array.
    """
    # Cast to avoid underflow-wrap on `hour - 12` / `day_of_year - 81`.
    doy = np.asarray(day_of_year, dtype=np.float64)
    hr = np.asarray(hour, dtype=np.float64)

    lat = math.radians(lat_deg)
    b = 2 * np.pi * (doy - 81) / 364
    declination = np.radians(23.45 * np.sin(b))

    # Hour angle (solar noon = 0)
    hour_angle = np.radians(15 * (hr - 12))

    # Extraterrestrial radiation (W/m²)
    cos_zenith = (
        math.sin(lat) * np.sin(declination)
        + math.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    )
    radiation = np.maximum(0.0, 1361.0 * cos_zenith)  # solar constant * cos(zenith)

    # Fractional sunshine (0-1) from minutes in this hour
    n_fraction = sunshine_min / 60.0

    return (radiation * (a_s + b_s * n_fraction)).astype(np.float32)


def rh_from(temperature: float, dew_pt: float) -> float:
    """Compute RH (%) from dry-bulb temperature and dew-point (both °C)."""
    es = 6.112 * math.exp((17.67 * temperature) / (temperature + 243.5))
    ea = 6.112 * math.exp((17.67 * dew_pt) / (dew_pt + 243.5))
    return 100.0 * ea / es


def round_sig(value: float, sig_figs: int) -> float:
    """Round ``value`` to ``sig_figs`` significant figures.

    Scale-invariant meaningful for parameters at very different magnitudes.
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, sig_figs - 1 - math.floor(math.log10(abs(value))))


def ordered(rng: tuple[float, float]) -> tuple[float, float]:
    """Return ``(lo, hi)`` unchanged; raise if it is not strictly increasing."""
    lo, hi = rng
    if lo >= hi:
        msg = f"range min ({lo}) must be < max ({hi})"
        raise ValueError(msg)
    return rng


def get_gps_tzinfo(latitude: float, longitude: float) -> ZoneInfo:
    """Get the timezone for the given GPS coordinates."""
    tz = TimezoneFinderL().timezone_at(lng=longitude, lat=latitude)
    if tz is None:
        msg = f"Could not determine timezone for coordinates: {latitude}, {longitude}"
        raise ValueError(msg)
    return ZoneInfo(tz)


def validate_gps(lat: float, lon: float) -> None:
    if not -LAT_MAX <= lat <= LAT_MAX:
        msg = f"Latitude out of range: {lat}"
        raise ValueError(msg)
    if not -LON_MAX <= lon <= LON_MAX:
        msg = f"Longitude out of range: {lon}"
        raise ValueError(msg)


def coerce_tz_datetime(dt: date | datetime | None, tzinfo: ZoneInfo) -> TzDatetime:
    if dt is None:
        return TzDatetime(datetime.now(tzinfo))
    if isinstance(dt, datetime):  # datetime <: date
        tzdt = dt.replace(tzinfo=tzinfo) if dt.tzinfo is None else dt.astimezone(tzinfo)
        return TzDatetime(tzdt)
    if isinstance(dt, date):
        return TzDatetime(datetime.combine(dt, time.min, tzinfo=tzinfo))
    assert_never(dt)


def or_else[T](value: T | None, default: T) -> T:
    """Return ``value`` if it is not None, else return ``default``."""
    return value if value is not None else default
