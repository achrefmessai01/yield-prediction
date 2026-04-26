import hashlib
from collections.abc import Iterable
from dataclasses import fields, is_dataclass
from typing import Any, Final, Literal, get_args

import numpy as np
from numpy import typing as npt
from typing_extensions import TypeIs

Communities = Literal["ag", "sb", "re"]
Formats = Literal["csv", "json", "ascii", "netcdf", "icasa", "xarray"]
TimeStandards = Literal["lst", "utc"]
Parameters = Literal[
    "RH2M",  # relative humidity [0-100]
    "T2M",  # degree Celsius
    "ALLSKY_SFC_SW_DWN",  # W/m^2 (on hourly basis)
]
PARAMETERS: Final[frozenset[Parameters]] = frozenset(get_args(Parameters))
DateTimeUnits = Literal["YEAR", "MO", "DY", "HR"]
DATETIME_UNITS: Final[frozenset[DateTimeUnits]] = frozenset(get_args(DateTimeUnits))
Transforms = Literal["shift", "radiation_shift", "identity"]
CacheFormat = Literal["raw", "processed"]
MISSING_DATA: Final[Literal[-999]] = -999
FIRST_YEAR: Final = 2001  # must by later than 2001
LAST_YEAR: Final = 2025
HOURS_PER_YEAR: Final = 8760
FEB: Final = 2
LEAP_DAY: Final = 29
LAT_STEP: Final = 0.5
LON_STEP: Final = 0.625
NASA_HOURLY_API: Final = "https://power.larc.nasa.gov/api/temporal/hourly/point"
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
    ]
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
        ]
    )


type NasaDataArray = npt.NDArray[np.void]


class FieldsHashMxn:

    @property
    def hash_fields(self) -> Iterable[str]:
        if not is_dataclass(self):
            raise TypeError(f"Default {type(self)} can only be mixed with dataclasses.")
        return (str(getattr(self, f.name)) for f in fields(self))

    def compute_hash(self) -> str:
        key = "_".join(self.hash_fields)
        return hashlib.md5(key.encode(encoding="utf-8")).hexdigest()[:8]


def is_parameter(value: str) -> TypeIs[Parameters]:
    return value in PARAMETERS


def is_datetime_unit(value: str) -> TypeIs[DateTimeUnits]:
    return value in DATETIME_UNITS


def remove_leap(data_array: NasaDataArray) -> NasaDataArray:
    # Remove rows where MO=2 and DY=29 (leap day)
    return data_array[~((data_array["MO"] == FEB) & (data_array["DY"] == LEAP_DAY))]
