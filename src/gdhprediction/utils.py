import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from typing import ClassVar, Final, Literal, get_args

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
Transforms = Literal["shift", "radiation_shift"]
MISSING_DATA: Final[Literal[-999]] = -999
FIRST_YEAR: Final = 2001  # must by later than 2001
LAST_YEAR: Final = 2025
HOURS_PER_YEAR: Final = 8760
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
    ],
)
# this is what we process and cache
HourlyDataDType = np.dtype(
    [
        ("MO", "<u1"),  # uint8 is sufficient for month (1-12)
        ("DY", "<u1"),  # uint8 is sufficient for day (1-31)
        ("HR", "<u1"),  # uint8 is sufficient for hour (0-23)
        ("RH2M", "<f4"),  # float32 for means to save memory
        ("T2M", "<f4"),  # float32
        ("GDH_p05", "<f4"),  # float32 for percentiles
        ("GDH_p25", "<f4"),
        ("GDH_p50", "<f4"),
        ("GDH_p75", "<f4"),
        ("GDH_p95", "<f4"),
    ],
)
PERCENTILES: Final = (5, 25, 50, 75, 95)

type NasaDataArray = npt.NDArray[np.void]


class TemperatureTransform(ABC):

    required_params: ClassVar[set[Parameters]] = set[Parameters]()

    @abstractmethod
    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]: ...

    def validate(self, data: NasaDataArray) -> None:
        data_params = set(filter(is_parameter, data.dtype.names or ()))
        missing_params = self.required_params - data_params
        if missing_params:
            msg = (
                f"Missing required parameters: {missing_params}. "
                f"Data parameters: {data_params}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Shift(TemperatureTransform):

    required_params: ClassVar[set[Parameters]] = {"T2M"}
    degree_shift: float

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        return data["T2M"].astype(np.float32) + self.degree_shift


@dataclass(frozen=True, slots=True)
class RadiationShift(Shift):

    required_params: ClassVar[set[Parameters]] = {"T2M", "ALLSKY_SFC_SW_DWN"}
    solar_factor: float = 0.01

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        t2m = data["T2M"].astype(np.float32)
        solar = data["ALLSKY_SFC_SW_DWN"].astype(np.float32)
        return self.degree_shift + self.solar_factor * solar + t2m


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
    return value in get_args(Parameters)


def remove_leap(data_array: NasaDataArray) -> NasaDataArray:
    # Remove rows where MO=2 and DY=29 (leap day)
    return data_array[~((data_array["MO"] == 2) & (data_array["DY"] == 29))]
