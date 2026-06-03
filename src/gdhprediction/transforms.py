from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, Final

import numpy as np
from numpy import typing as npt

from gdhprediction.utils import NasaDataArray, NasaParameters, is_parameter


class TemperatureTransform(ABC):

    __slots__ = ()
    required_params: ClassVar[set[NasaParameters]] = set[NasaParameters]()

    @abstractmethod
    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]: ...

    def validate(self, data: NasaDataArray) -> None:
        data_params = set(filter(is_parameter, data.dtype.names or ()))
        missing_params = self.required_params - data_params
        if missing_params:
            msg = (
                f"Missing required parameters: {missing_params}. "
                f"Parameters required by {type(self)}: {data_params}."
            )
            raise ValueError(msg)


class Identity(TemperatureTransform):

    __slots__ = ()

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        return data["T2M"].astype(np.float32)


IDENTITY: Final = Identity()


@dataclass(frozen=True, slots=True)
class Shift(TemperatureTransform):

    required_params: ClassVar[set[NasaParameters]] = {"T2M"}
    degree_shift: float

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        return data["T2M"].astype(np.float32) + self.degree_shift


@dataclass(frozen=True, slots=True)
class RadiationShift(Shift):

    required_params: ClassVar[set[NasaParameters]] = {"T2M", "ALLSKY_SFC_SW_DWN"}
    solar_factor: float = 0.01

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        t2m = data["T2M"].astype(np.float32)
        solar = data["ALLSKY_SFC_SW_DWN"].astype(np.float32)
        return self.degree_shift + self.solar_factor * solar + t2m
