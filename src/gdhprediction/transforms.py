"""Temperature transforms mapping outdoor weather to an indoor temperature for GDH."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, Final, Self

import numpy as np
from numpy import typing as npt

from gdhprediction.utils import (
    NasaDataArray,
    NasaParameters,
    is_parameter,
    round_sig,
)

DENSITY = 1.2  # kg/m^3
SPEC_HEAT_CAPACITY = 1006  # J/(kg*K)
VOL_HEAT_CAPACITY = DENSITY * SPEC_HEAT_CAPACITY  # J/(m^3*K)

# A transform field value / fingerprint element: a scalar or nested tuple of
# scalars. Constrained to what ``json.dumps`` accepts, since fingerprints are
# JSON-serialised in ``GdhPredictionConfig.compute_hash``.
type Fingerprintable = float | int | bool | str | tuple["Fingerprintable", ...]


# every transform is expected to be a frozen dataclass
@dataclass(frozen=True, slots=True)
class TemperatureTransform(ABC):
    """Base for transforms that compute an indoor temperature from outdoor weather data.

    :cvar required_params: Parameters that must be present in the input data for this
        transform to work.
    :cvar _SIG_FIGS: Number of significant figures to retain when quantizing float
        fields. This ensures that transforms that differ only by float noise resolve
        identically.
    """

    required_params: ClassVar[set[NasaParameters]] = {"T2M"}
    _SIG_FIGS: ClassVar[int] = 5

    @abstractmethod
    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]: ...

    def __post_init__(self) -> None:
        # Quantize every field so transforms that differ only by float noise
        # resolve identically.
        for f in fields(self):
            object.__setattr__(self, f.name, self._quantize(getattr(self, f.name)))

    def _quantize(self, value: Fingerprintable) -> Fingerprintable:
        """Round floats (and floats nested in tuples) to ``_SIG_FIGS``."""
        match value:
            case bool():  # bool <: int — leave as-is
                return value
            case float():
                return round_sig(value, self._SIG_FIGS)
            case tuple():
                return tuple(self._quantize(v) for v in value)
            case _:
                return value

    def fingerprint(self) -> tuple[Fingerprintable, ...]:
        """Values that determine this transform's output, for cache keying.

        Two transforms with equal fingerprints produce identical processed
        arrays, so their cache may be shared regardless of how each was
        constructed (see ``GdhPredictionConfig.compute_hash``). The leading
        class name distinguishes transforms that share field names.

        A transform that derives output-affecting state outside its fields
        (or holds a field that does not affect output) must override.
        """
        return (type(self).__name__, *(getattr(self, f.name) for f in fields(self)))

    def validate(self, data: NasaDataArray) -> None:
        data_params = set(filter(is_parameter, data.dtype.names or ()))
        missing_params = self.required_params - data_params
        if missing_params:
            msg = (
                f"Missing required parameters: {missing_params}. "
                f"Parameters required by {type(self)}: {data_params}."
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Identity(TemperatureTransform):
    """Indoor temperature equals the raw outdoor T2M."""

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        return data["T2M"].astype(np.float32)


IDENTITY: Final = Identity()


@dataclass(frozen=True, slots=True)
class Shift(TemperatureTransform):
    """Indoor temperature is the outdoor T2M plus a constant offset."""

    degree_shift: float

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        return data["T2M"].astype(np.float32) + self.degree_shift


@dataclass(frozen=True, slots=True)
class RadiationShift(Shift):
    """Add a linear solar-gain term to a shifted outdoor temperature."""

    required_params: ClassVar[set[NasaParameters]] = {"T2M", "ALLSKY_SFC_SW_DWN"}
    solar_factor: float

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        t2m = data["T2M"].astype(np.float32)
        solar = data["ALLSKY_SFC_SW_DWN"].astype(np.float32)
        return self.degree_shift + self.solar_factor * solar + t2m

    @classmethod
    def from_constants(  # noqa: PLR0913
        cls,
        footprint_area: float,
        coverage_area: float,
        heat_transfer_coeff: float,
        average_height: float,
        air_infiltration: float,
        solar_coeff: float,
    ) -> Self:
        volume = footprint_area * average_height
        heat_loss = (
            heat_transfer_coeff * coverage_area
            + VOL_HEAT_CAPACITY * volume * air_infiltration / 3600
        )
        solar_factor = solar_coeff * footprint_area / heat_loss
        return cls(degree_shift=0.0, solar_factor=solar_factor)


@dataclass(frozen=True, slots=True)
class AdaptiveShift(TemperatureTransform):
    """Energy-balance indoor temperature with passive, heating, and venting regimes."""

    base_heat_loss: float
    solar_factor: float
    temperature_range: tuple[float, float]
    ventilation: float
    heating: float

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        t2m = data["T2M"].astype(np.float32)
        solar = data["ALLSKY_SFC_SW_DWN"].astype(np.float32)
        min_temp, max_temp = self.temperature_range

        # temperature with no active climate control
        passive = t2m + (self.solar_factor / self.base_heat_loss) * solar

        # heat toward min_temp, limited by capacity, never past the setpoint
        heat_capacity = self.heating / self.base_heat_loss
        heated = np.where(
            passive < min_temp,
            np.minimum(passive + heat_capacity, min_temp),
            passive,
        )

        # vent toward max_temp, but never below ambient (can't cool below outside)
        vent_loss = self.base_heat_loss + self.ventilation * VOL_HEAT_CAPACITY
        vented = t2m + (self.solar_factor / vent_loss) * solar
        indoor = np.where(passive > max_temp, np.maximum(vented, t2m), heated)
        return indoor.astype(np.float32)

    @classmethod
    def from_constants(  # noqa: PLR0913
        cls,
        footprint_area: float,
        coverage_area: float,
        heat_transfer_coeff: float,
        average_height: float,
        air_infiltration: float,
        solar_coeff: float,
        ventilation: float,
        heating: float,
        temperature_range: tuple[float, float],
    ) -> Self:
        volume = footprint_area * average_height
        base_heat_loss = (
            heat_transfer_coeff * coverage_area
            + VOL_HEAT_CAPACITY * volume * air_infiltration / 3600
        )
        solar_factor = solar_coeff * footprint_area
        return cls(
            base_heat_loss,
            solar_factor,
            temperature_range,
            ventilation,
            heating,
        )


@dataclass(frozen=True, slots=True)
class Fixed(TemperatureTransform):
    """Replace the outdoor temperature with a single constant (e.g. a setpoint)."""

    temperature: float

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        t2m = data["T2M"].astype(np.float32)
        return np.full(shape=t2m.shape, fill_value=self.temperature, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Constrained(TemperatureTransform):
    """Clip the outdoor temperature into a ``[min, max]`` range."""

    temperature_range: tuple[float, float]

    def __call__(self, data: NasaDataArray) -> npt.NDArray[np.float32]:
        self.validate(data)
        t2m = data["T2M"].astype(np.float32)
        min_temp, max_temp = self.temperature_range
        return t2m.clip(min_temp, max_temp)
