"""Configuration models for GDH prediction.

A prediction config is a ``gdh_base`` plus a ``transform``. The transform a user
picks and the parameters it needs travel together as a discriminated (tagged)
union, :data:`TransformConfig`: ``transform.kind`` selects the transform, and for
transforms with more than one constructor a nested ``params.route`` selects the
parameterisation (see ``docs/TRANSFORMS.md`` Route A-i / A-ii). Each variant knows
how to build its own transform, so there is no central dispatch that can drift out
of sync with the union.

Fields carry their own UI metadata (``title``/``description``/``unit`` and soft
``ui_min``/``ui_max``/``step`` ranges via ``json_schema_extra``), so a form can be
generated generically. ``ge``/``gt`` are hard validation invariants that reject bad
input; the soft ranges only bound the widget.
"""

import hashlib
import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Annotated, ClassVar, Final, Literal, Self

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from gdhprediction.transforms import (
    IDENTITY,
    AdaptiveShift,
    Constrained,
    Fixed,
    RadiationShift,
    Shift,
    TemperatureTransform,
)
from gdhprediction.utils import ordered

# Reusable field types (see module docstring for the union + UI-metadata design).
# A validated (min, max) °C range, used wherever a transform takes one.
TempRange = Annotated[
    tuple[float, float],
    AfterValidator(ordered),
    Field(title="Temperature range", json_schema_extra={"unit": "°C"}),
]

# Reusable AdaptiveShift fields, shared by its coefficients and constants routes.
VentFlow = Annotated[
    float,
    Field(
        ge=0,
        title="Vent flow",
        description="Extra volumetric flow when venting above the range.",
        json_schema_extra={"unit": "m³/s", "ui_min": 0, "ui_max": 100, "step": 1},
    ),
]
HeatingPower = Annotated[
    float,
    Field(
        ge=0,
        title="Heating power",
        description="Sensible heat added when below the range.",
        json_schema_extra={"unit": "W", "ui_min": 0, "ui_max": 200000, "step": 1000},
    ),
]


class _Frozen(BaseModel):
    """Immutable model that rejects unknown keys (catches typo'd config)."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class _TransformCfg(_Frozen):
    """Base for every member of the ``TransformConfig`` union."""

    @abstractmethod
    def build(self) -> TemperatureTransform:
        """Construct the transform this config describes."""


class _Geometry(_Frozen):
    """Forward-physics inputs for a ``from_constants`` route.

    Footprint is per-house; the rest carry passive/closed-greenhouse defaults
    (``coverage_area`` defaults to ``1.4 * footprint_area``).
    """

    footprint_area: float = Field(
        gt=0,
        title="Floor area",
        description="Greenhouse footprint.",
        json_schema_extra={"unit": "m²", "ui_min": 100, "ui_max": 20000, "step": 50},
    )
    coverage_area: float = Field(
        gt=0,
        title="Cover area",
        description="Envelope area losing heat; defaults to 1.4x the footprint.",
        json_schema_extra={"unit": "m²", "ui_min": 100, "ui_max": 30000, "step": 50},
    )
    heat_transfer_coeff: float = Field(
        default=6.0,
        gt=0,
        title="U-value",
        description="Overall heat-transfer coefficient (single-skin poly tunnel ~6).",
        json_schema_extra={"unit": "W/m²K", "ui_min": 0.5, "ui_max": 10, "step": 0.5},
    )
    average_height: float = Field(
        default=3.0,
        gt=0,
        title="Mean height",
        json_schema_extra={"unit": "m", "ui_min": 1, "ui_max": 10, "step": 0.5},
    )
    air_infiltration: float = Field(
        default=1.0,
        ge=0,
        title="Air changes",
        description="Infiltration rate.",
        json_schema_extra={"unit": "1/h", "ui_min": 0, "ui_max": 10, "step": 0.5},
    )
    solar_coeff: float = Field(
        default=0.5,
        gt=0,
        title="Solar gain eta",
        description="Net solar-gain fraction (1-delta)*tau; tabulated delta ~ 0.5.",
        json_schema_extra={"ui_min": 0, "ui_max": 1, "step": 0.05},
    )

    @model_validator(mode="before")
    @classmethod
    def _default_coverage(cls, data: object) -> object:
        if isinstance(data, dict) and data.get("coverage_area") is None:
            footprint = data.get("footprint_area")
            if footprint is not None:
                data = {**data, "coverage_area": 1.4 * footprint}
        return data


# ── single-constructor transforms (no parameterisation route) ───────────────


class IdentityCfg(_TransformCfg):
    kind: Literal["identity"] = "identity"

    def build(self) -> TemperatureTransform:
        return IDENTITY


class ShiftCfg(_TransformCfg):
    kind: Literal["shift"] = "shift"
    degree_shift: float = Field(
        default=2.0,
        title="Temperature shift",
        description="Constant offset added to the outdoor temperature.",
        json_schema_extra={"unit": "°C", "ui_min": -5, "ui_max": 15, "step": 0.1},
    )

    def build(self) -> TemperatureTransform:
        return Shift(self.degree_shift)


class FixedCfg(_TransformCfg):
    kind: Literal["fixed"] = "fixed"
    temperature: float = Field(
        default=20.0,
        title="Fixed temperature",
        description="Constant indoor temperature (setpoint).",
        json_schema_extra={"unit": "°C", "ui_min": 0, "ui_max": 40, "step": 0.5},
    )

    def build(self) -> TemperatureTransform:
        return Fixed(self.temperature)


class ConstrainedCfg(_TransformCfg):
    kind: Literal["constrained"] = "constrained"
    temperature_range: TempRange = (4.5, 35.0)

    def build(self) -> TemperatureTransform:
        return Constrained(self.temperature_range)


# ── RadiationShift: coefficients (A-ii) ↔ physical constants (A-i) ──────────


class RadCoefficients(_Frozen):
    route: Literal["coefficients"] = "coefficients"
    degree_shift: float = Field(
        default=0.0,
        title="Temperature shift",
        description="Constant offset added to the outdoor temperature.",
        json_schema_extra={"unit": "°C", "ui_min": -5, "ui_max": 15, "step": 0.1},
    )
    solar_factor: float = Field(
        default=0.01,
        ge=0.0,
        title="Solar factor",
        description="Indoor warming per unit shortwave irradiance.",
        json_schema_extra={
            "unit": "°C·m²/W",
            "ui_min": 0.0,
            "ui_max": 0.1,
            "step": 0.001,
        },
    )

    def make(self) -> RadiationShift:
        return RadiationShift(self.degree_shift, self.solar_factor)


class RadConstants(_Geometry):
    route: Literal["constants"] = "constants"

    def make(self) -> RadiationShift:
        return RadiationShift.from_constants(
            footprint_area=self.footprint_area,
            coverage_area=self.coverage_area,
            heat_transfer_coeff=self.heat_transfer_coeff,
            average_height=self.average_height,
            air_infiltration=self.air_infiltration,
            solar_coeff=self.solar_coeff,
        )


_RadParams = Annotated[RadCoefficients | RadConstants, Field(discriminator="route")]


class RadiationShiftCfg(_TransformCfg):
    kind: Literal["radiation_shift"] = "radiation_shift"
    params: _RadParams = Field(default_factory=RadCoefficients)

    def build(self) -> TemperatureTransform:
        return self.params.make()


# ── AdaptiveShift: coefficients (A-ii) ↔ physical constants (A-i) ───────────


class AdaptiveCoefficients(_Frozen):
    route: Literal["coefficients"] = "coefficients"
    base_heat_loss: float = Field(
        gt=0,
        title="Base heat loss K",
        description="Conduction + infiltration loss denominator.",
        json_schema_extra={"unit": "W/K", "ui_min": 100, "ui_max": 100000, "step": 100},
    )
    solar_factor: float = Field(
        ge=0,
        title="Solar factor",
        description="Net solar gain (eta x footprint); divided by heat loss.",
        json_schema_extra={"ui_min": 0, "ui_max": 2000, "step": 10},
    )
    ventilation: VentFlow
    heating: HeatingPower
    temperature_range: TempRange = (4.5, 35.0)

    def make(self) -> AdaptiveShift:
        return AdaptiveShift(
            self.base_heat_loss,
            self.solar_factor,
            self.temperature_range,
            self.ventilation,
            self.heating,
        )


class AdaptiveConstants(_Geometry):
    route: Literal["constants"] = "constants"
    ventilation: VentFlow
    heating: HeatingPower
    temperature_range: TempRange = (4.5, 35.0)

    def make(self) -> AdaptiveShift:
        return AdaptiveShift.from_constants(
            footprint_area=self.footprint_area,
            coverage_area=self.coverage_area,
            heat_transfer_coeff=self.heat_transfer_coeff,
            average_height=self.average_height,
            air_infiltration=self.air_infiltration,
            solar_coeff=self.solar_coeff,
            ventilation=self.ventilation,
            heating=self.heating,
            temperature_range=self.temperature_range,
        )


_AdaptiveParams = Annotated[
    AdaptiveCoefficients | AdaptiveConstants,
    Field(discriminator="route"),
]


class AdaptiveShiftCfg(_TransformCfg):
    kind: Literal["adaptive_shift"] = "adaptive_shift"
    params: _AdaptiveParams

    def build(self) -> TemperatureTransform:
        return self.params.make()


TransformConfig = Annotated[
    IdentityCfg
    | ShiftCfg
    | RadiationShiftCfg
    | AdaptiveShiftCfg
    | FixedCfg
    | ConstrainedCfg,
    Field(discriminator="kind"),
]


# ── prediction configuration ────────────────────────────────────────────────


class GdhPredictionConfig(_Frozen):
    """Configuration for gdh prediction."""

    # Bump when the hashing logic in ``compute_hash`` changes, so caches keyed
    # by the old scheme are not mistaken for hits. ``ClassVar`` keeps pydantic
    # from treating it as a field.
    _HASH_VERSION: ClassVar[int] = 1

    gdh_base: float = Field(
        default=4.5,
        ge=0,
        title="GDH base temperature",
        description="Lower threshold for growth-degree-hour accumulation.",
        json_schema_extra={"unit": "°C", "ui_min": 0, "ui_max": 15, "step": 0.5},
    )
    transform: TransformConfig = Field(default_factory=RadiationShiftCfg)

    @property
    def temperature_transform(self) -> TemperatureTransform:
        return self.transform.build()

    @classmethod
    def from_path(cls, config_yaml: Path) -> Self:
        with config_yaml.open("r", encoding="utf-8") as fh:
            return cls.model_validate(yaml.safe_load(fh))

    def write_to_path(self, config_yaml: Path) -> None:
        with config_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.model_dump(mode="json"), fh, sort_keys=False)

    @classmethod
    def write_default_config(cls, config_yaml: Path) -> None:
        cls().write_to_path(config_yaml)

    def compute_hash(self) -> str:
        # Key the processed cache by the *inputs to _process*: the resolved
        # transform's behaviour (its fingerprint) plus gdh_base.
        identity = (self.gdh_base, self.temperature_transform.fingerprint())
        payload = json.dumps(identity, separators=(",", ":"))
        key = f"{payload}_v{self._HASH_VERSION}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()[:8]


class GdhLocalizedPredictionConfig(GdhPredictionConfig):
    """Localized configuration for gdh prediction, including GPS coordinates."""

    latitude: float
    longitude: float

    def unpack(self) -> tuple[float, float, GdhPredictionConfig]:
        """Unpack its components: latitude, longitude, and the base config."""
        return (
            self.latitude,
            self.longitude,
            GdhPredictionConfig(gdh_base=self.gdh_base, transform=self.transform),
        )

    @classmethod
    def pack(
        cls,
        latitude: float,
        longitude: float,
        config: GdhPredictionConfig,
    ) -> "GdhLocalizedPredictionConfig":
        """Pack the given components into a :class:`GdhLocalizedPredictionConfig`."""
        return cls(
            latitude=latitude,
            longitude=longitude,
            gdh_base=config.gdh_base,
            transform=config.transform,
        )


# cache and config path management

XdgPaths = Literal["XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"]
DefPaths = Literal[".cache", ".config", ".local/share"]


class GdhPredictionPaths:
    """Class to hold paths for gdh prediction project."""

    name: str
    cache: Path
    config: Path
    data: Path

    def __init__(self, name: str = "gdhprediction") -> None:
        self.name = name
        self.cache = self._make_dir("XDG_CACHE_HOME", ".cache")
        self.config = self._make_dir("XDG_CONFIG_HOME", ".config")
        self.data = self._make_dir("XDG_DATA_HOME", ".local/share")

    def _make_dir(self, xdg: XdgPaths, default: DefPaths) -> Path:
        dir_path = Path(os.getenv(xdg, Path.home() / default)) / self.name
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path

    def data_file(self, hashkey: str) -> Path:
        return self.data / f"{hashkey}.npy"

    def data_file_checksum(self, hashkey: str) -> Path:
        return self.data / f"{hashkey}.sha256"

    def cache_file(self, hashkey: str) -> Path:
        return self.cache / f"{hashkey}.npy"

    def cache_file_checksum(self, hashkey: str) -> Path:
        return self.cache / f"{hashkey}.sha256"

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}.yaml"


APP_PATHS: Final = GdhPredictionPaths()
