import os
from collections.abc import Mapping
from dataclasses import KW_ONLY, asdict, dataclass, fields
from pathlib import Path
from typing import (
    Any,
    Final,
    Literal,
    Self,
    TypedDict,
    assert_never,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from gdhprediction.transforms import (
    IDENTITY,
    RadiationShift,
    Shift,
    TemperatureTransform,
)
from gdhprediction.utils import FieldsHashMxn, Transforms

# processing configuration


class _GdhPredictionData(TypedDict):
    gdh_base: float
    temperature_shift: float
    solar_factor: float
    transform_key: Transforms


@dataclass(frozen=True, slots=True)
class GdhPredictionConfig(FieldsHashMxn):
    """Configuration for gdh prediction."""

    _: KW_ONLY
    gdh_base: float = 4.5
    temperature_shift: float = 2.0
    solar_factor: float = 0.01
    transform_key: Transforms = "shift"

    @classmethod
    def from_path(cls, config_yaml: Path) -> Self:
        with config_yaml.open("r", encoding="utf-8") as fh:
            config_mapping = yaml.safe_load(fh)
        config_mapping = cls._validate_serial(config_mapping)
        return cls(**config_mapping)

    @property
    def temperature_transform(self) -> TemperatureTransform:
        key = self.transform_key
        match key:
            case "identity":
                return IDENTITY
            case "shift":
                return Shift(self.temperature_shift)
            case "radiation_shift":
                return RadiationShift(self.temperature_shift, self.solar_factor)
            case _:
                assert_never(key)

    @classmethod
    def _validate_serial(cls, config_mapping: Mapping[str, Any]) -> _GdhPredictionData:
        ftp_hints: Mapping[str, type] = get_type_hints(cls)
        for f in fields(cls):
            name = f.name
            ftp = ftp_hints[name]
            if name not in config_mapping:
                raise ValueError(f"Missing required config field: {name}")
            value = config_mapping[name]
            # special case for the Transforms literal
            if get_origin(ftp) is Literal:
                if not isinstance(value, str):
                    msg = (
                        f"Invalid type for {cls.__name__}.{name}: "
                        f"expected a string literal, got {type(value)}."
                    )
                    raise TypeError(msg)
                allowed: tuple[Transforms, ...] = get_args(ftp)
                if value not in allowed:
                    msg = (
                        f"Invalid value for {cls.__name__}.{name}: "
                        f"expected one of {allowed}, got {value!r}."
                    )
                    raise ValueError(msg)
                continue
            if not isinstance(value, ftp):
                msg = (
                    f"Invalid type for {cls.__name__}.{name}: "
                    f"expected {ftp}, got {type(value)}."
                )
                raise TypeError(msg)
        # cast after validation
        return cast("_GdhPredictionData", config_mapping)

    @classmethod
    def write_default_config(cls, config_yaml: Path) -> None:
        default_cfg = cls()
        default_cfg.write_to_path(config_yaml)

    def write_to_path(self, config_yaml: Path) -> None:
        with config_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh)


@dataclass(frozen=True, slots=True)
class GdhLocalizedPredictionConfig(GdhPredictionConfig):
    """Localized configuration for gdh prediction, including GPS coordinates."""

    latitude: float
    longitude: float

    def unpack(self) -> tuple[float, float, GdhPredictionConfig]:
        """Unpack its components: latitude, longitude, and the base transform config."""
        return (
            self.latitude,
            self.longitude,
            GdhPredictionConfig(
                gdh_base=self.gdh_base,
                temperature_shift=self.temperature_shift,
                solar_factor=self.solar_factor,
                transform_key=self.transform_key,
            ),
        )

    @classmethod
    def pack(
        cls, latitude: float, longitude: float, config: GdhPredictionConfig,
    ) -> "GdhLocalizedPredictionConfig":
        """Pack the given components into a :class:`GdhLocalizedPredictionConfig`."""
        return GdhLocalizedPredictionConfig(
            latitude=latitude,
            longitude=longitude,
            gdh_base=config.gdh_base,
            temperature_shift=config.temperature_shift,
            solar_factor=config.solar_factor,
            transform_key=config.transform_key,
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
