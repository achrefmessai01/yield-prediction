import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Final, Literal, Self, assert_never, get_type_hints

import yaml

from gdhprediction.transforms import (
    IDENTITY,
    RadiationShift,
    Shift,
    TemperatureTransform,
)
from gdhprediction.utils import FieldsHashMxn, Transforms

# processing configuration


@dataclass(frozen=True, slots=True)
class GdhPredictionConfig(FieldsHashMxn):
    """Configuration for gdh prediction."""

    gdh_base: float = 4.5
    temperature_shift: float = 2.0
    solar_factor: float = 0.01
    transform_key: Transforms = "shift"

    @classmethod
    def from_path(cls, config_yaml: Path) -> Self:
        with config_yaml.open("r", encoding="utf-8") as fh:
            config_mapping = yaml.safe_load(fh)
        cls._validate_serial(config_mapping)
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
    def _validate_serial(cls, config_mapping: Mapping[str, Any]) -> None:
        ftp_hints: Mapping[str, type] = get_type_hints(cls)
        for f in fields(cls):
            name = f.name
            ftp = ftp_hints[name]
            if name not in config_mapping:
                raise ValueError(f"Missing required config field: {name}")
            value = config_mapping[name]
            if not isinstance(value, ftp):
                msg = (
                    f"Invalid type for {cls.__name__}.{name}: "
                    f"expected {ftp}, got {type(value)}."
                )
                raise TypeError(msg)

    @classmethod
    def write_default_config(cls, config_yaml: Path) -> None:
        default_cfg = cls()
        with config_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(default_cfg), fh)


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

    @property
    def forecast_cache(self) -> Path:
        return self.cache / "forecast.json"

    def climate_data(self, hashkey: str) -> Path:
        return self.data / f"climate_{hashkey}.npy"

    def climate_data_checksum(self, hashkey: str) -> Path:
        return self.data / f"climate_{hashkey}.sha256"

    @property
    def model_config(self) -> Path:
        return self.config / f"{self.name}.yaml"


APP_PATHS: Final = GdhPredictionPaths()
