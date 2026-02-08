from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Self, assert_never, get_type_hints

import yaml

from gdhprediction.utils import (
    FieldsHashMxn,
    RadiationShift,
    Shift,
    TemperatureTransform,
    Transforms,
)


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
