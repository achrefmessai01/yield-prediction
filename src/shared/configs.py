import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Self,
    TypedDict,
    assert_never,
    get_type_hints,
)

import yaml
from pydantic import TypeAdapter

from gdhprediction.transforms import (
    IDENTITY,
    RadiationShift,
    Shift,
    TemperatureTransform,
)
from gdhprediction.utils import Transforms
from shared.utils import FieldsHashMxn


@dataclass(frozen=False, slots=True)
class Config(FieldsHashMxn):
    @classmethod
    def from_path(cls, config_yaml: Path) -> Self:
        with config_yaml.open("r", encoding="utf-8") as fh:
            config_mapping = yaml.safe_load(fh)
        cls._validate_serial(config_mapping)
        return cls(**config_mapping)

    @classmethod
    def _validate_serial(cls, config_mapping: Mapping[str, Any]) -> None:
        ftp_hints: Mapping[str, type] = get_type_hints(cls)
        for f in fields(cls):
            name = f.name
            ftp = ftp_hints[name]
            if name not in config_mapping:
                raise ValueError(f"Missing required config field: {name}")
            value = config_mapping[name]
            type_fits = False
            msg = ""
            try:
                if isinstance(value, ftp):
                    type_fits = True
                else:
                    type_fits = False
                    msg = (
                        f"Invalid type for {cls.__name__}.{name}: "
                        f"expected {ftp}, got {type(value)}."
                    )
            except:
                # if we are here we try to use isinstance on a Typing type which does not work
                try:
                    typing_type_validation = TypeAdapter(ftp)
                    typing_type_validation.validate_python(value)
                    type_fits = True
                except:
                    type_fits = False
                    msg = (
                        f"Invalid type for {cls.__name__}.{name}: "
                        f"expected {ftp}, got {type(value)}."
                    )
            if not type_fits:
                raise TypeError(msg)

    def write_config(self, config_yaml: Path) -> None:
        with config_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(asdict(self), fh)


class _GdhPredictionData(TypedDict):
    gdh_base: float
    temperature_shift: float
    solar_factor: float
    transform_key: Transforms


@dataclass(frozen=False, slots=True, kw_only=True)
class GdhPredictionConfig(Config):
    """Configuration for gdh prediction."""

    gdh_base: float = 4.5
    temperature_shift: float = 2.0
    solar_factor: float = 0.01
    transform_key: Transforms = "shift"

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


@dataclass(frozen=False, slots=True, kw_only=True)
class GdhLocalizedPredictionConfig(GdhPredictionConfig):
    """Localized configuration for gdh prediction, including GPS coordinates."""

    latitude: float = 0.0
    longitude: float = 0.0

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
        cls,
        latitude: float,
        longitude: float,
        config: GdhPredictionConfig,
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


@dataclass(frozen=False)
class YieldPredictionConfig(Config):
    """Configuration for yield prediction."""

    # default config

    # ---------- General config ----------
    days_of_prediction: int = 28

    # ---------- Plant config ----------
    # percent of gdh needed for each stage to grow to the next one
    gdh_percent_stage_to_stage: Dict[str, float] = field(
        default_factory=lambda: {
            "Flower": 0.2,
            "Flower_Fruit": 0.2,
            "Green_Fruit": 0.2,
            "Green_White_Fruit": 0.2,
            "White_Red_Fruit": 0.2,
        }
    )
    total_gdh_needed: int = 7500
    # weight per ripe fruit in gram
    avg_weight: float = 15.0  # TODO: model weight over the growing period

    # ---------- Constraints config ----------
    constrain_min_gdh_percent_stage_to_stage: Dict[str, float] = field(
        default_factory=lambda: {
            "Flower": 0.1,
            "Flower_Fruit": 0.1,
            "Green_Fruit": 0.1,
            "Green_White_Fruit": 0.1,
            "White_Red_Fruit": 0.1,
        }
    )
    constrain_max_gdh_percent_stage_to_stage: Dict[str, float] = field(
        default_factory=lambda: {
            "Flower": 0.3,
            "Flower_Fruit": 0.3,
            "Green_Fruit": 0.3,
            "Green_White_Fruit": 0.3,
            "White_Red_Fruit": 0.3,
        }
    )
    class_index_name_map: Dict[int, str] = field(
        default_factory=lambda: {
            1: "Flower",
            2: "Flower_Fruit",
            3: "Green_Fruit",
            4: "Green_White_Fruit",
            5: "White_Red_Fruit",
            6: "Red_Fruit",
        }
    )

    visible_plants_per_camera: Dict[str, float] = field(
        default_factory=lambda: {"camera1": 4, "camera2": 4, "camera3": 2}
    )
    camera_row_map: Dict[str, int] = field(
        default_factory=lambda: {"camera1": 0, "camera2": 12, "camera3": 26}
    )
    growing_stages_order: List[str] = field(
        default_factory=lambda: [
            "Flower",
            "Flower_Fruit",
            "Green_Fruit",
            "Green_White_Fruit",
            "White_Red_Fruit",
            "Red_Fruit",
        ]
    )
    count_correction_factor: Dict[str, float] = field(
        default_factory=lambda: {
            "Flower": 1,
            "Flower_Fruit": 1,
            "Green_Fruit": 1,
            "Green_White_Fruit": 1,
            "White_Red_Fruit": 1,
            "Red_Fruit": 1,
        }
    )

    # maximal multiplier of the occlusion correction, must be larger than 1
    occlusion_correction_factor_max = 1.5

    # as a first idea just use a sigmoid function scaled between 1 and occlusion_correction_factor_max
    # Should be tuned later, at around 100 counts this function should be close to the max
    def get_occlusion_correction_factor(self, detected_objs: int) -> float:
        if self.occlusion_correction_factor_max <= 1:
            raise ValueError("occlusion_correction_factor_max must be larger than 1")
        return (
            (self.occlusion_correction_factor_max - 1)
            / (1 + math.exp(-0.07 * detected_objs + 5))
        ) + 1
