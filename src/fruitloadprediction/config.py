from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Any, Self, assert_never, get_type_hints
from typing import Dict, List
from pydantic import TypeAdapter

import math
import yaml

from fruitloadprediction.utils import (
    FieldsHashMxn
)


@dataclass(frozen=False, slots=True)
class FruitloadPredictionConfig(FieldsHashMxn):
    """Configuration for fruitload prediction."""

    # default config
    visible_plants_per_camera: Dict[str, int] = field(default_factory=lambda:{
        "camera1": 4,
        "camera2": 4,
        "camera3": 2
    })
    camera_row_map: Dict[str, int] = field(default_factory=lambda:{
        "camera1": 0,
        "camera2": 12,
        "camera3": 26
    })
    growing_stages_order: List[str] = field(default_factory=lambda:[
        "Flower",
        "Flower_Fruit",
        "Green_Fruit",
        "Green_White_Fruit",
        "White_Red_Fruit",
        "Red_Fruit"
    ])
    count_correction_factor: Dict[str, float] = field(default_factory=lambda:{
        "Flower": 1,
        "Flower_Fruit": 1,
        "Green_Fruit": 1,
        "Green_White_Fruit": 1,
        "White_Red_Fruit": 1,
        "Red_Fruit": 1
    })

    # maximal multiplier of the occlusion correction, must be larger than 1
    occlusion_correction_factor_max = 1.5
    # as a first idea just use a sigmoid function scaled between 1 and occlusion_correction_factor_max
    # Should be tuned later, at around 100 counts this function should be close to the max
    def get_occlusion_correction_factor(self, detected_objs: int) -> float:
        if self.occlusion_correction_factor_max <= 1:
            raise ValueError("occlusion_correction_factor_max must be larger than 1")
        return ((self.occlusion_correction_factor_max-1)/(1+math.exp(-0.07*detected_objs+5)))+1

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
        as_dict = asdict(self)
        with config_yaml.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(as_dict, fh)
