from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, field
from pathlib import Path
from typing import Any, Self, assert_never, get_type_hints
from typing import Dict, List
from pydantic import TypeAdapter

import yaml

from yieldprediction.utils import (
    FieldsHashMxn
)


@dataclass(frozen=False)
class YieldPredictionConfig(FieldsHashMxn):
    """Configuration for yield prediction."""

    # default config

    # ---------- General config ----------
    days_of_prediction: int = 28

    # ---------- Greenhouse config ----------
    latitude: float = 51.503721859917725
    longitude: float = 7.050592540313427

    # ---------- Plant config ----------
    # percent of gdh needed for each stage to grow to the next one
    gdh_percent_stage_to_stage: Dict[str, float] = field(default_factory=lambda:{
        "Flower": 0.2,
        "Flower_Fruit": 0.2,
        "Green_Fruit": 0.2,
        "Green_White_Fruit": 0.2,
        "White_Red_Fruit": 0.2
    })
    total_gdh_needed: int = 7500
    # weight per ripe fruit in gram
    avg_weight: float = 15.0 # TODO: model weight over the growing period

    # ---------- Constraints config ----------
    constrain_min_gdh_percent_stage_to_stage: Dict[str, float] = field(default_factory=lambda:{
        "Flower": 0.1,
        "Flower_Fruit": 0.1,
        "Green_Fruit": 0.1,
        "Green_White_Fruit": 0.1,
        "White_Red_Fruit": 0.1
    })
    constrain_max_gdh_percent_stage_to_stage: Dict[str, float] = field(default_factory=lambda:{
        "Flower": 0.3,
        "Flower_Fruit": 0.3,
        "Green_Fruit": 0.3,
        "Green_White_Fruit": 0.3,
        "White_Red_Fruit": 0.3
    })


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
