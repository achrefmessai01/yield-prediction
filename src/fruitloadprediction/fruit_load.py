"""
This script is used to calculate the fruit load at a specific time in the greenhouse.
It uses the counts of ripeness stages extracted from the images as well as some metadata about the camera setup and greenhouse.
"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Self
from typing import Dict, List, Tuple

import yaml

from fruitloadprediction.config import FruitloadPredictionConfig
from fruitloadprediction.paths import APP_PATHS, FruitloadPredictionPaths


class FruitLoadModel:

    # current counts of each stage per camera
    bins: Dict[str, Dict[str, int]]


    _config: FruitloadPredictionConfig

    def __init__(self, config: FruitloadPredictionConfig, bins: Dict[str, Dict[str, int]]) -> None:
        self._config = config
        self.bins = bins

    def get_current_fruit_load(self) -> Tuple[Dict[str, int], List[str]]:
        estimated_fruit_load = dict()
        for growing_stage in self._config.growing_stages_order:
            estimated_fruit_load[growing_stage] = 0
        num_plants_observed = sum(self._config.visible_plants_per_camera.values()) # total amount of plants monitored by cameras
        for cam, cur_bins in self.bins.items():
            if not cam in self._config.visible_plants_per_camera.keys():
                raise ValueError("Camera "+str(cam)+" has not data for visible number of plants, please add it in the config")
            cur_visible_plants = self._config.visible_plants_per_camera[cam]
            cur_cam_influence = (cur_visible_plants/num_plants_observed)*self._config.total_plants
            for stage, count in cur_bins.items():
                if not stage in estimated_fruit_load.keys():
                    raise ValueError("Stage "+str(stage)+" is not in growing_stages_order")
                estimated_fruit_load[stage] += round((self._config.count_correction_factor[stage]*self._config.get_occlusion_correction_factor(count)*(count/cur_visible_plants))*cur_cam_influence)
        return estimated_fruit_load, self._config.growing_stages_order