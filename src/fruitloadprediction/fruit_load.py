"""
This script is used to calculate the fruit load at a specific time in the greenhouse.
It uses the counts of ripeness stages extracted from the images as well as some metadata about the camera setup and greenhouse.
"""

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Self
from typing import Dict, List, Tuple

import numpy as np

from shared.configs import FruitloadPredictionConfig
from shared.paths import FruitloadPredictionPaths


class FruitLoadModel:

    # current counts of each stage per camera
    bins: Dict[str, Dict[str, int]]
    # days since last harvest for each row (row index = list index)
    days_since_harvest: List[int]
    plants_per_row: int
    total_number_rows: int

    # the beta for the softmax function used to weight the cameras for each row based on how long the last harvest was in days
    # higher values makes the weights sharped in terms giving more weight to cameras in rows that better match the days since last harvest
    beta: float = 2


    _config: FruitloadPredictionConfig

    def __init__(self, config: FruitloadPredictionConfig, bins: Dict[str, Dict[str, int]]) -> None:
        self._config = config
        self.bins = bins

    def set_data_from_supabase(self, days_since_harvest: List[int], plants_per_row: int, total_number_rows: int):
        self.days_since_harvest = days_since_harvest
        self.plants_per_row = plants_per_row
        self.total_number_rows = total_number_rows

    # return fruitload per row
    def get_current_fruit_load(self) -> Tuple[List[Dict[str, int]], List[str]]:
        estimated_fruit_load = list()
        days_since_harvest_array = np.array([self.days_since_harvest[r] for r in self._config.camera_row_map.values()])
        for row in range(self.total_number_rows):
            cur_days_since_harvest = self.days_since_harvest[row]
            # each camera gets a weight because in case a row was harvested 3 days ago but the camera we are using is in a row that was harvested 1 day ago the prediction is likely quite off
            # weight sums up to 1 for later multiplication
            cur_camera_weights = dict()
            diffs_per_cam = np.abs(days_since_harvest_array-cur_days_since_harvest)
            weights_array = np.exp(-self.beta*diffs_per_cam)
            weights_array = weights_array/np.sum(weights_array)
            for cam, weight in zip(self._config.camera_row_map.keys(), weights_array):
                cur_camera_weights[cam] = weight

            cur_estimated_fruit_load = dict()
            for growing_stage in self._config.growing_stages_order:
                cur_estimated_fruit_load[growing_stage] = 0
            num_plants_observed = sum(self._config.visible_plants_per_camera.values()) # total amount of plants monitored by cameras
            for cam, cur_bins in self.bins.items():
                if not cam in self._config.visible_plants_per_camera.keys():
                    raise ValueError("Camera "+str(cam)+" has not data for visible number of plants, please add it in the config")
                cur_cam_weight = cur_camera_weights[cam]
                cur_visible_plants = self._config.visible_plants_per_camera[cam]
                if cur_visible_plants <= 0:
                    raise ValueError("Camera "+str(cam)+" has no visible plants, please add it in the config")
                for stage, count in cur_bins.items():
                    if not stage in cur_estimated_fruit_load.keys():
                        raise ValueError("Stage "+str(stage)+" is not in growing_stages_order")
                    # weight each cam by the calculated weight, since it sums to 1 we can just use the sum
                    # for the count we need to calc the count per plant (based on number plants the current cam sees) first and multiply that by the plants per row to calc for the whole row
                    cur_estimated_fruit_load[stage] += round(cur_cam_weight*((count/cur_visible_plants)*self.plants_per_row))
            estimated_fruit_load.append(cur_estimated_fruit_load)
        return estimated_fruit_load, self._config.growing_stages_order