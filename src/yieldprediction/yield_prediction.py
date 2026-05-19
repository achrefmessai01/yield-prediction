from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Self
from dataclasses import asdict
from typing import Dict, List, Tuple
from collections import Counter
import yaml
import numpy as np
from random import randint
import os
import csv
import datetime

from fruitloadprediction.fruit_load import FruitLoadModel
from fruitloadprediction.config import FruitloadPredictionConfig
from fruitloadprediction.paths import FruitloadPredictionPaths

from gdhprediction.config import GdhPredictionConfig
from gdhprediction.apis import get_gdh_prediction

from yieldprediction.config import YieldPredictionConfig
from yieldprediction.paths import YieldPredictionPaths

class YieldPrediction:
    """Main entry point for the yield prediction application. Using the fruitload and gdh prediction"""

    estimated_fruit_load: list
    growing_stages_ordered: list
    estimated_gdh: list

    _config: YieldPredictionConfig
    paths: YieldPredictionPaths

    tenant: str
    batch: str
    

    def __init__(self, tenant: str, batch: str) -> None:

        if tenant == "":
            raise ValueError("Tenant cannot be empty")
        if batch == "":
            raise ValueError("Batch cannot be empty")
        self.tenant = tenant
        self.batch = batch

        self.paths = YieldPredictionPaths(tenant=tenant, batch=batch)
        self._config = YieldPredictionConfig()
        if not os.path.exists(self.paths.model_config):
            raise FileNotFoundError(f"Model config file not found at {self.paths.model_config}")
        self._config.from_path(self.paths.model_config)
        
        fruitload_paths = FruitloadPredictionPaths(tenant=tenant, batch=batch)
        fruitload_config = FruitloadPredictionConfig()
        if not os.path.exists(fruitload_paths.model_config):
            raise FileNotFoundError(f"Model config file not found at {fruitload_paths.model_config}")
        fruitload_config.from_path(fruitload_paths.model_config)
        flm = FruitLoadModel(fruitload_config, self.get_bins())
        self.estimated_fruit_load, self. growing_stages_ordered = flm.get_current_fruit_load()

        gdh_config = GdhPredictionConfig()
        self.estimated_gdh = get_gdh_prediction(
            self._config.latitude,
            self._config.longitude,
            gdh_config,
            frequency="daily"
        )
        print(self.estimated_gdh)
        self.convert_gdh()
        print(self.estimated_gdh)
        self.check_parameters()

    def check_parameters(self) -> None:
        # check if the gdh_percent_stage_to_stage add up to 1
        if sum(self._config.gdh_percent_stage_to_stage.values()) != 1:
            raise ValueError("The sum of gdh_percent_stage_to_stage must be equal to 1")
        if len(self.estimated_gdh) < self._config.days_of_prediction:
            raise ValueError("The number of days of prediction must be equal to the length of estimated_gdh")
        if not Counter(self.growing_stages_ordered) == Counter(self.estimated_fruit_load[0].keys()):
            raise ValueError("The growing stages ordered must match the estimated fruit load keys")
        
    def convert_gdh(self) -> None:
        last_val = self.estimated_gdh[0]
        for day in range(1, len(self.estimated_gdh)):
            new_val = self.estimated_gdh[day] - last_val
            last_val = self.estimated_gdh[day]
            self.estimated_gdh[day] = new_val

    def get_bins(self) -> Dict[str, Dict[str, int]]:
        # TODO: add db request to get bins, return dummy for local testing
        bins = {
            "camera1" : {
                "Flower" : 39,
                "Flower_Fruit" : 53,
                "Green_Fruit" : 45,
                "Green_White_Fruit" : 16,
                "White_Red_Fruit" : 13,
                "Red_Fruit" : 1
            },
            "camera2" : {
                "Flower" : 36,
                "Flower_Fruit" : 11,
                "Green_Fruit" : 24,
                "Green_White_Fruit" : 7, 
                "White_Red_Fruit" : 20,
                "Red_Fruit" : 17
            },
            "camera3" : {
                "Flower" : 5,
                "Flower_Fruit" : 8,
                "Green_Fruit" : 12,
                "Green_White_Fruit" : 35,
                "White_Red_Fruit" : 15,
                "Red_Fruit" : 37
            }
        }
        return bins

    def get_prediction(self, store_for_training: bool = False) -> Tuple[List[List[int]], List[List[float]]]:
        prediction_count = list()
        prediction_weight = list()
        for row in range(len(self.estimated_fruit_load)):
            cur_prediction_count = list()
            full_steps = list()

            cur_fruitload = self.estimated_fruit_load[row]

            # lets simulate each fruit in its state and the gdh it has collected so far
            # for now we just init it randomly but later we should store this values to be more accurate
            cur_fruit_stages = dict()
            for fruit_stage, count in cur_fruitload.items():
                if fruit_stage == self.growing_stages_ordered[-1]:
                    # for the ripe fruit there is no need to collect any more gdh so we ignore them
                    cur_fruit_stages[fruit_stage] = [0 for _ in range(count)]
                    continue
                cur_stage_needed_gdh = int(self._config.gdh_percent_stage_to_stage[fruit_stage]*self._config.total_gdh_needed)
                cur_fruit_stages[fruit_stage] = [randint(0, cur_stage_needed_gdh) for _ in range(count)]
            full_steps.append({stage : len(fruitlist) for stage, fruitlist in cur_fruit_stages.items()})

            for day in range(self._config.days_of_prediction):
                cur_day_gdh = self.estimated_gdh[day]
                # advance the fruit stages by one day
                evolved_fruits_next = list()
                evolved_fruits_cur = list()
                for fruit_stage in cur_fruit_stages:
                    if fruit_stage == self.growing_stages_ordered[-1]:
                        # no need to look at the last stage, it will not further develop
                        # but we need to add the fruits that developed into the last stage
                        cur_fruit_stages[fruit_stage] = cur_fruit_stages[fruit_stage]+evolved_fruits_cur
                        cur_prediction_count.append(len(cur_fruit_stages[fruit_stage]))
                        continue
                    cur_stage_needed_gdh = int(self._config.gdh_percent_stage_to_stage[fruit_stage]*self._config.total_gdh_needed)
                    cur_stage_updated = list()
                    for i in range(len(cur_fruit_stages[fruit_stage])):
                        cur_fruit_gdh = cur_fruit_stages[fruit_stage][i] + cur_day_gdh
                        if cur_stage_needed_gdh > cur_fruit_gdh:
                            cur_stage_updated.append(cur_fruit_gdh)
                        else:
                            evolved_fruits_next.append(cur_fruit_gdh - cur_stage_needed_gdh)
                    cur_fruit_stages[fruit_stage] = cur_stage_updated+evolved_fruits_cur
                    evolved_fruits_cur = evolved_fruits_next.copy()
                    evolved_fruits_next = list()
                full_steps.append({stage : len(fruitlist) for stage, fruitlist in cur_fruit_stages.items()})

            # use avg size to get the weight
            prediction_weight.append(list(self._config.avg_weight * np.array(cur_prediction_count)))
            if store_for_training:
                cur_time = datetime.datetime.now()
                store_path = os.path.join(self.paths.data, "full_pred_data-{date:%Y-%m-%d_%H:%M:%S}.csv".format(date=cur_time))
                print(store_path)
                with open(store_path, 'w', newline='') as output_file:
                    dict_writer = csv.DictWriter(output_file, full_steps[0].keys())
                    dict_writer.writeheader()
                    dict_writer.writerows(full_steps)
                # also store the used yaml config to make sure we always have the correct parameters used
                with open(os.path.join(self.paths.data, "config-{date:%Y-%m-%d_%H:%M:%S}.yaml".format(date=cur_time)), 'w') as config_file:
                    yaml.safe_dump(asdict(self._config), config_file)
            
            prediction_count.append(cur_prediction_count)
                
        # Results are per row now, but we only need the accumulated results over all rows
        accumulated_prediction_count = list()
        accumulated_prediction_weights = list()
        for row_index in range(len(prediction_count)):
            for day in range(len(prediction_count[row_index])):
                if row_index == 0:
                    accumulated_prediction_count.append(0)
                    accumulated_prediction_weights.append(0)
                accumulated_prediction_count[day] += prediction_count[row][day]
                accumulated_prediction_weights[day] += float(prediction_weight[row][day])
        return accumulated_prediction_count, accumulated_prediction_weights