from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, Literal, Self
from typing import Dict, List, Tuple
from collections import Counter
import yaml
import numpy as np
from random import randint

from fruitloadprediction.fruit_load import FruitLoadModel
from fruitloadprediction.config import FruitloadPredictionConfig

from gdhprediction.config import GdhPredictionConfig


from yieldprediction.config import YieldPredictionConfig
from yieldprediction.paths import APP_PATHS, YieldPredictionPaths

class YieldPrediction:
    """Main entry point for the yield prediction application. Using the fruitload and gdh prediction"""

    estimated_fruit_load: list
    growing_stages_ordered: list
    estimated_gdh: list

    _config: YieldPredictionConfig
    

    def __init__(self, yield_pred_config: YieldPredictionConfig, fruitload_config: FruitloadPredictionConfig, gdh_config: GdhPredictionConfig) -> None:
        self._config = yield_pred_config
        
        flm = FruitLoadModel(fruitload_config, self.get_bins())
        self.estimated_fruit_load, self. growing_stages_ordered = flm.get_current_fruit_load()
        # TODO: get actual list of gdh list per day here
        self.estimated_gdh = [randint(230, 290) for _ in range(28)]
        self.check_parameters()

    def check_parameters(self) -> None:
        # check if the gdh_percent_stage_to_stage add up to 1
        if sum(self._config.gdh_percent_stage_to_stage.values()) != 1:
            raise ValueError("The sum of gdh_percent_stage_to_stage must be equal to 1")
        if len(self.estimated_gdh) < self._config.days_of_prediction:
            raise ValueError("The number of days of prediction must be equal to the length of estimated_gdh")
        if not Counter(self.growing_stages_ordered) == Counter(self.estimated_fruit_load.keys()):
            raise ValueError("The growing stages ordered must match the estimated fruit load keys")
        
    def get_bins(self) -> Dict[str, Dict[str, int]]:
        # TODO: add db request to get bins, return dummy for local testing
        bins = {
            "camera1" : {
                "Flower" : 10,
                "Flower_Fruit" : 10,
                "Green_Fruit" : 10,
                "Green_White_Fruit" : 10,
                "White_Red_Fruit" : 10,
                "Red_Fruit" : 10
            },
            "camera2" : {
                "Flower" : 10,
                "Flower_Fruit" : 10,
                "Green_Fruit" : 10,
                "Green_White_Fruit" : 10,
                "White_Red_Fruit" : 10,
                "Red_Fruit" : 10
            },
            "camera3" : {
                "Flower" : 5,
                "Flower_Fruit" : 5,
                "Green_Fruit" : 5,
                "Green_White_Fruit" : 5,
                "White_Red_Fruit" : 5,
                "Red_Fruit" : 5
            }
        }
        return bins

    def get_prediction(self) -> Tuple[List[int], List[float]]:
        prediction_count = list()

        # lets simulate each fruit in its state and the gdh it has collected so far
        # for now we just init it randomly but later we should store this values to be more accurate
        cur_fruit_stages = dict()
        for fruit_stage, count in self.estimated_fruit_load.items():
            if fruit_stage == self.growing_stages_ordered[-1]:
                # for the ripe fruit there is no need to collect any more gdh so we ignore them
                cur_fruit_stages[fruit_stage] = [0 for _ in range(count)]
                continue
            cur_stage_needed_gdh = int(self._config.gdh_percent_stage_to_stage[fruit_stage]*self._config.total_gdh_needed)
            cur_fruit_stages[fruit_stage] = [randint(0, cur_stage_needed_gdh) for _ in range(count)]

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
                    prediction_count.append(len(cur_fruit_stages[fruit_stage]))
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
            # when the day is over we need to model the adding of new stage 1 fruits (flowers in strawberry case)

        # use avg size to get the weight
        prediction_weight = self._config.avg_weight * np.array(prediction_count)

        return prediction_count, prediction_weight
        

if __name__ == "__main__":
    yield_conf = YieldPredictionConfig()
    fruitload_conf = FruitloadPredictionConfig()
    gdh_conf = GdhPredictionConfig()

    yp = YieldPrediction(yield_pred_config=yield_conf, fruitload_config=fruitload_conf, gdh_config=gdh_conf)
    pred_count, pred_weight = yp.get_prediction()
    print(pred_weight)
    import matplotlib.pyplot as plt
    plt.plot(pred_count)
    plt.show()