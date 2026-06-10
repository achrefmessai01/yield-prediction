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
from shared.configs import FruitloadPredictionConfig, YieldPredictionConfig, GdhPredictionConfig
from shared.paths import FruitloadPredictionPaths, YieldPredictionPaths, GdhPredictionPaths

from shared.supabase_interface import SupaBaseInterface
from shared.influxdb_interface import InfluxDBInterface

from gdhprediction.apis import get_gdh_prediction

class YieldPrediction:
    """Yield prediction application, using the fruitload and gdh prediction"""

    estimated_fruit_load: list
    growing_stages_ordered: list
    estimated_gdh: list

    _config: YieldPredictionConfig
    paths: YieldPredictionPaths

    tenant: str
    batch: str

    influx: InfluxDBInterface
    supabase: SupaBaseInterface

    start_date: datetime.datetime
    

    def __init__(self, tenant: str, batch: str, start_date: datetime.datetime, supabase: SupaBaseInterface, influx: InfluxDBInterface) -> None:

        if tenant == "":
            raise ValueError("Tenant cannot be empty")
        if batch == "":
            raise ValueError("Batch cannot be empty")
        self.tenant = tenant
        self.batch = batch

        self.influx = influx
        self.supabase = supabase

        self.start_date = start_date

        self.paths = YieldPredictionPaths(tenant=tenant, batch=batch)
        if not os.path.exists(self.paths.model_config):
            raise FileNotFoundError(f"Model config file not found at {self.paths.model_config}")
        self._config = YieldPredictionConfig.from_path(self.paths.model_config)
        
        fruitload_paths = FruitloadPredictionPaths(tenant=tenant, batch=batch)
        if not os.path.exists(fruitload_paths.model_config):
            raise FileNotFoundError(f"Model config file not found at {fruitload_paths.model_config}")
        fruitload_config = FruitloadPredictionConfig.from_path(fruitload_paths.model_config)
        flm = FruitLoadModel(fruitload_config, self.get_bins())
        flm.set_data_from_supabase(self.supabase.get_days_since_harvest(self.tenant, self.batch),
                                   self.supabase.get_plants_per_row(self.tenant, self.batch), 
                                   self.supabase.get_total_number_rows(self.tenant, self.batch))
        self.estimated_fruit_load, self.growing_stages_ordered = flm.get_current_fruit_load()

        gdh_paths = GdhPredictionPaths()
        if not os.path.exists(gdh_paths.model_config):
            raise FileNotFoundError(f"Model config file not found at {gdh_paths.model_config}")
        gdh_config = GdhPredictionConfig.from_path(gdh_paths.model_config)
        self.estimated_gdh = get_gdh_prediction(
            self._config.latitude,
            self._config.longitude,
            gdh_config,
            frequency="daily"
        )
        self.convert_gdh()
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

    # get data from Influx and convert to bins
    def get_bins(self) -> Dict[str, Dict[str, int]]:
        bins = dict()

        # since we give the start date we need to give a date before our start date
        # we will use one day here since the data is otherwise too old anyway
        data = self.influx.get_strawberry_count_data(self.tenant, self.batch, self.start_date-datetime.timedelta(days=1))
        # for each camera get the data at timestamp thats closest to our start date but still before the start date
        data_per_camera = dict()
        for a_date_str in data.keys():
            a_date = datetime.datetime.strptime(a_date_str, "%Y_%m_%d_%H_%M_%S")
            # we can not use data in the future (from perspective of start date)
            if a_date > self.start_date:
                continue
            for a_camera in data[a_date_str].keys():
                if not a_camera in data_per_camera:
                    data_per_camera[a_camera] = dict()
                    data_per_camera[a_camera]["date"] = a_date
                    data_per_camera[a_camera]["data"] = data[a_date_str][a_camera]
                elif data_per_camera[a_camera]["date"] < a_date:
                    # here the current date is closer to the start date than the previous one
                    data_per_camera[a_camera]["date"] = a_date
                    data_per_camera[a_camera]["data"] = data[a_date_str][a_camera]
        
        # we should have data for each camera now
        for a_camera in data_per_camera.keys():
            bins[a_camera] = dict()
            # prefill the dict with all the stages we have
            for a_stage in list(self._config.class_index_name_map.values()):
                bins[a_camera][a_stage] = 0
            for a_bb_index in data_per_camera[a_camera]["data"].keys():
                cur_stage = data_per_camera[a_camera]["data"][a_bb_index]["class"]
                if not cur_stage in self._config.class_index_name_map:
                    raise ValueError(f"Unknown class index {cur_stage}")
                cur_stage = self._config.class_index_name_map[cur_stage]
                bins[a_camera][cur_stage] += 1

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