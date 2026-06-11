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

from yieldprediction.fruit_load import FruitLoadModel
from shared.configs import YieldPredictionConfig, GdhPredictionConfig
from shared.paths import YieldPredictionPaths, GdhPredictionPaths

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
    

    def __init__(self, tenant: str, batch: str, start_date: datetime.datetime, gdh_config: GdhPredictionConfig, supabase: SupaBaseInterface, influx: InfluxDBInterface) -> None:

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
        
        flm = FruitLoadModel(self._config, self.get_bins())
        flm.set_data_from_supabase(self.supabase.get_days_since_harvest(self.tenant, self.batch, self.start_date),
                                   self.supabase.get_plants_per_row(self.tenant, self.batch), 
                                   self.supabase.get_total_number_rows(self.tenant, self.batch))
        self.estimated_fruit_load = flm.get_current_fruit_load()

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
        if not Counter(self._config.growing_stages_order) == Counter(self.estimated_fruit_load[0].keys()):
            raise ValueError("The growing stages ordered must match the estimated fruit load keys")
        if self._config.days_of_prediction <= 0:
            raise ValueError("The number of days of prediction must be greater than 0")
        if self._config.latitude > 90 or self._config.latitude < -90:
            raise ValueError("The latitude must be between -90 and 90")
        if self._config.longitude > 180 or self._config.longitude < -180:
            raise ValueError("The longitude must be between -180 and 180")
        if self._config.total_gdh_needed <= 0:
            raise ValueError("The total gdh needed must be greater than 0")
        if self._config.avg_weight <= 0:
            raise ValueError("The average weight must be greater than 0")
        if not sorted(list(self._config.count_correction_factor.keys())) == sorted(self._config.growing_stages_order):
            raise ValueError("The keys of count_correction_factor must match the growing stages order")
        if not sorted(list(self._config.class_index_name_map.values())) == sorted(self._config.growing_stages_order):
            raise ValueError("The values of class_index_name_map must match the growing stages order")
        if not sorted(list(self._config.visible_plants_per_camera.keys())) == sorted(list(self._config.camera_row_map.keys())):
            raise ValueError("The keys of visible_plants_per_camera must match the keys of camera_row_map")
        
        
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

        # check bins against the config to make sure there are no issues later
        # check if the cameras match the cameras in the config
        for a_camera in self._config.visible_plants_per_camera:
            if not a_camera in bins.keys():
                raise ValueError(f"Camera {a_camera} defined in the config has no data from the counts")
            
        return bins
    
    # get the counts from the database and extrapolate them using the fruit load prediction
    def get_actual_counts_from_db(self) -> List[int]:
        data = self.influx.get_strawberry_count_data(self.tenant, self.batch, self.start_date)

        actual_counts = list()
        for day in range(self._config.days_of_prediction):
            cur_date = self.start_date+datetime.timedelta(days=day)
            cur_data_per_camera = dict()
            for a_date_str in data.keys():
                a_date = datetime.datetime.strptime(a_date_str, "%Y_%m_%d_%H_%M_%S")
                # check if 
                if a_date >= cur_date and a_date < cur_date+datetime.timedelta(days=1):
                    for a_camera in data[a_date_str].keys():
                        if not a_camera in cur_data_per_camera:
                            cur_data_per_camera[a_camera] = list()
                            cur_data_per_camera[a_camera].append(data[a_date_str][a_camera])
                        elif cur_data_per_camera[a_camera]["date"] < a_date:
                            cur_data_per_camera[a_camera].append(data[a_date_str][a_camera])
    
            # in case there are no counts its likely in the future so we append -1 which means no data
            if len(cur_data_per_camera) == 0:
                actual_counts.append(-1)
                continue

            # now we have all data for the current day
            # we likely have multiple datasets for each camera since they do record multiple times a day
            # since we predict per day we will use some avg of the day
            bins_cur_date = dict()
            for a_camera in cur_data_per_camera.keys():
                bins_cur_date[a_camera] = dict()
                # prefill the dict with all the stages we have
                for a_stage in list(self._config.class_index_name_map.values()):
                    bins_cur_date[a_camera][a_stage] = 0
                # we might have more than one result per cam here
                for a_data in cur_data_per_camera[a_camera]:
                    for a_bb_index in a_data:
                        cur_stage = a_data[a_bb_index]["class"]
                        if not cur_stage in self._config.class_index_name_map:
                            raise ValueError(f"Unknown class index {cur_stage}")
                        cur_stage = self._config.class_index_name_map[cur_stage]
                        bins_cur_date[a_camera][cur_stage] += 1
                # we might have used multiple counts for one day, normalize
                for a_stage in list(self._config.class_index_name_map.values()):
                    bins_cur_date[a_camera][a_stage] = round(bins_cur_date[a_camera][a_stage]/len(cur_data_per_camera[a_camera]))
            
            # now we got the avg bins of the day, extrapolate it to the whole greenhouse using fruit load prediction
            cur_flm = FruitLoadModel(self._config, bins_cur_date)
            cur_flm.set_data_from_supabase(self.supabase.get_days_since_harvest(self.tenant, self.batch, cur_date),
                            self.supabase.get_plants_per_row(self.tenant, self.batch), 
                            self.supabase.get_total_number_rows(self.tenant, self.batch))
            cur_estimated_fruit_load = cur_flm.get_current_fruit_load()




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
                if fruit_stage == self._config.growing_stages_order[-1]:
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
                    if fruit_stage == self._config.growing_stages_order[-1]:
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