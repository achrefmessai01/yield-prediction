import os
import json
from pathlib import Path

from shared.configs import GdhLocalizedPredictionConfig, YieldPredictionConfig
from shared.paths import  GdhPredictionPaths, YieldPredictionPaths

from shared.supabase_interface import SupaBaseInterface
from shared.influxdb_interface import InfluxDBInterface
from shared.coco_interface import get_coco_classes

class ConfigHandling:

    configs_found: bool = False
    yield_prediction_config: YieldPredictionConfig = None
    gdh_prediction_config: GdhLocalizedPredictionConfig = None


    def __init__(self, tenant: str, batch: str, greenhouse: str):
        self.tenant = tenant
        self.batch = batch

        self.yield_prediction_paths = YieldPredictionPaths(tenant=tenant, batch=batch)
        self.gdh_prediction_paths = GdhPredictionPaths(tenant=tenant, greenhouse=greenhouse)

    # just get the default configs
    def create_default_configs(self):
        self.yield_prediction_config = YieldPredictionConfig()
        self.gdh_prediction_config = GdhLocalizedPredictionConfig()

    # check if files are present and if yes load from there
    def load_from_files(self):
        if os.path.exists(self.yield_prediction_paths.model_config) and os.path.exists(self.gdh_prediction_paths.model_config):
            self.configs_found = True
            self.yield_prediction_config = YieldPredictionConfig.from_path(self.yield_prediction_paths.model_config)
            self.gdh_prediction_config = GdhLocalizedPredictionConfig.from_path(self.gdh_prediction_paths.model_config)
            self.configs_found = True
        else:
            raise FileNotFoundError("One or more model configuration files not found.")
        
    # load all data that can be collected from the database
    # the rest of the data will be defaults or if the flag is set and files are present, read from files
    # we will also need the coco config json files to get the classes present in the 
    def load_from_database(self, influx: InfluxDBInterface, supabase: SupaBaseInterface, coco_configs_path: Path, read_from_files: bool = False):
        if read_from_files:
            try:
                self.load_from_files()
            except Exception as e:
                print(f"Error when loading config files: {e} Will use default configs")
                self.create_default_configs
        else:
            self.create_default_configs()

        # here we either have default config or config from files
        # now we can fill that with data from the database

        # SupaBase data
        # get species (e.g. strawberry) and based on that the classes we get from the coco configs
        species = supabase.get_species(self.tenant, self.batch)
        class_index_name_map = get_coco_classes(coco_configs_path, species)
        self.yield_prediction_config.class_index_name_map = class_index_name_map
        classes = list(class_index_name_map.values())
        # set everything in the config
        self.yield_prediction_config.growing_stages_order = classes
        self.yield_prediction_config.count_correction_factor = {c:1 for c in classes}
        self.yield_prediction_config.gdh_percent_stage_to_stage = {c:(1./(len(classes)-1)) for c in classes[:-1]}
        self.yield_prediction_config.constrain_min_gdh_percent_stage_to_stage = {c:((1./(len(classes)-1))*0.5) for c in classes[:-1]}
        self.yield_prediction_config.constrain_max_gdh_percent_stage_to_stage = {c:((1./(len(classes)-1))*1.5) for c in classes[:-1]}

        # InfluxDB data
        # get the camera names
        bb_data = influx.get_strawberry_count_data(self.tenant, self.batch)
        camera_list = list()
        for t in bb_data.keys():
            for c in bb_data[t].keys():
                if not c in camera_list:
                    camera_list.append(c)
        self.yield_prediction_config.visible_plants_per_camera = {c:0 for c in camera_list}
        self.yield_prediction_config.camera_row_map = {c:0 for c in camera_list} # TODO set row when they are correct in the influx later

        