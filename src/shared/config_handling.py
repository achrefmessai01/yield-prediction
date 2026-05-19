import os



from fruitloadprediction.config import FruitloadPredictionConfig
from fruitloadprediction.paths import FruitloadPredictionPaths

from gdhprediction.config import GdhPredictionConfig, GdhPredictionPaths

from yieldprediction.config import YieldPredictionConfig
from yieldprediction.paths import YieldPredictionPaths



class ConfigHandling:

    configs_found: bool = False
    fruitload_config: FruitloadPredictionConfig = None
    yield_prediction_config: YieldPredictionConfig = None
    gdh_prediction_config: GdhPredictionConfig = None


    def __init__(self, tenant: str, batch: str):
        self.tenant = tenant
        self.batch = batch

        fruitload_paths = FruitloadPredictionPaths(tenant=tenant, batch=batch)
        yield_prediction_paths = YieldPredictionPaths(tenant=tenant, batch=batch)
        gdh_prediction_paths = GdhPredictionPaths()

        if os.path.exists(fruitload_paths.model_config) and os.path.exists(yield_prediction_paths.model_config) and os.path.exists(gdh_prediction_paths.model_config):
            self.configs_found = True
            self.fruitload_config = FruitloadPredictionConfig()
            self.fruitload_config.from_path(fruitload_paths.model_config)
            self.yield_prediction_config = YieldPredictionConfig()
            self.yield_prediction_config.from_path(yield_prediction_paths.model_config)
            self.gdh_prediction_config = GdhPredictionConfig()
            self.gdh_prediction_config.from_path(gdh_prediction_paths.model_config)
            self.configs_found = True

