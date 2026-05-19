import argparse


from config_handling import ConfigHandling
from yieldprediction.yield_prediction import YieldPrediction


'''
Main entry point for the yield prediction
Call when there are new counts available to update the SupaBase
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='YieldPredcition',
                    description='Script to run the yield prediction')
    
    parser.add_argument('--tenant', type=str, required=True,
                        help='Tenant name for the configuration')
    parser.add_argument('--batch', type=str, required=True,
                        help='Batch name for the configuration')
    
    args = parser.parse_args()
    tenant = args.tenant
    batch = args.batch

    configs = ConfigHandling(tenant, batch)
    if not configs.configs_found:
        raise ValueError("At least one configuration file is missing")
    
    yp = YieldPrediction(tenant, batch)
    pred_count, pred_weight = yp.get_prediction(store_for_training=True)
    # TODO write into DB, for now just plot
    print(pred_weight)
    print(pred_count)
    import matplotlib.pyplot as plt
    plt.plot(pred_count)
    plt.show()