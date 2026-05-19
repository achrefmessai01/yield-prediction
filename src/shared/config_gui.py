from nicegui import ui

from yieldprediction.config import YieldPredictionConfig
from yieldprediction.paths import YieldPredictionPaths

from fruitloadprediction.config import FruitloadPredictionConfig
from fruitloadprediction.paths import FruitloadPredictionPaths

# TODO: read from supabase later
tenant_list = ["tenant1", "tenant2"]
batch_dict = {
    "tenant1": ["batch1", "batch2"],
    "tenant2": ["batch1"],
}

yieldpredpath = None
fruitpredpath = None
yieldpredconfig = None
fruitpredconfig = None
selection_made = False

# defaults
default_tenant = tenant_list[0]
default_batches = list(batch_dict.get(default_tenant, []))
default_batch = default_batches[0] if default_batches else None

selected_tenant = default_tenant
selected_batch = default_batch

def on_tenant_change(e):
    selected_tenant = e.value
    batches = list(batch_dict.get(selected_tenant, []))  # force a fresh list object
    batch_select.options = batches
    batch_select.value = batches[0] if batches else None
    batch_select.update()

def on_batch_change(e):
    selected_batch = e.value
    yieldpredpath = YieldPredictionPaths(tenant=selected_tenant, batch=selected_batch)
    fruitpredpath = FruitloadPredictionPaths(tenant=selected_tenant, batch=selected_batch)
    yieldpredconfig = YieldPredictionConfig.from_yaml(yieldpredpath.model_config)
    fruitpredconfig = FruitloadPredictionConfig.from_yaml(fruitpredpath.model_config)
    

tenant_select = ui.select(
    tenant_list,
    label="Tenant",
    value=default_tenant,
    on_change=on_tenant_change,
)

batch_select = ui.select(
    default_batches,
    label="Batch",
    value=default_batch,
    on_change=on_batch_change,
)

ui.run()
