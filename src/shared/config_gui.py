from nicegui import ui

import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from datetime import datetime

from dataclasses import fields, is_dataclass
from typing import Callable, Type, Optional, Sequence


from shared.configs import FruitloadPredictionConfig, GdhPredictionConfig, YieldPredictionConfig
from shared.config_handling import ConfigHandling
from shared.paths import FruitloadPredictionPaths, GdhPredictionPaths, YieldPredictionPaths

from yieldprediction.yield_prediction import YieldPrediction

from shared.supabase_interface import SupaBaseInterface
from shared.influxdb_interface import InfluxDBInterface

'''
Data part --------------------------
'''
load_dotenv()
token = os.environ.get("INFLUXDB_TOKEN")
supabase_key = os.environ.get("SUPABASE_KEY")

supabase_interface = SupaBaseInterface("https://erkjktrengqtbkilhmpe.supabase.co", supabase_key)
influx_interface = InfluxDBInterface("https://vgreens-influx.fly.dev", "vgreens", token, supabase_interface)

coco_configs_path = "/home/tom/workspace_code/vGreens/yield_pred_coco_database"

batch_dict = supabase_interface.get_all_tenant_with_batches()
tenant_list = list(batch_dict.keys())

configs = None

'''
general functions ----------------------------
'''

def load_configs():
    global selected_tenant, selected_batch
    global configs

    configs = ConfigHandling(selected_tenant, selected_batch)
    try:
        configs.load_from_files()
    except:
        configs.create_default_configs()
    refresh_all_renders()

def update_gdh_percent_stage_to_stage():
    global configs
    configs.yield_prediction_config.gdh_percent_stage_to_stage = {stage : 1/(len(configs.fruitload_config.growing_stages_order)-1) for stage in configs.fruitload_config.growing_stages_order[:-1]}

'''
GUI data ----------------------------
'''
selected_tenant = tenant_list[0]
selected_batch = batch_dict[selected_tenant][0]

selected_datetime = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)

predicted_counts = None
predicted_weights = None



gdh_percent_stage_to_stage_gui_renderer = None
predicted_plot_counts_renderer = None
predicted_plot_weight_renderer = None
visible_plants_per_camera_renderer = None
camera_row_map_renderer = None
ripeness_stage_order_renderer = None

batch_select = None

'''
GUI functions ----------------------------
'''
def refresh_all_renders():
    global gdh_percent_stage_to_stage_gui_renderer, predicted_plot_counts_renderer, predicted_plot_weight_renderer, visible_plants_per_camera_renderer
    global camera_row_map_renderer, ripeness_stage_order_renderer
    if not gdh_percent_stage_to_stage_gui_renderer is None:
        gdh_percent_stage_to_stage_gui_renderer.refresh()
    if not predicted_plot_counts_renderer is None:
        predicted_plot_counts_renderer.refresh()
    if not predicted_plot_weight_renderer is None:
        predicted_plot_weight_renderer.refresh()
    if not visible_plants_per_camera_renderer is None:
        visible_plants_per_camera_renderer.refresh()
    if not camera_row_map_renderer is None:
        camera_row_map_renderer.refresh()
    if not ripeness_stage_order_renderer is None:
        ripeness_stage_order_renderer.refresh()

def update_camera_after_removing():
    global configs
    global camera_row_map_renderer, visible_plants_per_camera_renderer
    
    # check which one is larger
    if len(configs.fruitload_config.visible_plants_per_camera.keys()) > len(configs.fruitload_config.camera_row_map.keys()):
        configs.fruitload_config.visible_plants_per_camera = {k:v for k,v in configs.fruitload_config.visible_plants_per_camera.items() if k in configs.fruitload_config.camera_row_map.keys()}
        visible_plants_per_camera_renderer.refresh()
    elif len(configs.fruitload_config.visible_plants_per_camera.keys()) < len(configs.fruitload_config.camera_row_map.keys()):
        configs.fruitload_config.camera_row_map = {k:v for k,v in configs.fruitload_config.camera_row_map.items() if k in configs.fruitload_config.visible_plants_per_camera.keys()}
        camera_row_map_renderer.refresh()
    else:
        pass

def on_tenant_change(e):
    global selected_tenant, selected_batch, batch_select
    selected_tenant = e.value
    batch_select.options = batch_dict[selected_tenant]
    batch_select.value = batch_dict[selected_tenant][0]
    selected_batch = batch_dict[selected_tenant][0]
    refresh_all_renders()

def on_batch_change(e):
    global selected_batch
    selected_batch = e.value
    # on selection of tenant/batch we need to create the configs
    load_configs()

def load_data_from_dbs():
    global configs, influx_interface, supabase_interface, coco_configs_path
    configs.load_from_database(influx_interface, supabase_interface, coco_configs_path, read_from_files=True)

    refresh_all_renders()

def apply():
    # apply will run the prediction and show the results in a plot
    global configs
    global selected_tenant, selected_batch
    global predicted_counts, predicted_weights
    global predicted_plot_counts_renderer, predicted_plot_weight_renderer
    global selected_datetime, influx_interface, supabase_interface

    # store configs locally
    configs.yield_prediction_config.write_config(configs.yield_prediction_paths.model_config)
    configs.fruitload_config.write_config(configs.fruitload_paths.model_config)
    configs.gdh_prediction_config.write_config(configs.gdh_prediction_paths.model_config)
    
    cur_yield_prediction = YieldPrediction(selected_tenant, selected_batch, selected_datetime, supabase_interface, influx_interface)
    try:
        cur_yield_prediction = YieldPrediction(selected_tenant, selected_batch, selected_datetime, supabase_interface, influx_interface)
    except Exception as e:
        with ui.dialog() as error_dialog, ui.card():
            error_label = ui.label('There was an error while running the yield prediction:\n'+str(e))
            ui.button('Close', on_click=error_dialog.close)
            error_dialog.open()
        return
    
    # here the config should bne clean and we can run the prediction
    predicted_counts, predicted_weights = cur_yield_prediction.get_prediction()

    refresh_all_renders()



def reorderable_list(items: list[str], label: str = 'Items', allow_adding: bool = False, allow_removing = False):
    @ui.refreshable
    def render():
        ui.label(label).classes('text-lg font-semibold mb-2')

        for i, item in enumerate(items):
            with ui.row().classes('items-center gap-2 w-full'):

                # editable item value
                inp = ui.input(value=item).classes('flex-1')
                inp.on('update:model-value',
                       lambda e, i=i: items.__setitem__(i, e.value))

                # move buttons
                ui.button(icon='arrow_upward',
                          on_click=lambda i=i: move_up(i)).props('flat')

                ui.button(icon='arrow_downward',
                          on_click=lambda i=i: move_down(i)).props('flat')

                if allow_removing:
                    # remove button
                    ui.button(icon='delete',
                            on_click=lambda i=i: remove(i)).props('flat color=red')

        # add new item
        if allow_adding:
            with ui.row().classes('gap-2 mt-4'):
                new_item_input = ui.input(placeholder='New item').classes('flex-1')
                ui.button('Add', on_click=lambda: add(new_item_input))
        

        def move_up(i):
            if i > 0:
                items[i-1], items[i] = items[i], items[i-1]
                update_gdh_percent_stage_to_stage()
                render.refresh()

        def move_down(i):
            if i < len(items) - 1:
                items[i+1], items[i] = items[i], items[i+1]
                update_gdh_percent_stage_to_stage()
                render.refresh()

        def remove(i):
            items.pop(i)
            update_gdh_percent_stage_to_stage()
            render.refresh()

        def add(input_field):
            value = input_field.value
            if value:
                items.append(value)
                input_field.value = ''
                update_gdh_percent_stage_to_stage()
                render.refresh()

    render()
    return render


def validate_positive(value):
    if value is None or value <= 0:
        return 'Must be > 0'
    return None

def editable_number_dict(
    data_getter,
    label_fn,
    *,
    cast=float,
    step=1.0,
    min_value=None,
    validation=None,
    allow_removing=False,
):

    @ui.refreshable
    def render():

        data = data_getter()

        def safe_validation(v):
            try:
                val = cast(v)
            except:
                return 'Must be a number'

            if min_value is not None and val < min_value:
                return f"Must be ≥ {min_value}"

            if validation:
                return validation(val)

            return None

        def handler(e, key):
            try:
                val = cast(e.args)
            except:
                return

            if min_value is not None and val < min_value:
                val = cast(min_value)

            if data[key] != val:
                data[key] = val
                render.refresh()

        def remove_item(key):
            if key in data:
                del data[key]
                render.refresh()
            update_camera_after_removing()

        for i, (key, value) in enumerate(list(data.items())):
            with ui.row().classes('items-center gap-2 w-full'):

                ui.number(
                    label=label_fn(key, value, i),
                    value=cast(value),
                    validation=safe_validation,
                ).style('flex: 1').props(
                    f"step={step}"
                ).on(
                    'update:model-value',
                    lambda e, key=key: handler(e, key)
                )

                if allow_removing:
                    ui.button(
                        icon='delete',
                        on_click=lambda key=key: remove_item(key),
                    ).props('flat color=red')

    render()
    return render


def make_predicted_counts_plot_renderer(
    data_getter: Callable[[], Optional[Sequence[float]]],
    *,
    title: str = 'Predicted Counts',
    x_label: str = 'Time step',
    y_label: str = 'Count',
    figsize=(8, 4),
):
    @ui.refreshable
    def render():
        data = data_getter()

        if data is None:
            ui.label('No data yet. Click "Apply" to run the prediction.').classes('text-gray-500')
            return

        y = np.asarray(data).reshape(-1)
        if y.size == 0:
            ui.label('No data to plot (empty array).').classes('text-gray-500')
            return

        x = np.arange(len(y))

        with ui.pyplot(figsize=figsize):
            plt.plot(x, y, linewidth=2)
            plt.title(title)
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)

    render()
    return render


def on_date_change(e):
    global selected_datetime
    if e.value:
        # replace the whole datetime with the chosen date at 00:00:00
        selected_datetime = datetime.strptime(e.value, '%Y-%m-%d')



'''
GUI implementation ----------------------------
'''

load_configs() # init configs on startup

# select tenant and batch
with ui.column().classes('w-full gap-4 p-4'):
    with ui.card().classes('w-full'):
        ui.label('Selection').classes('text-lg font-semibold mb-2')

        with ui.row().classes('gap-4'):
            # tanant selection
            ui.select(
                tenant_list,
                label="Tenant",
                value=selected_tenant,
                on_change=on_tenant_change,
            ).classes('w-64')

            # batch selection
            batch_select = ui.select(
                batch_dict[selected_tenant],
                label="Batch",
                value=selected_batch,
                on_change=on_batch_change,
            ).classes('w-64')


            ui.date_input(
                'Date',
                value=selected_datetime.strftime('%Y-%m-%d'),
                on_change=on_date_change,
            )
            # load data from database button
            ui.button(
                "Load Data from DBs",
                on_click=load_data_from_dbs
            ).classes('w-64')

            # apply button
            ui.button(
                "Apply",
                on_click=apply
            ).classes('w-64')



# 
with ui.row().classes('w-full gap-4 items-stretch'):
    # LEFT SIDE for configs
    with ui.column().classes('flex-1 gap-4'):
        ui.label('Yield Prediction Config')
        with ui.card().classes('w-full min-h-[200px]'):
            with ui.row().classes('w-full gap-4 items-stretch'):
                with ui.column().classes('flex-1 gap-4 items-stretch'):
                    with ui.card().classes('flex-1 gap-4 items-stretch'):
                        ui.label('General settings:')

                        ui.number(
                            label="Days of prediction",
                            value=configs.yield_prediction_config.days_of_prediction,
                            validation=validate_positive
                        ).props('step=1').bind_value(
                                    configs.yield_prediction_config, 'days_of_prediction',
                                    forward=lambda v: int(v) if v is not None else 0,
                                    backward=lambda v: int(v) if v is not None else 0,
                                )
                        
                        ui.number(
                            label="Latitude",
                            value=configs.yield_prediction_config.latitude
                        ).bind_value(configs.yield_prediction_config, 'latitude')

                        ui.number(
                            label="Longitude",
                            value=configs.yield_prediction_config.longitude
                        ).bind_value(configs.yield_prediction_config, 'longitude')
                        
                        ui.number(
                            label="Total needed GDH",
                            value=configs.yield_prediction_config.total_gdh_needed
                        ).props('step=1').bind_value(
                                    configs.yield_prediction_config, 'total_gdh_needed',
                                    forward=lambda v: int(v) if v is not None else 0,
                                    backward=lambda v: int(v) if v is not None else 0,
                                )

                        ui.number(
                            label="Average weight per harvested fruit (g)",
                            value=configs.yield_prediction_config.avg_weight
                        ).bind_value(configs.yield_prediction_config, 'avg_weight')

                    with ui.card().classes('flex-1 gap-4 items-stretch'):
                        ui.label('Plants per Camera:')
                        visible_plants_per_camera_renderer = editable_number_dict(
                            lambda: configs.fruitload_config.visible_plants_per_camera,
                            label_fn=lambda camera, value, i:
                                f"Plants seen by camera {camera}",
                            cast=float,
                            step=1,
                            min_value=0,
                            validation=lambda v: 'Must be > 0' if v is None or v <= 0 else None,
                            allow_removing=True
                        )
                    with ui.card().classes('flex-1 gap-4 items-stretch'):
                        ui.label('Row per Camera:')
                        camera_row_map_renderer = editable_number_dict(
                            lambda: configs.fruitload_config.camera_row_map,
                            label_fn=lambda camera, value, i:
                                f"Rows seen by camera {camera}",
                            cast=int,
                            step=1,
                            min_value=0,
                            validation=lambda v: 'Must be > 0' if v is None or v <= 0 else None,
                            allow_removing=True
                        )

                    with ui.card().classes('flex-1 gap-4 items-stretch'):
                        ui.label('Count correction factor for each stage:')
                        count_correction_factor_renderer = editable_number_dict(
                            lambda: configs.fruitload_config.count_correction_factor,
                            label_fn=lambda stage, value, i:
                                f"Factor of stage {stage}",
                            cast=float,
                            step=0.1,
                            min_value=0,
                            validation=lambda v: 'Must be > 0' if v is None or v <= 0 else None,
                        )


                with ui.column().classes('flex-1 gap-4 items-stretch'):
                    with ui.card().classes('flex-1 gap-4 items-stretch'):
                        ui.label('Ripeness stage order:')
                        ripeness_stage_order_renderer = reorderable_list(configs.fruitload_config.growing_stages_order, allow_removing=True)

                    with ui.card().classes('flex-1 gap-4 items-stretch'):
                        ui.label('GDH Percent Stage to Stage:')
                        gdh_percent_stage_to_stage_gui_renderer = editable_number_dict(
                            lambda: configs.yield_prediction_config.gdh_percent_stage_to_stage,
                            label_fn=lambda stage, value, i:
                                f"{stage} -> {configs.fruitload_config.growing_stages_order[i+1]}",
                            cast=float,
                            step=0.1,
                            min_value=0,
                            validation=lambda v: 'Must be > 0' if v is None or v <= 0 else None,
                        )


        
        with ui.card().classes('w-full min-h-[200px]'):
            ui.label('GDH Prediction Config')

    # RIGHT SIDE for plots
    with ui.column().classes('flex-1 gap-4'):
        ui.label('Plots')            
        with ui.card().classes('w-full min-h-[600px]'):
            predicted_plot_counts_renderer = make_predicted_counts_plot_renderer(lambda: predicted_counts, title="Predicted ripe fruits", x_label="Days", y_label="Number ripe fruits")
            predicted_plot_weight_renderer = make_predicted_counts_plot_renderer(lambda: predicted_weights, title="Predicted weight", x_label="Days", y_label="Weight (g)")




ui.run()