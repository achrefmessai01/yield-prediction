
from influxdb_client import InfluxDBClient

import os
from dotenv import load_dotenv

import datetime

from supabase_interface import SupaBaseInterface


class InfluxDBInterface:

    supabase_interface: SupaBaseInterface

    url: str = ""
    org: str = ""
    token: str = ""



    def __init__(self, url: str, org: str, token: str, supabase_interface: SupaBaseInterface) -> None:
        self.supabase_interface = supabase_interface

        self.url = url
        self.org = org
        self.token = token


    def get_strawberry_count_data(self, tenant: str, batch: str, start_date: datetime.datetime = None):

        # we will get the tenant name like it is setup in the SupaBase and therefore the frontend
        # but we will need it the fitting bucket id
        tenant_bucket = self.supabase_interface.get_tenant_influx_bucket(tenant)
        
        # the greenhouse should also be correct
        selected_greenhouse = self.supabase_interface.get_greenhouse(tenant, batch)

        if start_date == None:
            # no date given means just all entries, so just use date long ago
            start_date = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
        else:
            # convert date to utc
            start_date = start_date.astimezone(datetime.timezone.utc)

        with InfluxDBClient(url=self.url, token=self.token, org=self.org) as client:
            query_api = client.query_api()
            query = f'''from(bucket: "{tenant_bucket}")
                |> range(start: {start_date.isoformat().replace("+00:00", "Z")})
                |> filter(fn: (r) => r._measurement == "vgreens_strawberry")'''
            tables = query_api.query(query, org="vgreens")

        # get all bounding boxes with their classes
        # dict will be per timestamp -> camera -> bounding box index -> bounding box data
        bounding_boxes = dict()

        for table in tables:
            for record in table.records:
                try:
                    field = record["_field"]
                    if not "boxes" in field and not "classes" in field and not "scores" in field:
                        continue

                    value = record["_value"]
                    device = record["device"]
                    greenhouse = record["greenhouse"]
                    camera = record["camera"]
                    timestamp = record["_time"]

                    '''
                    TODO: rows are not correct in the influx db, therefore we can not yet handle them here
                    Later when they are correct (the logic needs to go into the NodeRed) we can activate the following code
                    For now it is deactivated and handeled by the config manager

                    row_start, row_end = self.supabase_interface.get_rows_per_batch(tenant, batch)
                    row = record["row"] # this mapping is not correct right now which is why we will not use it in the later steps but rather use manual mapping in the config
                    # skip in case the rows do not fit the batch
                    if row > row_end or row < row_start:
                        continue
                    '''

                    # TODO: add back when real data is used, right now there is no match
                    # skip in case the greenhouse does not fit
                    #if not greenhouse == selected_greenhouse:
                    #    continue

                    time_key = timestamp.strftime("%Y_%m_%d_%H_%M_%S")
                    if not time_key in bounding_boxes.keys():
                        bounding_boxes[time_key] = dict()
                    camera_id = device+"_cam"+str(camera)
                    if not camera_id in bounding_boxes[time_key].keys():
                        bounding_boxes[time_key][camera_id] = dict()
                    if "boxes_" in field:
                        bb_parsed = field.replace("boxes_", "").split("_")
                        bb_index = int(bb_parsed[0])
                        bb_corner = int(bb_parsed[1])
                        if not bb_index in bounding_boxes[time_key][camera_id].keys():
                            bounding_boxes[time_key][camera_id][bb_index] = dict()
                            bounding_boxes[time_key][camera_id][bb_index]["corners"] = [-1,-1,-1,-1]
                        bounding_boxes[time_key][camera_id][bb_index]["corners"][bb_corner] = value
                    elif "classes_" in field:
                        bb_index = int(field.replace("classes_",""))
                        if not bb_index in bounding_boxes[time_key][camera_id].keys():
                            bounding_boxes[time_key][camera_id][bb_index] = dict()
                        bounding_boxes[time_key][camera_id][bb_index]["class"] = value
                    elif "scores_" in field:
                        bb_index = int(field.replace("scores_",""))
                        if not bb_index in bounding_boxes[time_key][camera_id].keys():
                            bounding_boxes[time_key][camera_id][bb_index] = dict()
                        bounding_boxes[time_key][camera_id][bb_index]["score"] = value
                except:
                    continue
        
        # now we go over the bounding boxes and check if they all have all the needed data, otherwise we need to remove that data
        invalid_timestamps = list()
        for timestamp in bounding_boxes.keys():
            invalid_cameras = list()
            for camera_id in bounding_boxes[timestamp].keys():
                invalid_bbs = list()
                for bb_index in bounding_boxes[timestamp][camera_id].keys():
                    if not "corners" in bounding_boxes[timestamp][camera_id][bb_index]:
                        invalid_bbs.append(bb_index)
                        continue
                    if -1 in bounding_boxes[timestamp][camera_id][bb_index]["corners"]:
                        invalid_bbs.append(bb_index)
                        continue
                    if not "class" in bounding_boxes[timestamp][camera_id][bb_index]:
                        invalid_bbs.append(bb_index)
                        continue
                    if not "score" in bounding_boxes[timestamp][camera_id][bb_index]:
                        invalid_bbs.append(bb_index)
                        continue
                for invalid_bb in invalid_bbs:
                    del bounding_boxes[timestamp][camera_id][invalid_bb]
                if len(bounding_boxes[timestamp][camera_id].keys()) == 0:
                    invalid_cameras.append(camera_id)
            for invalid_camera in invalid_cameras:
                del bounding_boxes[timestamp][invalid_camera]
            if len(bounding_boxes[timestamp].keys()) == 0:
                invalid_timestamps.append(timestamp)
        for invalid_timestamp in invalid_timestamps:
            del bounding_boxes[invalid_timestamp]

        return bounding_boxes


if __name__ == "__main__":
    url = "https://vgreens-influx.fly.dev"
    org = "vgreens"
    load_dotenv()
    token = os.environ.get("INFLUXDB_TOKEN")
    supabase_key = os.environ.get("SUPABASE_KEY")

    influx_interface = InfluxDBInterface(url, org, token, SupaBaseInterface("https://erkjktrengqtbkilhmpe.supabase.co", supabase_key))
    
    tenant = "Knaup"
    batch = "Strawberry - Lady Emma"
    bb_data = influx_interface.get_strawberry_count_data(tenant, batch)
    print(bb_data)

    


