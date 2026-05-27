import os
from dotenv import load_dotenv
from supabase import create_client, Client

from typing import List, Dict
import datetime


class SupaBaseInterface:

    supabase: Client

    def __init__(self, url: str, key: str) -> None:
        self.supabase = create_client(url, key)

    def get_tetant_id(self, tenant: str) -> str:
        tenant_id = self.supabase.table("tenants").select("*").eq("name", tenant).execute().data
        if len(tenant_id) > 0:
            return tenant_id[0]["id"]
        else:
            raise ValueError(f"No id found for tenant {tenant}")

    def get_days_since_harvest(self, tenant: str, batch: str) -> List[int]:
        # to get the days sionce harvest for each row we need to get each harvest entry in the database for current tenant/batch that has the complete flag
        tenant_id = self.get_tetant_id(tenant)
        data = self.supabase.table("harvests").select("*").eq("tenantId", tenant_id).eq("batch", batch).eq("status", "completed").execute().data
        if len(data) == 0:
            raise ValueError(f"No harvest data found for tenant {tenant} and batch {batch}")
        # get all dates and the rows harvested to calc when was the last time a row war harvested
        # there is one issue: If there was no harvest yet, what value will we put? Best option might be to use the plant date
        # In the end this should not matter, the values are just use to weight the cameras counts
        # So the value just needs to be unique which it should be since there can not be a harvest before planting
        planting_date = data[0]["plantingDate"]
        date_format = "%Y-%m-%d"
        today = datetime.date.today()
        planting_date = datetime.datetime.strptime(planting_date, date_format).date()
        days_since_planting = (today - planting_date).days
        days_since_harvest = list()
        for d in range(data[0]["numberOfRows"]):
            days_since_harvest.append(days_since_planting)
        for a_harvest_entry in data:
            start_row = a_harvest_entry["rowX"]
            end_row = a_harvest_entry["rowY"]
            cur_harvest_date = datetime.datetime.strptime(a_harvest_entry["date"], date_format).date()
            cur_days_since_harvest = (today - cur_harvest_date).days
            for r in range(start_row, end_row+1):
                if days_since_harvest[r] > cur_days_since_harvest:
                    days_since_harvest[r] = cur_days_since_harvest
        return days_since_harvest

    def get_plants_per_row(self, tenant: str, batch: str) -> int:
        tenant_id = self.get_tetant_id(tenant)
        plants_per_meter = self.supabase.table("harvests").select("plantsPerMeter").eq("tenantId", tenant_id).eq("batch", batch).execute().data
        row_length = self.supabase.table("harvests").select("rowLength").eq("tenantId", tenant_id).eq("batch", batch).execute().data
        if len(plants_per_meter) > 0 and len(row_length) > 0:
            plants_per_meter = plants_per_meter[0]["plantsPerMeter"]#
            row_length = row_length[0]["rowLength"]
            return plants_per_meter*row_length
        else:
            raise ValueError(f"No data found for tenant {tenant} and batch {batch}")

    def get_total_number_rows(self, tenant: str, batch: str) -> int:
        tenant_id = self.get_tetant_id(tenant)
        num_rows = self.supabase.table("harvests").select("numberOfRows").eq("tenantId", tenant_id).eq("batch", batch).execute().data
        if len(num_rows) > 0:
            return num_rows[0]["numberOfRows"]
        else:
            raise ValueError(f"No data found for tenant {tenant} and batch {batch}")
        
    def get_tenant_list(self) -> List[str]:
        tenants = self.supabase.table("tenants").select("*").execute().data
        tenant_list = list()
        for tenant in tenants:
            tenant_list.append(tenant["name"])
        return tenant_list
    
    def get_tenant_influx_bucket(self, tenant: str) -> str:
        tenant_id = self.get_tetant_id(tenant)
        tenants = self.supabase.table("tenants").select("influxBucketName").eq("id", tenant_id).execute().data
        if len(tenants) > 0:
            return tenants[0]["influxBucketName"]
        else:
            raise ValueError(f"No data found for tenant {tenant}")
        
    
    def get_greenhouse(self, tenant: str, batch: str) -> str:
        tenant_id = self.get_tetant_id(tenant)
        greenhouse_list = self.supabase.table("harvests").select("greenhouse").eq("tenantId", tenant_id).eq("batch", batch).eq("status", "").execute().data
        if len(greenhouse_list) > 0:
            greenhouse = greenhouse_list[0]["greenhouse"]
            for g in greenhouse_list:
                if not g == greenhouse:
                    raise ValueError(f"Multiple greenhouses found for tenant {tenant} and batch {batch}")
            return greenhouse
        else:
            raise ValueError(f"No data found for tenant {tenant} and batch {batch}")

        
    def set_yield_prediction(self, tenant: str, batch: str, yield_prediction: List[List[float]]):
        pass #TODO create new table to set results
        

if __name__ == "__main__":

    url = "https://erkjktrengqtbkilhmpe.supabase.co"
    load_dotenv()
    key = os.environ.get("SUPABASE_KEY")

    supabase_interface = SupaBaseInterface(url, key)
    t = "Knaup"
    b = "awdwadaw - awdwad"
    #days_since_harvest = supabase_interface.get_days_since_harvest(t, b)
    #print(days_since_harvest)
    greenhouse = supabase_interface.get_greenhouse(t, b)
    print(greenhouse)
    