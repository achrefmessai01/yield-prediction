"""Live contract test for the BrightSky/DWD weather endpoint.

Fetches through the raw-request layer, then asserts the fields, cadence, and
value ranges the pipeline relies on — a missing field, wrong type, or units
change fails here.
"""

from gdhprediction.utils import BrightSkyDataArray
from gdhprediction.weather import BrightskyDataRequest, WeatherRequestData
from tests._base import ArrayTestCase
from tests.test_gdhprediction.live._support import (
    LAT,
    LON,
    LiveApiContractMixin,
    requires_live,
)

_DURATION = 2  # days
_MIN_RECORDS = 24  # at least a full day of hourly records


@requires_live
class TestBrightSkyLive(LiveApiContractMixin, ArrayTestCase):

    @staticmethod
    def fetch() -> BrightSkyDataArray:
        # start_datetime=None → now, in the GPS-derived tz.
        request_data = WeatherRequestData(LAT, LON, None, _DURATION)
        return BrightskyDataRequest(request_data).request_brightsky_data()

    def test_returns_hourly_records_spanning_the_window(self) -> None:
        # Tolerate snapping and DST shift by a few hours; insist on real data.
        self.assertGreaterEqual(len(self.data), _MIN_RECORDS)
        self.assertLessEqual(len(self.data), (_DURATION + 1) * 24)
