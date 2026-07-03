"""Live contract test for the NASA POWER hourly point endpoint.

Fetches through the raw-request layer, then asserts the fields, shape, and value
ranges the pipeline relies on — a renamed column, wrong shape, sentinel leak, or
units change fails here.
"""

from datetime import date

from gdhprediction.climate import ClimateDataRequest, ClimateGdhData, RequestData
from gdhprediction.utils import HOURS_PER_YEAR, NasaDataArray
from tests._base import ArrayTestCase
from tests.test_gdhprediction.live._support import (
    LAT,
    LON,
    LiveApiContractMixin,
    requires_live,
)

_END_YEAR = date.today().year - 1  # noqa: DTZ011 (.year only)
_START_YEAR = _END_YEAR - 1
_YEARS = _END_YEAR - _START_YEAR + 1  # only two years


@requires_live
class TestNasaPowerLive(LiveApiContractMixin, ArrayTestCase):

    @staticmethod
    def fetch() -> NasaDataArray:
        request_data = RequestData(
            LAT,
            LON,
            ClimateGdhData.PARAMETERS,
            _START_YEAR,
            _END_YEAR,
            ClimateGdhData.COMMUNITY,
            ClimateGdhData.FORMAT,
            ClimateGdhData.TIME_STANDARD,
        )
        return ClimateDataRequest(request_data).request_nasa_data()

    def test_shape_is_one_non_leap_year_by_year_count(self) -> None:
        self.assertEqual(self.data.shape, (HOURS_PER_YEAR,))
        # Each parameter column carries one value per requested year.
        self.assertEqual(self.data["T2M"].shape, (HOURS_PER_YEAR, _YEARS))
