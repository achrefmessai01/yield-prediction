from typing import ClassVar
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from gdhprediction.climate import ClimateData, RequestData
from gdhprediction.paths import APP_PATHS, GdhPredictionPaths


class TestRequestData(TestCase):

    def test_init(self):
        rd = RequestData(12.345, 67.890, ("T2M",), 2012, 2027, "ag", "csv", "lst")
        self.assertEqual(rd.latitude, 12.5)
        self.assertEqual(rd.longitude, 68.125)
        self.assertEqual(rd.start, "20150101")
        self.assertEqual(rd.end, "20251231")


class TestCaching(TestCase):

    TEST_HASHKEY: ClassVar[str] = "testhashkey"
    TEST_PATHS: ClassVar[GdhPredictionPaths] = APP_PATHS

    @classmethod
    def setUpClass(cls) -> None:
        cls.TEST_PATHS.annual_data(cls.TEST_HASHKEY).unlink(missing_ok=True)
        cls.TEST_PATHS.annual_data_checksum(cls.TEST_HASHKEY).unlink(missing_ok=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.TEST_PATHS.annual_data(cls.TEST_HASHKEY).unlink(missing_ok=True)
        cls.TEST_PATHS.annual_data_checksum(cls.TEST_HASHKEY).unlink(missing_ok=True)

    def test_cache_annual_data(self):
        request_data_mock = Mock(spec=RequestData)
        # add mock fields to satisfy the constructor of ClimateData
        request_data_mock.start = "2015"
        request_data_mock.end = "2025"
        request_data_mock.parameters = "foo,bar"
        request_data_mock.compute_cache_hash = Mock(return_value=self.TEST_HASHKEY)
        climate_data = ClimateData(request_data_mock, self.TEST_PATHS)
        # create dummy data to cache
        dummy_data = np.array([[1, 2], [3, 4]])
        climate_data.data_array = dummy_data
        climate_data.write_to_cache()
        # check if files are created
        self.assertTrue(self.TEST_PATHS.annual_data(self.TEST_HASHKEY).exists())
        self.assertTrue(
            self.TEST_PATHS.annual_data_checksum(self.TEST_HASHKEY).exists()
        )
        # check integrity
        self.assertTrue(climate_data.check_uncorrupted_cache())
        # read back the data and check if it matches
        climate_data.data_array = None  # clear the data array
        climate_data.load_from_cache()
        self.assertTrue(np.isclose(climate_data.data_array, dummy_data).all())
        # check corruption detection by modifying the data file
        with self.TEST_PATHS.annual_data(self.TEST_HASHKEY).open("ab") as f:
            f.write(b"corruption")
        self.assertFalse(climate_data.check_uncorrupted_cache())
