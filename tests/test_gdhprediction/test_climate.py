from typing import ClassVar
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from gdhprediction.climate import CacheManager, RequestData
from gdhprediction.config import APP_PATHS, GdhPredictionPaths


# ruff: noqa: SLF001


class TestRequestData(TestCase):

    def test_init(self):
        rd = RequestData(12.345, 67.890, ("T2M",), 2012, 2027, "ag", "csv", "lst")
        self.assertEqual(rd.latitude, 12.5)
        self.assertEqual(rd.longitude, 68.125)
        self.assertEqual(rd.start, "20120101")
        self.assertEqual(rd.end, "20251231")


class TestCaching(TestCase):

    RAW_HASHKEY: ClassVar[str] = "test_raw_hashkey"
    CFG_HASHKEY: ClassVar[str] = "config_hashkey"
    PROCESSED_HASHKEY: ClassVar[str] = f"{RAW_HASHKEY}_{CFG_HASHKEY}"
    RAW_ARRAY = np.array([[1, 2], [3, 4]])
    PROCESSED_ARRAY = np.array([[5, 6], [7, 8]])
    TEST_PATHS: ClassVar[GdhPredictionPaths] = APP_PATHS

    CACHE_MANAGER: ClassVar[CacheManager]

    @classmethod
    def setUpClass(cls) -> None:
        cls.TEST_PATHS.climate_data(cls.RAW_HASHKEY).unlink(missing_ok=True)
        cls.TEST_PATHS.climate_data_checksum(cls.RAW_HASHKEY).unlink(missing_ok=True)
        # Mock the hash computation to return a predictable processed hashkey
        with patch.object(CacheManager, "__post_init__", autospec=True) as _p:
            _p.side_effect = lambda self, _: object.__setattr__(
                self, "processed_hash", cls.PROCESSED_HASHKEY
            )
            cls.CACHE_MANAGER = CacheManager(
                cls.RAW_HASHKEY, cls.CFG_HASHKEY, cls.TEST_PATHS
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.TEST_PATHS.climate_data(cls.RAW_HASHKEY).unlink(missing_ok=True)
        cls.TEST_PATHS.climate_data_checksum(cls.RAW_HASHKEY).unlink(missing_ok=True)

    def test_combined_hashkey(self) -> None:
        actual = self.CACHE_MANAGER.processed_hash
        self.assertEqual(actual, self.PROCESSED_HASHKEY)

    def test_cache_climate_data(self):
        # write dummy data to cache
        self.CACHE_MANAGER.write_to_cache(self.RAW_ARRAY, "raw")
        self.CACHE_MANAGER.write_to_cache(self.PROCESSED_ARRAY, "processed")
        # check if files are created
        self.assertTrue(self.TEST_PATHS.climate_data(self.RAW_HASHKEY).exists())
        self.assertTrue(
            self.TEST_PATHS.climate_data_checksum(self.RAW_HASHKEY).exists()
        )
        self.assertTrue(self.TEST_PATHS.climate_data(self.PROCESSED_HASHKEY).exists())
        self.assertTrue(
            self.TEST_PATHS.climate_data_checksum(self.PROCESSED_HASHKEY).exists()
        )
        # check integrity
        self.assertTrue(self.CACHE_MANAGER._check_cache(self.RAW_HASHKEY))
        self.assertTrue(self.CACHE_MANAGER._check_cache(self.PROCESSED_HASHKEY))
        # read back the data and check if it matches
        cached_raw = self.CACHE_MANAGER.load_from_cache("raw")
        cached_processed = self.CACHE_MANAGER.load_from_cache("processed")
        self.assertTrue(np.isclose(cached_raw, self.RAW_ARRAY).all())
        self.assertTrue(np.isclose(cached_processed, self.PROCESSED_ARRAY).all())
        # check corruption detection by modifying the data file
        with self.TEST_PATHS.climate_data(self.RAW_HASHKEY).open("ab") as f:
            f.write(b"corruption")
        self.assertFalse(self.CACHE_MANAGER._check_cache(self.RAW_HASHKEY))
