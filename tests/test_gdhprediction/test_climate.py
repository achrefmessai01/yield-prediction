from datetime import date
from typing import ClassVar, cast
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from gdhprediction.climate import ClimateGdhData, RequestData
from gdhprediction.request import CacheManager
from gdhprediction.utils import HOURS_PER_YEAR, PERCENTILES, HourlyAnnualizedGdhDType
from shared.configs import GdhPredictionConfig
from shared.paths import APP_PATHS, GdhPredictionPaths

# ruff: noqa: SLF001


def _rd(
    lat: float = 0.0,
    lon: float = 0.0,
    start_year: int = 2012,
    end_year: int = 2020,
) -> RequestData:
    """Build a RequestData with only the inputs under test exposed."""
    return RequestData(
        lat,
        lon,
        ("T2M",),
        start_year,
        end_year,
        "ag",
        "csv",
        "lst",
    )


class TestRequestData(TestCase):

    def test_init(self):
        rd = RequestData(12.345, 67.890, ("T2M",), 2012, 2127, "ag", "csv", "lst")
        self.assertEqual(rd.latitude, 12.5)
        self.assertEqual(rd.longitude, 68.125)
        self.assertEqual(rd.start, "20120101")
        self.assertEqual(rd.end, "20251231")

    # --- _snap_coordinates: rounds to the NASA POWER grid (0.5° / 0.625°) ---

    def test_snap_coordinates_northern_eastern(self) -> None:
        rd = _rd(lat=12.345, lon=67.890)
        self.assertEqual(rd.latitude, 12.5)
        self.assertEqual(rd.longitude, 68.125)

    def test_snap_coordinates_southern_western(self) -> None:
        # -33.86 / 0.5 = -67.72 → round to -68 → -34.0
        # -57.43 / 0.625 = -91.888 → round to -92 → -57.5
        rd = _rd(lat=-33.86, lon=-57.43)
        self.assertEqual(rd.latitude, -34.0)
        self.assertEqual(rd.longitude, -57.5)

    def test_snap_coordinates_already_on_grid_is_idempotent(self) -> None:
        rd = _rd(lat=12.5, lon=68.125)
        self.assertEqual(rd.latitude, 12.5)
        self.assertEqual(rd.longitude, 68.125)

    def test_snap_coordinates_zero_stays_zero(self) -> None:
        rd = _rd(lat=0.0, lon=0.0)
        self.assertEqual(rd.latitude, 0.0)
        self.assertEqual(rd.longitude, 0.0)

    # --- _valid_years: clamps to [FIRST_YEAR, date.today().year - 1] ---

    def test_valid_years_start_after_end_resets_start_to_first_year(self) -> None:
        # start > end => start = FIRST_YEAR, end unchanged
        rd = _rd(start_year=2020, end_year=2010)
        self.assertEqual(rd.start, "20010101")
        self.assertEqual(rd.end, "20101231")

    def test_valid_years_start_below_first_year_clamps(self) -> None:
        rd = _rd(start_year=1990, end_year=2010)
        self.assertEqual(rd.start, "20010101")
        self.assertEqual(rd.end, "20101231")

    @patch("gdhprediction.climate.date")
    def test_valid_years_end_above_today_clamps(self, date_mock: Mock) -> None:
        date_mock.today.return_value = date(2030, 7, 1)
        rd = _rd(start_year=2012, end_year=2200)
        self.assertEqual(rd.end, "20291231")

    def test_valid_years_inside_range_passes(self) -> None:
        rd = _rd(start_year=2010, end_year=2020)
        self.assertEqual(rd.start, "20100101")
        self.assertEqual(rd.end, "20201231")


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
        cls.TEST_PATHS.data_file(cls.RAW_HASHKEY).unlink(missing_ok=True)
        cls.TEST_PATHS.data_file_checksum(cls.RAW_HASHKEY).unlink(missing_ok=True)
        # Mock the hash computation to return a predictable processed hashkey
        with patch.object(CacheManager, "__post_init__", autospec=True) as _p:
            _p.side_effect = lambda self, _: object.__setattr__(
                self,
                "processed_hash",
                cls.PROCESSED_HASHKEY,
            )
            cls.CACHE_MANAGER = CacheManager(
                cls.RAW_HASHKEY,
                cls.CFG_HASHKEY,
                cls.TEST_PATHS,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.TEST_PATHS.data_file(cls.RAW_HASHKEY).unlink(missing_ok=True)
        cls.TEST_PATHS.data_file_checksum(cls.RAW_HASHKEY).unlink(missing_ok=True)

    def test_combined_hashkey(self) -> None:
        actual = self.CACHE_MANAGER.processed_hash
        self.assertEqual(actual, self.PROCESSED_HASHKEY)

    def test_cache_climate_data(self):
        # write dummy data to cache
        self.CACHE_MANAGER.write_to_cache(self.RAW_ARRAY, "raw")
        self.CACHE_MANAGER.write_to_cache(self.PROCESSED_ARRAY, "processed")
        # check if files are created
        self.assertTrue(self.TEST_PATHS.data_file(self.RAW_HASHKEY).exists())
        self.assertTrue(self.TEST_PATHS.data_file_checksum(self.RAW_HASHKEY).exists())
        self.assertTrue(self.TEST_PATHS.data_file(self.PROCESSED_HASHKEY).exists())
        self.assertTrue(
            self.TEST_PATHS.data_file_checksum(self.PROCESSED_HASHKEY).exists(),
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
        with self.TEST_PATHS.data_file(self.RAW_HASHKEY).open("ab") as f:
            f.write(b"corruption")
        self.assertFalse(self.CACHE_MANAGER._check_cache(self.RAW_HASHKEY))


def _synthetic_climate_array() -> np.ndarray:
    """8760-row climate array with CUMGDH_pXX[i] = i**2 across all percentiles.

    The square gives each starting index a distinctive forecast trajectory, so
    a test can recover today_hour from the second element of the result:
        result[1] = (today_hour + 1)**2 - today_hour**2 = 2 * today_hour + 1
    """
    arr = np.empty(HOURS_PER_YEAR, dtype=HourlyAnnualizedGdhDType)
    indices = np.arange(HOURS_PER_YEAR, dtype=np.float32)
    squared = (indices * indices).astype(np.float32)
    for q in PERCENTILES:
        arr[f"CUMGDH_p{q:02d}"] = squared
    return arr


def _make_cgd() -> ClimateGdhData:
    """Construct a ClimateGdhData wired up with a synthetic in-memory array."""
    request_data = RequestData(52.52, 13.41, ("T2M",), 2010, 2020, "ag", "csv", "lst")
    cgd = ClimateGdhData(request_data, GdhPredictionConfig())
    cgd.data_array = _synthetic_climate_array()
    return cgd


class TestClimateForecastGdhWindow(TestCase):
    """Test :meth:`ClimateGdhData.forecast_gdh`.

    forecast_gdh slices into the 8760-row, non-leap climate array using
    today_hour = (tm_yday - 1) * 24, with two non-trivial behaviors:
    #. The result is re-anchored to zero at the start hour.
    #. In leap years, today_hour is shifted back by 24h to compensate Feb 29.
    """

    def test_mid_year_non_leap_anchors_to_zero(self) -> None:
        # date(2023, 3, 1): tm_yday=60, today_hour = 59*24 = 1416.
        cgd = _make_cgd()
        result = cgd.forecast_gdh(start_date=date(2023, 3, 1))
        self.assertEqual(result.shape, (28 * 24,))
        self.assertAlmostEqual(float(result["GDH_p50"][0]), 0.0, places=4)
        # result[1] = 2*today_hour + 1 = 2833 pins today_hour to 1416.
        self.assertAlmostEqual(float(result["GDH_p50"][1]), 2833.0, places=4)
        # result[k] = (today_hour+k)^2 - today_hour^2 for the contiguous case.
        # k=10 → (1426)^2 - (1416)^2 = 28420.
        self.assertAlmostEqual(float(result["GDH_p50"][10]), 28420.0, places=4)

    def test_leap_year_after_feb_subtracts_one_day(self) -> None:
        # date(2024, 3, 1): tm_yday=61, today_hour = 60*24 = 1440, then -=24.
        # After correction it should resolve to the same array index (1416)
        # as Mar 1 in a non-leap year, so result[1] matches the non-leap test.
        cgd = _make_cgd()
        result = cgd.forecast_gdh(start_date=date(2024, 3, 1))
        self.assertEqual(result.shape, (28 * 24,))
        self.assertAlmostEqual(float(result["GDH_p50"][0]), 0.0, places=4)
        self.assertAlmostEqual(float(result["GDH_p50"][1]), 2833.0, places=4)

    def test_leap_year_feb_is_not_shifted(self) -> None:
        # date(2024, 2, 28): tm_yday=59, today_hour = 58*24 = 1392.
        # The correction is gated on month > FEB, so Feb 28 is untouched.
        cgd = _make_cgd()
        result = cgd.forecast_gdh(start_date=date(2024, 2, 28))
        self.assertEqual(result.shape, (28 * 24,))
        self.assertAlmostEqual(float(result["GDH_p50"][1]), 2785.0, places=4)

    def test_year_wrap_concatenates_correctly(self) -> None:
        # date(2023, 12, 20): tm_yday=354, today_hour=8472. forecast=28 days
        # crosses Dec 31 → Jan 1, so the slice wraps:
        # forecast_cumgdh = data[8472:] (288 rows) + data[:384] (384 rows).
        cgd = _make_cgd()
        result = cgd.forecast_gdh(start_date=date(2023, 12, 20))
        self.assertEqual(result.shape, (28 * 24,))
        # Result is still anchored at zero at the start.
        self.assertAlmostEqual(float(result["GDH_p50"][0]), 0.0, places=4)
        # Element differences pin both ends of the concatenation. In a
        # consecutive difference the anchor cancels, so each step should
        # match the same step in the underlying source array.
        data_array = cast("np.ndarray", cgd.data_array)
        src = data_array["CUMGDH_p50"]
        p50 = result["GDH_p50"]
        # Pre-wrap step (286→287) mirrors source step 8758→8759, confirming
        # the first chunk is a contiguous slice from today_hour onward.
        self.assertAlmostEqual(
            float(p50[287] - p50[286]),
            float(src[8759] - src[8758]),  # ≈ 17517
            delta=10.0,
        )
        # The wrap (287→288) is data[0] - data[8759] — a ~-76M jump that
        # only a Dec 31 → Jan 1 concatenation can produce. A misindexed
        # contiguous slice would give a small positive step here instead.
        self.assertAlmostEqual(
            float(p50[288] - p50[287]),
            float(src[0] - src[8759]),  # ≈ -76,725,081
            delta=10.0,
        )
        # Last element pins the right end of the post-wrap chunk to
        # data[last_hour - 1] = data[383], confirming the slice stops
        # exactly at last_hour = 384.
        self.assertAlmostEqual(
            float(p50[-1]),
            float(src[383] - src[8472]),
            delta=10.0,
        )

    def test_renamed_columns(self) -> None:
        # forecast_gdh strips CUMGDH_ and exposes GDH_pXX
        cgd = _make_cgd()
        result = cgd.forecast_gdh(start_date=date(2023, 6, 15), forecast=7)
        expected = tuple(f"GDH_p{q:02d}" for q in PERCENTILES)
        self.assertEqual(result.dtype.names, expected)

    def test_does_not_mutate_source_array(self) -> None:
        # forecast_gdh re-anchors via in-place subtraction on a .copy()
        cgd = _make_cgd()
        data_array = cast("np.ndarray", cgd.data_array)
        before = data_array["CUMGDH_p50"].copy()
        cgd.forecast_gdh(start_date=date(2023, 6, 15), forecast=14)
        np.testing.assert_array_equal(data_array["CUMGDH_p50"], before)
