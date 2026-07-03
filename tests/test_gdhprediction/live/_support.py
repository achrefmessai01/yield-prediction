"""Scaffolding shared by the live contract tests."""

import os
from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar
from unittest import SkipTest, skipUnless

import numpy as np
from requests import ConnectionError as RequestsConnectionError
from requests import Timeout

from gdhprediction.request import RequestApiError
from gdhprediction.utils import StructuredDataArray

# The shared assertions live in a mixin. At runtime it inherits `object` so
# unittest discovery — which collects every TestCase subclass it imports —
# ignores it; type-checkers see the real base so `self.assert*` resolves.
# Concrete cases mix it in alongside ArrayTestCase.
if TYPE_CHECKING:
    from tests._base import ArrayTestCase

    _ContractBase = ArrayTestCase
else:
    _ContractBase = object

_ENV_FLAG = "RUN_LIVE_TESTS"
requires_live = skipUnless(
    os.getenv(_ENV_FLAG),
    f"live API test; set {_ENV_FLAG}=1 to run",
)

# Berlin — inside DWD coverage (BrightSky is DWD-backed, Germany-centric).
LAT, LON = 52.52, 13.41
T2M_MIN, T2M_MAX = -90.0, 60.0  # °C, bracketing recorded surface extremes
RH_MIN, RH_MAX = 0.0, 100.0  # %
RADIATION_MIN = 0.0  # W/m^2, downwelling shortwave is non-negative


def fetch_or_skip[T](fetch: Callable[[], T]) -> T:
    """Run a live fetch, turning a genuine network outage into a skip.

    The session already retries transient 429/5xx and connect errors
    (`DataRequest._RETRY`). A `RequestApiError` still caused by a connection
    or timeout error is therefore a test-environment problem (offline runner,
    DNS) and is skipped; any other failure — a persistent HTTP error, a schema
    or shape mismatch — is a real contract break and propagates as a failure.
    Raising SkipTest works from both a test method and setUpClass.
    """
    try:
        return fetch()
    except RequestApiError as err:
        if isinstance(err.__cause__, (RequestsConnectionError, Timeout)):
            msg = f"network unavailable: {err}"
            raise SkipTest(msg) from err
        raise


class LiveApiContractMixin(_ContractBase, metaclass=ABCMeta):
    """Assertions shared by every live endpoint: fields and value ranges.

    Subclasses supply :meth:`fetch` for one endpoint; the array is fetched once
    per class and reused. NaN tolerance covers NASA's -999 sentinels (already NaN'd).
    """

    EXPECTED_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {"MO", "DY", "HR", "T2M", "RH2M", "ALLSKY_SFC_SW_DWN"},
    )
    data: ClassVar[StructuredDataArray]

    @staticmethod
    @abstractmethod
    def fetch() -> StructuredDataArray:
        """Hit the endpoint through its raw-request layer and return the array."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = fetch_or_skip(cls.fetch)

    def test_expected_fields_present(self) -> None:
        missing = self.EXPECTED_FIELDS - set(self.data.dtype.names or ())
        self.assertFalse(missing, msg=f"missing fields: {missing}")

    def test_values_are_physically_sane(self) -> None:
        for field, lo, hi in (
            ("T2M", T2M_MIN, T2M_MAX),
            ("RH2M", RH_MIN, RH_MAX),
            ("ALLSKY_SFC_SW_DWN", RADIATION_MIN, None),
        ):
            with self.subTest(field=field):
                col = self.data[field]
                self.assertFalse(
                    np.isnan(col).all(),
                    msg=f"{field} is entirely NaN — no real data returned",
                )
                self.assertGreaterEqual(float(np.nanmin(col)), lo)
                if hi is not None:
                    self.assertLessEqual(float(np.nanmax(col)), hi)
