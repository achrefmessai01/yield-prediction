"""Shared TestCase extensions for array-valued assertions."""

from unittest import TestCase

import numpy as np
from numpy import typing as npt


class ArrayTestCase(TestCase):
    """TestCase whose numpy array assertions surface as unittest failures."""

    def assertAllClose(
        self,
        actual: npt.ArrayLike,
        desired: npt.ArrayLike,
        *,
        atol: float = 0.0,
        rtol: float = 0.0,
        msg: str = "",
    ) -> None:
        # Defaults give an exact comparison; pass atol/rtol for approximate
        # checks. asarray lets a scalar desired broadcast against an array actual.
        # Re-raising as failureException keeps it a failure, not an error.
        try:
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(desired),
                rtol=rtol,
                atol=atol,
            )
        except AssertionError as err:
            raise self.failureException(msg or str(err)) from err

    def assertArrayEqual(
        self,
        actual: npt.ArrayLike,
        desired: npt.ArrayLike,
        *,
        msg: str = "",
    ) -> None:
        try:
            np.testing.assert_array_equal(actual, desired)
        except AssertionError as err:
            raise self.failureException(msg or str(err)) from err
