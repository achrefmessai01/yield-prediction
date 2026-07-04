import math
from dataclasses import dataclass
from unittest import TestCase

import numpy as np

from gdhprediction.utils import FieldsHashMxn, round_sig, sunshine_to_radiation
from tests._base import ArrayTestCase


@dataclass(frozen=True, slots=True)
class _Toy(FieldsHashMxn):
    x: float
    y: int


@dataclass(frozen=True, slots=True)
class _ToyReordered(FieldsHashMxn):
    y: int
    x: float


class _PlainNotDataclass(FieldsHashMxn):
    pass


class TestFieldsHashMxn(TestCase):

    def test_hash_is_deterministic(self) -> None:
        self.assertEqual(_Toy(1.0, 2).compute_hash(), _Toy(1.0, 2).compute_hash())

    def test_hash_ignores_float_noise_below_5dp(self) -> None:
        # 1.0 and 1.000001 both round to "1.00000" at 5dp.
        self.assertEqual(
            _Toy(1.0, 2).compute_hash(),
            _Toy(1.000001, 2).compute_hash(),
        )

    def test_hash_sensitive_at_5dp(self) -> None:
        # 1.0 → "1.00000", 1.00001 → "1.00001".
        self.assertNotEqual(
            _Toy(1.0, 2).compute_hash(),
            _Toy(1.00001, 2).compute_hash(),
        )

    def test_hash_sensitive_to_int_field(self) -> None:
        self.assertNotEqual(
            _Toy(1.0, 2).compute_hash(),
            _Toy(1.0, 3).compute_hash(),
        )

    def test_hash_independent_of_field_declaration_order(self) -> None:
        # Mixin sorts fields by name before hashing, so reordering the
        # dataclass declaration of fields with the same names + values is a
        # no-op. (This is what makes the cache key stable across cosmetic
        # refactors of the request dataclass.)
        self.assertEqual(
            _Toy(1.0, 2).compute_hash(),
            _ToyReordered(2, 1.0).compute_hash(),
        )

    def test_hash_sensitive_to_field_name(self) -> None:
        # Because fields are sorted by name, *renaming* a field changes the
        # iteration order and therefore the hash, even if values match.
        # _Toy has fields (x, y); _ToyRenamedField has (x, z).
        # Sorted: ['x','y'] vs ['x','z'] → same value sequence here, BUT
        # only because y and z happen to come after x. Use a name that
        # sorts before x to confirm the order is name-driven.
        @dataclass(frozen=True, slots=True)
        class _ToyPrependedName(FieldsHashMxn):
            x: float
            a: int  # sorts before x

        # _Toy(x=1.0, y=2): sorted ['x','y'] → "1.00000_2"
        # _ToyPrependedName(x=1.0, a=2): sorted ['a','x'] → "2_1.00000"
        self.assertNotEqual(
            _Toy(1.0, 2).compute_hash(),
            _ToyPrependedName(1.0, 2).compute_hash(),
        )

    def test_hash_includes_version_tag(self) -> None:
        # compute_hash appends "_v{hash_version}" before MD5'ing. Bumping the
        # version (the canonical way to invalidate every cached file at once)
        # must change the hash.
        original = _Toy(1.0, 2).compute_hash()
        try:
            FieldsHashMxn.hash_version = 999
            bumped = _Toy(1.0, 2).compute_hash()
        finally:
            FieldsHashMxn.hash_version = 1
        self.assertNotEqual(original, bumped)

    def test_hash_is_eight_char_hex(self) -> None:
        h = _Toy(1.0, 2).compute_hash()
        self.assertEqual(len(h), 8)
        int(h, 16)  # parses as hex; raises if not

    def test_raises_on_non_dataclass(self) -> None:
        with self.assertRaises(TypeError):
            _PlainNotDataclass().compute_hash()


class TestRoundSig(TestCase):
    """``round_sig`` keeps N significant figures, scale-invariantly."""

    def test_rounds_to_significant_figures(self) -> None:
        # 5 sig figs keeps the leading 5 digits regardless of magnitude.
        self.assertEqual(round_sig(6123.4567, 5), 6123.5)
        self.assertEqual(round_sig(0.0080000001, 5), 0.008)

    def test_collapses_sub_precision_noise(self) -> None:
        # Values agreeing to 5 sig figs round to the same float.
        self.assertEqual(round_sig(0.008, 5), round_sig(0.00800000001, 5))

    def test_distinguishes_within_precision(self) -> None:
        self.assertNotEqual(round_sig(0.0080000, 5), round_sig(0.0080001, 5))

    def test_zero_and_non_finite_pass_through(self) -> None:
        self.assertEqual(round_sig(0.0, 5), 0.0)
        self.assertTrue(math.isnan(round_sig(math.nan, 5)))
        self.assertEqual(round_sig(math.inf, 5), math.inf)


def _radiation_ref(
    sun: float,
    lat_deg: float,
    doy: int,
    hour: int,
    a_s: float = 0.25,
    b_s: float = 0.50,
) -> float:
    """Scalar reference: the pre-vectorisation formula, with plain Python ints."""
    lat = math.radians(lat_deg)
    b = 2 * math.pi * (doy - 81) / 364
    dec = math.radians(23.45 * math.sin(b))
    ha = math.radians(15 * (hour - 12))
    cz = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * math.cos(ha)
    return max(0.0, 1361.0 * cz) * (a_s + b_s * sun / 60.0)


class TestSunshineToRadiation(ArrayTestCase):
    """``sunshine_to_radiation`` is vectorised and dtype-safe.

    The caller passes whole columns: sunshine (float32), day-of-year and hour as
    the *unsigned* dtypes stored in the arrays. The subtraction inside the formula
    must not underflow-wrap on those unsigned inputs (the regression this guards).
    """

    LAT = 52.0
    # Span the underflow-prone corners: early-year days and morning hours, plus
    # afternoon / mid-year for good measure.
    DOY = (1, 30, 80, 172, 200, 355)
    HOUR = (0, 6, 11, 12, 15, 23)
    SUN = (0, 15, 30, 45, 60, 20)

    def _call_with_unsigned(self) -> np.ndarray:
        return sunshine_to_radiation(
            np.array(self.SUN, dtype=np.float32),
            self.LAT,
            np.array(self.DOY, dtype=np.uint16),  # unsigned, as the caller passes
            np.array(self.HOUR, dtype=np.uint8),  # unsigned, as stored in the dtype
        )

    def test_matches_scalar_reference(self) -> None:
        # Vectorised output must equal the old per-element scalar computation.
        expected = np.array(
            [
                _radiation_ref(s, self.LAT, d, h)
                for s, d, h in zip(self.SUN, self.DOY, self.HOUR, strict=True)
            ],
        )
        self.assertAllClose(self._call_with_unsigned(), expected, atol=1e-3)

    def test_no_underflow_on_unsigned_inputs(self) -> None:
        # Underflow would produce garbage declination/hour-angle and thus wildly
        # wrong (or negative) irradiance. Result must be finite and non-negative.
        out = self._call_with_unsigned()
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertTrue(np.all(out >= 0.0))

    def test_returns_float32(self) -> None:
        self.assertEqual(self._call_with_unsigned().dtype, np.float32)

    def test_shape_preserved(self) -> None:
        self.assertEqual(self._call_with_unsigned().shape, (len(self.SUN),))

    def test_night_hour_clamps_to_zero(self) -> None:
        # Midnight in winter: sun below the horizon → cos(zenith) < 0 → clamped 0.
        out = sunshine_to_radiation(
            np.array([30.0], dtype=np.float32),
            self.LAT,
            np.array([1], dtype=np.uint16),
            np.array([0], dtype=np.uint8),
        )
        self.assertEqual(float(out[0]), 0.0)
