import math
from dataclasses import dataclass
from unittest import TestCase

from gdhprediction.utils import FieldsHashMxn, round_sig


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
