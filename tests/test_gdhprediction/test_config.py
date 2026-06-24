from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase

from pydantic import ValidationError

from gdhprediction.config import (
    AdaptiveCoefficients,
    AdaptiveConstants,
    AdaptiveShiftCfg,
    ConstrainedCfg,
    FixedCfg,
    GdhPredictionConfig,
    IdentityCfg,
    RadCoefficients,
    RadConstants,
    RadiationShiftCfg,
    ShiftCfg,
)
from gdhprediction.transforms import (
    AdaptiveShift,
    Constrained,
    Fixed,
    Identity,
    RadiationShift,
    Shift,
    TemperatureTransform,
)


def _rad_constants() -> RadConstants:
    return RadConstants(
        footprint_area=1000.0,
        coverage_area=1400.0,
        heat_transfer_coeff=4.0,
        average_height=3.0,
        air_infiltration=1.0,
        solar_coeff=0.5,
    )


def _adaptive_coeffs() -> AdaptiveCoefficients:
    return AdaptiveCoefficients(
        base_heat_loss=6000.0,
        solar_factor=500.0,
        ventilation=50.0,
        heating=20000.0,
    )


def _adaptive_constants() -> AdaptiveConstants:
    return AdaptiveConstants(
        footprint_area=1000.0,
        coverage_area=1400.0,
        heat_transfer_coeff=4.0,
        average_height=3.0,
        air_infiltration=1.0,
        solar_coeff=0.5,
        ventilation=50.0,
        heating=20000.0,
    )


# One representative config per transform variant / parameterisation route.
# Covers every member of the discriminated union, so adding a transform without
# wiring its serialization is caught by the roundtrip and hash tests below.
_VARIANTS: list[GdhPredictionConfig] = [
    GdhPredictionConfig(transform=IdentityCfg()),
    GdhPredictionConfig(transform=ShiftCfg(degree_shift=1.5)),
    GdhPredictionConfig(
        transform=RadiationShiftCfg(
            params=RadCoefficients(degree_shift=1.0, solar_factor=0.02),
        ),
    ),
    GdhPredictionConfig(transform=RadiationShiftCfg(params=_rad_constants())),
    GdhPredictionConfig(transform=AdaptiveShiftCfg(params=_adaptive_coeffs())),
    GdhPredictionConfig(transform=AdaptiveShiftCfg(params=_adaptive_constants())),
    GdhPredictionConfig(transform=FixedCfg(temperature=20.0)),
    GdhPredictionConfig(transform=ConstrainedCfg(temperature_range=(4.5, 35.0))),
]


class TestGdhPredictionConfigRoundtrip(TestCase):
    """``write_to_path`` → ``from_path`` must produce an equal config.

    The YAML serialization is the on-disk contract for both
    ``write_default_config`` and user-edited configs. The cache layer keys
    files by the config's hash, so a roundtrip that preserves equality but
    drifts the hash would silently invalidate every cached array.
    """

    def setUp(self) -> None:
        tmpdir = TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.config_path = Path(tmpdir.name) / "config.yaml"

    def _roundtrip(self, cfg: GdhPredictionConfig) -> GdhPredictionConfig:
        cfg.write_to_path(self.config_path)
        return GdhPredictionConfig.from_path(self.config_path)

    def test_default_roundtrips(self) -> None:
        cfg = GdhPredictionConfig()
        self.assertEqual(self._roundtrip(cfg), cfg)

    def test_every_variant_roundtrips(self) -> None:
        for cfg in _VARIANTS:
            with self.subTest(transform=cfg.transform.kind):
                self.assertEqual(self._roundtrip(cfg), cfg)

    def test_write_default_config_roundtrips(self) -> None:
        GdhPredictionConfig.write_default_config(self.config_path)
        loaded = GdhPredictionConfig.from_path(self.config_path)
        self.assertEqual(loaded, GdhPredictionConfig())

    def test_hash_stable_across_roundtrip(self) -> None:
        for cfg in _VARIANTS:
            with self.subTest(transform=cfg.transform.kind):
                self.assertEqual(
                    self._roundtrip(cfg).compute_hash(),
                    cfg.compute_hash(),
                )

    def test_distinct_params_hash_differently(self) -> None:
        a = GdhPredictionConfig(transform=ShiftCfg(degree_shift=1.0))
        b = GdhPredictionConfig(transform=ShiftCfg(degree_shift=2.0))
        self.assertNotEqual(a.compute_hash(), b.compute_hash())

    def test_distinct_transform_kinds_hash_differently(self) -> None:
        a = GdhPredictionConfig(transform=ShiftCfg(degree_shift=2.0))
        b = GdhPredictionConfig(transform=IdentityCfg())
        self.assertNotEqual(a.compute_hash(), b.compute_hash())

    def test_gdh_base_affects_hash(self) -> None:
        a = GdhPredictionConfig(gdh_base=4.5)
        b = GdhPredictionConfig(gdh_base=3.0)
        self.assertNotEqual(a.compute_hash(), b.compute_hash())

    def test_route_invariance_same_resolved_transform_same_hash(self) -> None:
        # Two different config representations that resolve to the *same*
        # transform must share a cache key — the point of keying on the
        # resolved transform rather than the config representation.
        via_constants = GdhPredictionConfig(
            transform=RadiationShiftCfg(params=_rad_constants()),
        )
        resolved = via_constants.temperature_transform
        assert isinstance(resolved, RadiationShift)
        via_coeffs = GdhPredictionConfig(
            transform=RadiationShiftCfg(
                params=RadCoefficients(
                    degree_shift=resolved.degree_shift,
                    solar_factor=resolved.solar_factor,
                ),
            ),
        )
        self.assertEqual(via_constants.compute_hash(), via_coeffs.compute_hash())

    def test_float_noise_below_precision_shares_hash(self) -> None:
        # Transforms quantize float fields at construction, so coefficients
        # that differ only below the significant-figure threshold (e.g. a
        # ``from_constants`` result vs a hand-written value) resolve to the
        # same transform and therefore the same cache key.
        clean = GdhPredictionConfig(
            transform=RadiationShiftCfg(
                params=RadCoefficients(degree_shift=0.0, solar_factor=0.008),
            ),
        )
        noisy = GdhPredictionConfig(
            transform=RadiationShiftCfg(
                params=RadCoefficients(degree_shift=0.0, solar_factor=0.00800000001),
            ),
        )
        self.assertEqual(clean.compute_hash(), noisy.compute_hash())


class TestTransformDispatch(TestCase):
    """``temperature_transform`` must build the transform each variant names."""

    def test_build_returns_expected_transform(self) -> None:
        cases: list[tuple[Any, type[TemperatureTransform]]] = [
            (IdentityCfg(), Identity),
            (ShiftCfg(), Shift),
            (RadiationShiftCfg(), RadiationShift),
            (RadiationShiftCfg(params=_rad_constants()), RadiationShift),
            (AdaptiveShiftCfg(params=_adaptive_coeffs()), AdaptiveShift),
            (AdaptiveShiftCfg(params=_adaptive_constants()), AdaptiveShift),
            (FixedCfg(), Fixed),
            (ConstrainedCfg(), Constrained),
        ]
        for variant, expected in cases:
            with self.subTest(transform=variant.kind):
                cfg = GdhPredictionConfig(transform=variant)
                self.assertIsInstance(cfg.temperature_transform, expected)


class TestConfigValidation(TestCase):
    """The discriminated union must reject malformed config up front."""

    def test_unknown_key_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GdhPredictionConfig.model_validate(
                {"gdh_base": 4.5, "transform": {"kind": "shift", "bogus": 1}},
            )

    def test_unknown_transform_kind_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GdhPredictionConfig.model_validate({"transform": {"kind": "nope"}})

    def test_constants_route_requires_geometry(self) -> None:
        with self.assertRaises(ValidationError):
            GdhPredictionConfig.model_validate(
                {
                    "transform": {
                        "kind": "radiation_shift",
                        "params": {"route": "constants"},
                    },
                },
            )
