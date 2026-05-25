from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args, get_type_hints
from unittest import TestCase

from gdhprediction.config import GdhPredictionConfig, _GdhPredictionData
from gdhprediction.utils import Transforms


class TestGdhPredictionConfigRoundtrip(TestCase):
    """``write_to_path`` → ``from_path`` must produce an equal config.

    The YAML serialization is the on-disk contract for both
    ``write_default_config`` and user-edited configs. Any drift in
    dataclass fields, default values, or ``transform_key`` literals would
    corrupt cached state — the cache layer keys files by the config's
    ``FieldsHashMxn`` hash, so a roundtrip that preserves equality but
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

    def test_non_default_roundtrips(self) -> None:
        cfg = GdhPredictionConfig(
            gdh_base=3.0,
            temperature_shift=1.5,
            solar_factor=0.02,
            transform_key="radiation_shift",
        )
        self.assertEqual(self._roundtrip(cfg), cfg)

    def test_every_transform_key_roundtrips(self) -> None:
        # Cover every literal in Transforms so adding a new key without
        # also wiring it through asdict/safe_dump/from_path is caught here.
        for key in get_args(Transforms):
            with self.subTest(transform_key=key):
                cfg = GdhPredictionConfig(transform_key=key)
                self.assertEqual(self._roundtrip(cfg), cfg)

    def test_write_default_config_roundtrips(self) -> None:
        GdhPredictionConfig.write_default_config(self.config_path)
        loaded = GdhPredictionConfig.from_path(self.config_path)
        self.assertEqual(loaded, GdhPredictionConfig())

    def test_hash_stable_across_roundtrip(self) -> None:
        cfg = GdhPredictionConfig(
            gdh_base=3.0,
            temperature_shift=1.5,
            solar_factor=0.02,
            transform_key="radiation_shift",
        )
        self.assertEqual(self._roundtrip(cfg).compute_hash(), cfg.compute_hash())


class TestConfigTypedDictContract(TestCase):
    """``_GdhPredictionData`` must mirror ``GdhPredictionConfig`` exactly.

    ``from_path`` validates the YAML mapping and then does
    ``cast(_GdhPredictionData, config_mapping)`` before unpacking into
    the dataclass constructor. The cast is a static-typing promise; if
    the TypedDict ever drifts from the dataclass — a field added on one
    side, a type changed on the other — the cast becomes a lie and
    ``cls(**config_mapping)`` may raise or silently swallow the wrong
    type. This contract test pins the two together.
    """

    def test_typed_dict_field_names_match_dataclass(self) -> None:
        dc_names = {f.name for f in fields(GdhPredictionConfig)}
        td_names = set(get_type_hints(_GdhPredictionData).keys())
        self.assertEqual(dc_names, td_names)

    def test_typed_dict_field_types_match_dataclass(self) -> None:
        dc_hints = get_type_hints(GdhPredictionConfig)
        td_hints = get_type_hints(_GdhPredictionData)
        dc_field_names = {f.name for f in fields(GdhPredictionConfig)}
        # The dataclass hints can include ClassVar/inherited annotations
        # (e.g. FieldsHashMxn.hash_version); compare only declared fields.
        for name in dc_field_names:
            with self.subTest(field=name):
                self.assertEqual(
                    dc_hints[name],
                    td_hints[name],
                    f"{name}: dataclass={dc_hints[name]!r}, "
                    f"TypedDict={td_hints[name]!r}",
                )
