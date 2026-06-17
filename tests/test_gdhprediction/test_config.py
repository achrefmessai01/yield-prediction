from dataclasses import fields
from typing import get_type_hints
from unittest import TestCase

from shared.configs import GdhPredictionConfig, _GdhPredictionData


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
