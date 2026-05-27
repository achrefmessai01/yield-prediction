import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, fields, is_dataclass
from typing import ClassVar, Final, Literal, get_args

import numpy as np
from numpy import typing as npt
from typing_extensions import TypeIs


class FieldsHashMxn:

    @property
    def hash_fields(self) -> Iterable[str]:
        if not is_dataclass(self):
            raise TypeError(f"Default {type(self)} can only be mixed with dataclasses.")
        return (str(getattr(self, f.name)) for f in fields(self))

    def compute_hash(self) -> str:
        key = "_".join(self.hash_fields)
        return hashlib.md5(key.encode(encoding="utf-8")).hexdigest()[:8]

