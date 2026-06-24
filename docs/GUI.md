# Generic GUI for transform configuration

This document explains how to render the transform configuration in the NiceGUI
config app (`src/shared/config_gui.py`) **generically** — driven entirely by the
pydantic models, so adding a new transform variant never requires touching the
GUI.

It is the implementation guide for two remaining pieces:

- **Layer 2** — a reusable form engine (`src/shared/pydantic_form.py`) that turns
  any pydantic model into widgets by introspection.
- **Layer 3** — wiring that engine into `config_gui.py` and validating on *Apply*.

**Layer 1 is already done**: the config fields in `gdhprediction/config.py` carry
their own UI metadata and validation. This guide builds on that.

---

## Background: why this works

The transform configuration is a *discriminated (tagged) union*,
`TransformConfig`:

```python
TransformConfig = Annotated[
    IdentityCfg | ShiftCfg | RadiationShiftCfg | AdaptiveShiftCfg
    | FixedCfg | ConstrainedCfg,
    Field(discriminator="kind"),
]
```

Each member is a pydantic model tagged by a `kind` literal. Two of them
(`RadiationShiftCfg`, `AdaptiveShiftCfg`) hold a *nested* union — `params`, tagged
by `route` (`coefficients` vs. `constants`). So the whole thing is a tree of
unions and models, and a single recursive renderer covers every level.

Layer 1 put four things on the models that the engine reads back out:

| pydantic feature | how the GUI uses it |
|---|---|
| `model_fields` (per field: `annotation`, `default`, `is_required()`) | decide which widget, and seed its initial value |
| the discriminator (`kind` / `route`, a `Literal` field) | render a selector and pick the variant |
| `Field(title=, description=, json_schema_extra={...})` | labels, tooltips, units, and soft `ui_min`/`ui_max`/`step` ranges |
| `TypeAdapter(...).validate_python(...)` | validate the whole edited tree at once, with field-keyed errors |

Because all of that lives on the model, the engine is generic: a seventh
transform variant simply appears in the form, validated and labelled.

One constraint shapes the design: the config models are **frozen**
(`ConfigDict(frozen=True)`), so you cannot bind a widget to a model and mutate it
in place. The engine therefore edits a plain mutable `dict` (a *draft*) and
reconstructs the validated, frozen model only on *Apply*. That is the right shape
anyway — validation belongs at the boundary, not on every keystroke.

---

## Layer 2 — the form engine

Create `src/shared/pydantic_form.py`. It is ~70 lines, written once, and never
needs editing when transforms change.

```python
"""Generic NiceGUI form rendering for pydantic config models.

Renders any pydantic model — or a discriminated union of them — into editable
widgets by introspection, reading the presentation hints declared on the fields
(see the ``gdhprediction.config`` module docstring). No transform-specific code
lives here: adding a variant to ``TransformConfig`` makes it appear in the form
automatically.

The models are frozen, so widgets edit a mutable ``dict`` (a *draft*); the caller
reconstructs the validated model from that draft on submit.
"""

import types
from typing import Literal, Union, get_args, get_origin

from nicegui import ui
from pydantic import BaseModel
from pydantic_core import PydanticUndefined


def _strip_annotated(ann: object) -> object:
    """Peel ``Annotated[T, ...]`` down to ``T``; leave other types unchanged."""
    return get_args(ann)[0] if hasattr(ann, "__metadata__") else ann


def _union_members(ann: object) -> tuple[type[BaseModel], ...] | None:
    """The model classes if ``ann`` is a (possibly Annotated) union of models."""
    ann = _strip_annotated(ann)
    if get_origin(ann) in (Union, types.UnionType):
        members = tuple(
            a for a in get_args(ann)
            if isinstance(a, type) and issubclass(a, BaseModel)
        )
        return members or None
    return None


def _discriminator(members: tuple[type[BaseModel], ...]) -> str:
    """The single ``Literal`` field every union member shares (``kind``/``route``)."""
    for name, fi in members[0].model_fields.items():
        if get_origin(fi.annotation) is Literal:
            return name
    msg = f"{members[0].__name__} has no Literal discriminator field"
    raise ValueError(msg)


def _hints(fi) -> dict:
    """Presentation hints declared on a field: label, help, unit, soft ranges."""
    extra = fi.json_schema_extra if isinstance(fi.json_schema_extra, dict) else {}
    return {"label": fi.title, "help": fi.description, **extra}


def render_union(annotation: object, data: dict, *, label: str) -> None:
    """Render a discriminated union: a selector plus the chosen variant's fields.

    ``data`` is the mutable dict this union writes into. It ends up holding the
    discriminator key plus the selected variant's field values, ready to feed to
    ``TypeAdapter.validate_python``.
    """
    members = _union_members(annotation)
    disc = _discriminator(members)
    by_tag = {m.model_fields[disc].default: m for m in members}
    data.setdefault(disc, next(iter(by_tag)))  # keep a pre-loaded tag, else default

    @ui.refreshable
    def variant_fields() -> None:
        render_model(by_tag[data[disc]], data)

    def on_select(event) -> None:
        # A user-initiated switch resets the params to the new variant's
        # defaults: its fields differ, so carrying old values over is meaningless.
        data.clear()
        data[disc] = event.value
        variant_fields.refresh()

    ui.select(list(by_tag), value=data[disc], label=label, on_change=on_select)
    variant_fields()


def render_model(model_cls: type[BaseModel], data: dict) -> None:
    """Render every editable field of ``model_cls`` into ``data``."""
    for name, fi in model_cls.model_fields.items():
        ann = fi.annotation
        if get_origin(ann) is Literal:
            continue  # the discriminator — owned by the enclosing select
        h = _hints(fi)
        label = h.get("label") or name.replace("_", " ").capitalize()

        if _strip_annotated(ann) == tuple[float, float]:   # TempRange etc.
            _render_range(name, fi, data, label, h.get("unit", ""))
        elif _union_members(ann):                          # nested union (params)
            data[name] = data.get(name) or {}
            render_union(ann, data[name], label=label)
        elif _strip_annotated(ann) is float:
            _render_number(name, fi, data, label, h)
        # add str / int / bool branches here as new field types appear


def _render_number(name: str, fi, data: dict, label: str, h: dict) -> None:
    default = fi.default if fi.default is not PydanticUndefined else 0.0
    data.setdefault(name, default)                  # keep loaded value if present
    element = ui.number(
        label, value=data[name], suffix=h.get("unit", ""),
        min=h.get("ui_min"), max=h.get("ui_max"), step=h.get("step"),
        on_change=lambda e: data.__setitem__(name, e.value),
    )
    if h.get("help"):
        element.tooltip(h["help"])


def _render_range(name: str, fi, data: dict, label: str, unit: str) -> None:
    default = fi.default if fi.default is not PydanticUndefined else (0.0, 0.0)
    lo, hi = data.get(name) or default
    data[name] = [lo, hi]                            # list; pydantic coerces to tuple
    with ui.row():
        ui.number(f"{label} min", value=lo, suffix=unit,
                  on_change=lambda e: data[name].__setitem__(0, e.value))
        ui.number(f"{label} max", value=hi, suffix=unit,
                  on_change=lambda e: data[name].__setitem__(1, e.value))
```

### How it behaves, walked through

Render the whole transform with one call:

```python
draft: dict = {}
render_union(TransformConfig, draft, label="Transform")
```

1. `render_union` finds the six members, reads their discriminator (`kind`), and
   builds `{"identity": IdentityCfg, "shift": ShiftCfg, ...}`. It seeds
   `draft["kind"]` (default `radiation_shift`, the union's default) and draws the
   selector.
2. `variant_fields()` calls `render_model(RadiationShiftCfg, draft)`. That model
   has a `kind` (skipped) and a `params` field whose annotation is the
   `RadCoefficients | RadConstants` union — so `render_model` recurses via
   `render_union(ann, draft["params"], label="Params")`.
3. The nested union draws a second selector (`coefficients` / `constants`). Pick
   `constants` and the six `_Geometry` fields render as numbers — each with the
   `title` as its label, the `unit` as a suffix, `ui_min`/`ui_max`/`step` bounding
   the spinner, and `description` as a tooltip — all read straight off the model.
4. Switching either selector clears that union's sub-dict and re-renders the new
   variant's defaults. Switching the outer `kind` clears the whole `params` too,
   since it belongs to the old variant.

`draft` now mirrors the shape `TransformConfig` expects, e.g.:

```python
{"kind": "radiation_shift",
 "params": {"route": "constants", "footprint_area": 1000.0, ...}}
```

---

## Layer 3 — wiring into `config_gui.py`

The GUI holds the scratch `draft`, renders the engine into it, and validates once
on *Apply*. Validation, bounds, and the `min < max` range rule all come for free
from Layer 1.

```python
from pydantic import TypeAdapter, ValidationError

from gdhprediction.config import GdhLocalizedPredictionConfig, TransformConfig
from shared.pydantic_form import render_union

_TRANSFORM = TypeAdapter(TransformConfig)   # the union is not a BaseModel, so adapt
draft: dict = {}                            # module/page state


def build_transform_section() -> None:
    # In the left column, replacing the hand-written transform widgets:
    render_union(TransformConfig, draft, label="Transform")


def on_apply() -> None:
    try:
        transform = _TRANSFORM.validate_python(draft)   # every rule fires here
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])  # e.g. params.footprint_area
            ui.notify(f"{loc}: {err['msg']}", type="negative")
        return

    cfg = GdhLocalizedPredictionConfig(
        latitude=lat_input.value,
        longitude=lon_input.value,
        gdh_base=gdh_base_input.value,
        transform=transform,
    )
    cfg.write_to_path(paths.model_config)
    # ... build YieldPrediction, run get_prediction(), refresh the plots
```

Every failure mode surfaces through the same channel, keyed by field:

- wrong type → `params.solar_factor: Input should be a valid number`
- broken invariant (`gt=0`) → `params.footprint_area: Input should be greater than 0`
- inverted range (`TempRange`) → `temperature_range: Value error, range min ... must be < max ...`
- unknown field (`extra="forbid"`) → caught too

No validation logic is duplicated in the GUI.

### Editing an existing config

To open the form on a config loaded from disk, pre-fill the draft before
rendering. `render_union`'s `setdefault` keeps the loaded discriminator, and each
field's `setdefault` keeps the loaded value:

```python
draft.clear()
draft.update(existing_cfg.transform.model_dump())   # dict shaped like the union
build_transform_section()
```

---

## Design rationale & gotchas

- **Single source of truth.** The engine knows nothing about specific transforms;
  everything comes from `model_fields` and the `Field(...)` hints. A new variant
  needs zero GUI changes — the same drift-resistance the tagged union gave the
  data model, extended to the UI.
- **Frozen models → edit a draft dict, validate at the boundary.** Do not try to
  bind widgets to the models. (`model_copy(update=...)` is *not* a shortcut here:
  it skips validation, so a bad value would pass silently.)
- **Hard constraints vs. soft ranges.** `ge`/`gt` on the model are real
  invariants and *reject* input on Apply; `ui_min`/`ui_max` only bound the
  spinner and never reject. Keep them distinct (Layer 1 already does).
- **Switching a selector resets that variant's fields.** This is deliberate — a
  different `kind`/`route` has different parameters. If you ever need to preserve
  overlapping values across a switch, cache them keyed by tag instead of
  `data.clear()`.
- **`Field(default=...)` keyword form.** When adding fields, always use the
  keyword form; the pydantic mypy plugin does not recognise a positional
  `Field(4.5, ...)` default and will treat the field as required.
- **Reusable.** The same engine renders `YieldPredictionConfig` (stages, weights,
  camera maps) once those fields carry the same `Field(...)` hints.

### Optional: per-variant custom widgets

The generic form is good, not bespoke. If one variant needs a custom control
(e.g. a "Heavily ventilated / Closed" preset that fills several geometry fields
at once), add a small override keyed by `kind` at the top of `render_model`, and
fall through to the generic path for everything else. Keep overrides rare.

---

## Testing

The engine is mostly UI, but its introspection helpers are pure and worth pinning
without spinning up NiceGUI:

- `_union_members(TransformConfig)` returns all six variants;
  `_union_members(RadiationShiftCfg.model_fields["params"].annotation)` returns
  the two routes.
- `_discriminator(...)` returns `"kind"` for the top union and `"route"` for the
  nested ones.
- Round-trip: build a `draft` for each variant, `validate_python` it, and assert
  it `.build()`s the expected transform — mirrors the existing
  `TestTransformDispatch` in `tests/test_gdhprediction/test_config.py`.
- Error mapping: feed a deliberately invalid draft (negative area, inverted
  range) and assert `ValidationError.errors()` carries the expected `loc`.
```