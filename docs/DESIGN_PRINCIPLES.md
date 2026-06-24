# Design principles

This file defines the design principles for the project.
It is supposed to be a living document, and should be updated as the project evolves.

## Rules at a glance

1. [Unidirectional dependencies](#unidirectional-dependencies): no cycles in the package graph.
2. [Encapsulation](#encapsulation): talk through public contracts, not internals.
3. [Boundary contracts](#boundary-contracts-units-time-and-meaning): declare units, time frame, and meaning at every boundary.
4. [Immutability and dataclasses](#immutability-and-dataclasses): prefer values that cannot change after creation.
5. [Determinism and reproducibility](#determinism-and-reproducibility): same inputs, same outputs; seed any randomness.
6. [Class reuse](#class-reuse): subclass within LSP when you own the type, compose when you don't.
7. [Error handling](#error-handling-fail-fast-fail-loud): validate early and fail loudly; never swallow errors.

## Unidirectional dependencies

**Dependencies between packages point in one direction only; the dependency graph must contain no cycles.**

The project is made of several packages, each of them having a specific responsibility.
The dependencies between these packages should be unidirectional,
meaning that a package should only depend on packages that are "below" it in the dependency graph.

There are currently three packages in the project:

- `gdhprediction`: This package contains the core logic for growing degree hours prediction.
- `yieldprediction`: This package contains the core logic for yield prediction.
- `shared`: This package contains shared utilities, abstractions, and general data structures
  that are used by both `gdhprediction` and `yieldprediction`.

Packages `gdhprediction` and `yieldprediction` depend on `shared`, but they do not depend on each other.
The package `shared` is a bottom package, meaning that it does not depend on any other package.

Arguably, the GUI should be in a separate package which depends on both `gdhprediction` and `yieldprediction`.

*Further reading: [Acyclic dependencies principle](https://en.wikipedia.org/wiki/Acyclic_dependencies_principle) (Wikipedia). The dependency graph between components should contain no cycles.*

## Encapsulation

**Each package exposes a small public API and communicates across boundaries through named types, never raw internal structures.**

Each package should encapsulate its internal logic and data structures, exposing only what is necessary for other
packages to use it.
This means that each package should have a clear public API, and should not expose internal details that are not
relevant for other packages.

Instead of using other packages' internal data structures, each package should define and maintain contracts (
interfaces).
This permits to decouple the packages and allows for easier maintenance and evolution of the codebase.

Express a contract as a *named type*, not a loosely-typed structure.
Passing raw dictionaries, tuples, or bare arrays across a boundary forces the receiver to know the producer's internal
layout. That is exactly the coupling contracts are meant to remove.
Prefer a small dataclass whose fields document what the data means.
A change in how the data is produced then fails at the boundary, not deep inside unrelated logic.

*Further reading: [Encapsulation](https://en.wikipedia.org/wiki/Encapsulation_(computer_programming)) (Wikipedia), and
the [Primitive Obsession](https://refactoring.guru/smells/primitive-obsession) code smell. Both cover why a domain
concept deserves a named type rather than a bare string or number.*

## Boundary contracts: units, time, and meaning

**A value's type is not its whole contract: its units, timezone, and calendar assumptions must be explicit too.**

When data crosses a boundary, between packages or in from an external source, its type alone does not capture everything
the receiver needs to know.
A `float` could be Celsius or Kelvin. A timestamp could be UTC, local standard time, or a fixed offset.
An hourly series could follow the civil calendar (with 23- and 25-hour days at daylight-saving transitions) or a
uniform clock.
Two values can share a type and still be incompatible.

Many subtle, expensive pipeline bugs originate here. The code runs, the types match, and the numbers are wrong by an
hour, a factor, or a day.
Nothing crashes, so the error is easy to miss.

To guard against this:

- **Make the frame explicit.** State a value's units, timezone, and calendar assumptions in its name, its type, or a
  comment where it enters the system. `temperature_celsius` and `start_utc` carry their contract; `temperature` and
  `start` do not.
- **Convert in exactly one place.** When two sources must be reconciled (different units, or different time frames),
  convert at a single named seam and document why. Conversions scattered across call sites drift apart over time and
  become nearly impossible to verify.
- **Treat the frame as part of the contract.** If a function promises "hourly values in UTC", that promise is as binding
  as its return type. Changing it is a breaking change, even though the type signature does not move.

*Further reading: the [Mars Climate Orbiter](https://en.wikipedia.org/wiki/Mars_Climate_Orbiter) (Wikipedia) was lost
because one component produced pound-force·seconds where another expected newton·seconds. The types matched; the units
did not.*

## Immutability and dataclasses

**Prefer immutable dataclasses: build an object once and never mutate it afterwards.**

Immutability means that once an object is created, it cannot be modified.
This makes code easier to reason about and debug, and prevents a whole class of bugs caused by shared, mutating state.
Dataclasses provide a convenient way to define such structures with little boilerplate, and can be made immutable by
setting `frozen=True`.

*Further reading: [Immutable object](https://en.wikipedia.org/wiki/Immutable_object) (Wikipedia), and the
Python [`dataclasses`](https://docs.python.org/3/library/dataclasses.html) documentation (see the `frozen` parameter).*

## Determinism and reproducibility

**The same inputs must always produce the same outputs; any randomness is explicit and seedable.**

Determinism makes results reproducible: the same inputs always yield the same output.
A wrong result can then be re-run and traced to its cause.
Non-deterministic output cannot be reliably reproduced, which impedes debugging.

Randomness is sometimes needed, for example to model uncertainty.
When it is, make the source explicit and controllable, typically by accepting a seed or a random generator as a
parameter.
A run can then be reproduced exactly when investigating a problem, and left to vary only when that is the intent.
Hidden, unseeded randomness impedes reproducibility and debugging, and weakens the other guarantees in this document.

*Further reading: [Deterministic algorithm](https://en.wikipedia.org/wiki/Deterministic_algorithm) (Wikipedia).*

## Class reuse

**Reuse by subclassing only when you own the base type and respect the Liskov Substitution Principle; otherwise reuse by composition.**

If a package needs to use a modified version of a class defined in another package, it should not modify the original
class directly.
Instead, it should create a new class that inherits from the original class and adds the necessary modifications.
This allows for better separation of concerns and prevents unintended side effects on the original class.
The new class must respect the contract defined by the original class and Liskov Substitution Principle.
That is, it should be able to be used in place of the original class without breaking the functionality of the code that
uses it.

If a package needs to use a class defined in another package it does not control, it should reach it through delegation
(composition) rather than inheritance.
Use an adapter to re-express a single foreign class's interface as the one your code expects.
Use a facade to present one simplified interface over a more complex set of classes, for example the subsystems piped
into the GUI.

The builder pattern serves the same goal. It accumulates values in a mutable helper and emits a single finished object
at the end. This produces an immutable result without ever exposing a half-built, mutable one.

*Further reading: [Liskov substitution principle](https://en.wikipedia.org/wiki/Liskov_substitution_principle) (
Wikipedia), and [Composition over inheritance](https://en.wikipedia.org/wiki/Composition_over_inheritance) (Wikipedia).
Delegation is often safer than subclassing.*

## Error handling: fail fast, fail loud

**Validate assumptions at the boundary and fail loudly; never swallow errors with a silent catch-all.**

When something is wrong, stop as early as possible and say clearly what happened.
The best place to check an assumption is the boundary, where data enters the system or crosses between packages.
That is where we still have the context for a helpful error message.

Avoid the silent catch-all: a broad `except` that swallows any error and continues.
This does not prevent the failure; it defers it to a less obvious location.
A missing field ignored at load time resurfaces later as an error in unrelated logic, or as a plausible but incorrect
result.

A few guidelines:

- **Validate inputs at construction or entry**, and raise immediately with a message that names what was expected and
  what was found.
- **Catch only the specific exceptions you can actually handle**, and only where you can do something meaningful about
  them. A bare `except:` should be rare and deliberate.
- **Fail at the point of detection, not later.** A crash is recoverable; a silently incorrect result that downstream
  code trusts is not.

This rule complements the boundary contracts above: checking a contract is only worthwhile if a violation is reported
rather than hidden.

*Further reading: [Fail-fast](https://en.wikipedia.org/wiki/Fail-fast) (Wikipedia), and Martin Fowler's short
article [Fail Fast](https://martinfowler.com/ieeeSoftware/failFast.pdf) (PDF, IEEE Software).*