"""Typed parameter spaces for deterministic strategy optimization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite
from random import Random
from typing import TypeAlias

ParameterValue: TypeAlias = bool | int | float | str


def _name(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("parameter name cannot be empty")
    return normalized


def _normalize_value(value: ParameterValue) -> ParameterValue:
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("floating parameter values must be finite")
        return float(value)
    if isinstance(value, bool | int | str):
        return value
    raise TypeError(f"Unsupported parameter value: {type(value)!r}")


@dataclass(frozen=True, slots=True)
class ParameterSet:
    """Immutable, order-independent parameter assignment."""

    values: tuple[tuple[str, ParameterValue], ...]

    def __post_init__(self) -> None:
        normalized: list[tuple[str, ParameterValue]] = []
        seen: set[str] = set()
        for raw_name, raw_value in self.values:
            name = _name(raw_name)
            if name in seen:
                raise ValueError(f"Duplicate parameter name: {name}")
            seen.add(name)
            normalized.append((name, _normalize_value(raw_value)))
        object.__setattr__(self, "values", tuple(sorted(normalized)))

    @classmethod
    def from_mapping(cls, values: Mapping[str, ParameterValue]) -> ParameterSet:
        return cls(tuple(values.items()))

    def as_dict(self) -> dict[str, ParameterValue]:
        return dict(self.values)

    def __getitem__(self, name: str) -> ParameterValue:
        try:
            return dict(self.values)[name]
        except KeyError as exc:
            raise KeyError(f"Unknown parameter: {name}") from exc

    def get(self, name: str, default: ParameterValue | None = None) -> ParameterValue | None:
        return dict(self.values).get(name, default)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CategoricalParameter:
    """Finite categorical parameter."""

    name: str
    choices: tuple[ParameterValue, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        normalized = tuple(_normalize_value(value) for value in self.choices)
        if not normalized:
            raise ValueError("choices cannot be empty")
        if len(set(normalized)) != len(normalized):
            raise ValueError("choices must be unique")
        object.__setattr__(self, "choices", normalized)

    def candidates(self) -> tuple[ParameterValue, ...]:
        return self.choices


@dataclass(frozen=True, slots=True)
class IntegerParameter:
    """Inclusive integer range."""

    name: str
    minimum: int
    maximum: int
    step: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.maximum < self.minimum:
            raise ValueError("maximum cannot be below minimum")
        if (self.maximum - self.minimum) % self.step != 0:
            raise ValueError("range must be exactly divisible by step")

    def candidates(self) -> tuple[ParameterValue, ...]:
        return tuple(range(self.minimum, self.maximum + 1, self.step))


@dataclass(frozen=True, slots=True)
class FloatParameter:
    """Inclusive finite floating-point grid."""

    name: str
    minimum: float
    maximum: float
    step: float
    precision: int = 12

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name))
        for field_name in ("minimum", "maximum", "step"):
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)
        if self.step <= 0.0:
            raise ValueError("step must be positive")
        if self.maximum < self.minimum:
            raise ValueError("maximum cannot be below minimum")
        if not 0 <= self.precision <= 15:
            raise ValueError("precision must be between 0 and 15")
        span = (self.maximum - self.minimum) / self.step
        if abs(span - round(span)) > 1e-9:
            raise ValueError("range must be exactly divisible by step")

    def candidates(self) -> tuple[ParameterValue, ...]:
        count = round((self.maximum - self.minimum) / self.step)
        return tuple(
            round(self.minimum + index * self.step, self.precision)
            for index in range(count + 1)
        )


ParameterDefinition: TypeAlias = (
    CategoricalParameter | IntegerParameter | FloatParameter
)


@dataclass(frozen=True, slots=True)
class ParameterSpace:
    """Finite parameter space with deterministic grid and random sampling."""

    definitions: tuple[ParameterDefinition, ...]

    def __post_init__(self) -> None:
        if not self.definitions:
            raise ValueError("definitions cannot be empty")
        names = tuple(item.name for item in self.definitions)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        object.__setattr__(
            self,
            "definitions",
            tuple(sorted(self.definitions, key=lambda item: item.name)),
        )

    @property
    def cardinality(self) -> int:
        total = 1
        for definition in self.definitions:
            total *= len(definition.candidates())
        return total

    @property
    def fingerprint(self) -> str:
        payload = [
            [item.name, list(item.candidates())]
            for item in self.definitions
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def definition(self, name: str) -> ParameterDefinition:
        normalized = _name(name)
        for item in self.definitions:
            if item.name == normalized:
                return item
        raise KeyError(f"Unknown parameter: {normalized}")

    def validate(self, parameter_set: ParameterSet) -> None:
        expected = {item.name for item in self.definitions}
        actual = {name for name, _ in parameter_set.values}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Parameter mismatch; missing={missing}, extra={extra}")
        for definition in self.definitions:
            value = parameter_set[definition.name]
            if value not in definition.candidates():
                raise ValueError(
                    f"Invalid value for {definition.name}: {value!r}"
                )

    def grid(self, *, limit: int | None = None) -> tuple[ParameterSet, ...]:
        if limit is not None and limit <= 0:
            raise ValueError("limit must be positive")
        output: list[ParameterSet] = []
        names = tuple(item.name for item in self.definitions)
        candidate_lists = tuple(item.candidates() for item in self.definitions)
        for values in product(*candidate_lists):
            output.append(ParameterSet(tuple(zip(names, values, strict=True))))
            if limit is not None and len(output) >= limit:
                break
        return tuple(output)

    def random_sample(
        self,
        count: int,
        *,
        seed: int,
    ) -> tuple[ParameterSet, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        sample_size = min(count, self.cardinality)
        indices = Random(seed).sample(range(self.cardinality), sample_size)
        return tuple(self._decode_index(index) for index in indices)

    def neighbors(self, center: ParameterSet) -> tuple[ParameterSet, ...]:
        self.validate(center)
        output: list[ParameterSet] = []
        base = center.as_dict()
        for definition in self.definitions:
            candidates = definition.candidates()
            current = base[definition.name]
            index = candidates.index(current)
            for neighbor_index in (index - 1, index + 1):
                if 0 <= neighbor_index < len(candidates):
                    changed = dict(base)
                    changed[definition.name] = candidates[neighbor_index]
                    output.append(ParameterSet.from_mapping(changed))
        return tuple(output)

    def __iter__(self) -> Iterator[ParameterDefinition]:
        return iter(self.definitions)

    def _decode_index(self, index: int) -> ParameterSet:
        if not 0 <= index < self.cardinality:
            raise IndexError("parameter-space index out of range")
        values: list[tuple[str, ParameterValue]] = []
        remainder = index
        reversed_definitions: Sequence[ParameterDefinition] = tuple(
            reversed(self.definitions)
        )
        decoded: list[tuple[str, ParameterValue]] = []
        for definition in reversed_definitions:
            candidates = definition.candidates()
            remainder, offset = divmod(remainder, len(candidates))
            decoded.append((definition.name, candidates[offset]))
        values.extend(reversed(decoded))
        return ParameterSet(tuple(values))
