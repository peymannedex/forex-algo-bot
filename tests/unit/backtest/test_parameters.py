from __future__ import annotations

import pytest

from fxbot.backtest.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntegerParameter,
    ParameterSet,
    ParameterSpace,
)


def make_space() -> ParameterSpace:
    return ParameterSpace(
        (
            IntegerParameter("period", 10, 20, 5),
            FloatParameter("threshold", 0.1, 0.3, 0.1),
            CategoricalParameter("mode", ("fast", "slow")),
        )
    )


def test_parameter_set_is_sorted_and_fingerprint_is_stable() -> None:
    left = ParameterSet((("b", 2), ("a", 1)))
    right = ParameterSet((("a", 1), ("b", 2)))
    assert left.values == (("a", 1), ("b", 2))
    assert left.fingerprint == right.fingerprint


def test_parameter_set_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        ParameterSet((("a", 1), ("a", 2)))


def test_numeric_parameters_generate_inclusive_values() -> None:
    assert IntegerParameter("x", 1, 5, 2).candidates() == (1, 3, 5)
    assert FloatParameter("y", 0.1, 0.3, 0.1).candidates() == (0.1, 0.2, 0.3)


def test_invalid_ranges_are_rejected() -> None:
    with pytest.raises(ValueError):
        IntegerParameter("x", 0, 5, 2)
    with pytest.raises(ValueError):
        FloatParameter("y", 0.0, 1.0, 0.3)


def test_space_grid_is_deterministic_and_complete() -> None:
    space = make_space()
    grid = space.grid()
    assert len(grid) == 18
    assert space.cardinality == 18
    assert grid == space.grid()


def test_random_sample_is_seeded_and_unique() -> None:
    space = make_space()
    first = space.random_sample(7, seed=42)
    second = space.random_sample(7, seed=42)
    assert first == second
    assert len(set(first)) == 7


def test_space_validation_rejects_missing_and_invalid_values() -> None:
    space = make_space()
    with pytest.raises(ValueError, match="missing"):
        space.validate(ParameterSet.from_mapping({"period": 10}))
    with pytest.raises(ValueError, match="Invalid value"):
        space.validate(
            ParameterSet.from_mapping(
                {"period": 11, "threshold": 0.1, "mode": "fast"}
            )
        )


def test_neighbors_change_one_grid_step() -> None:
    space = make_space()
    center = ParameterSet.from_mapping(
        {"period": 15, "threshold": 0.2, "mode": "fast"}
    )
    neighbors = space.neighbors(center)
    assert len(neighbors) == 5
    assert all(item != center for item in neighbors)
