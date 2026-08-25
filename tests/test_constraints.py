"""Constraint metadata and violation scoring."""

import jax.numpy as jnp
import pytest

from cfm.core.constraints import (
    EQUALITY,
    INEQUALITY,
    Constraint,
    equality,
    inequality,
)


def test_equality_violation_is_absolute_residual():
    c = equality(lambda x: jnp.sum(x**2) - 1.0)
    assert c.kind == EQUALITY
    assert float(c.violation(jnp.array([2.0, 0.0]))) == pytest.approx(3.0)
    assert float(c.violation(jnp.array([1.0, 0.0]))) == pytest.approx(0.0)


def test_equality_violation_is_symmetric():
    """Overshooting and undershooting count the same."""
    c = equality(lambda x: jnp.atleast_1d(x[0]))
    assert float(c.violation(jnp.array([-0.5]))) == pytest.approx(0.5)
    assert float(c.violation(jnp.array([0.5]))) == pytest.approx(0.5)


def test_inequality_violation_is_zero_when_feasible():
    c = inequality(lambda x: jnp.atleast_1d(-x[0]))
    assert c.kind == INEQUALITY
    assert float(c.violation(jnp.array([3.0]))) == 0.0
    assert float(c.violation(jnp.array([-2.0]))) == pytest.approx(2.0)


def test_vector_residual_reports_the_worst_row():
    c = equality(lambda x: x - jnp.array([1.0, 1.0, 1.0]))
    v = c.violation(jnp.array([1.0, 1.0, 4.0]))
    assert float(v) == pytest.approx(3.0)


def test_violations_maps_over_a_batch():
    c = inequality(lambda x: jnp.atleast_1d(-x[0]))
    batch = jnp.array([[1.0], [-1.0], [-3.0]])
    assert jnp.allclose(c.violations(batch), jnp.array([0.0, 1.0, 3.0]))


def test_custom_violation_metric_is_respected():
    c = equality(lambda x: x, violation=lambda x: jnp.sum(jnp.abs(x)))
    assert float(c.violation(jnp.array([1.0, 2.0]))) == pytest.approx(3.0)


def test_rejects_unknown_kind():
    with pytest.raises(ValueError, match="kind must be one of"):
        Constraint(fn=lambda x: x, kind="approximate", violation=lambda x: x)
