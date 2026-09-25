"""Gauss-Newton projection."""

import jax
import jax.numpy as jnp
import pytest

from cfm.core.projection import (
    gauss_newton_project,
    gauss_newton_step,
    project_batch,
)


def circle(x):
    """Residual for the unit circle."""
    return jnp.atleast_1d(jnp.sum(x**2) - 1.0)


def right_half(x):
    """Inequality residual: violated when x[0] < 0."""
    return jnp.atleast_1d(-x[0])


def test_converges_onto_the_manifold():
    x = jnp.array([2.5, -1.7])
    out = gauss_newton_project(circle, x, num_iters=10)
    assert float(jnp.abs(circle(out))[0]) < 1e-6


def test_single_step_overshoots_near_the_origin():
    """One linearized step is not enough near the origin."""
    x = jnp.array([0.05, 0.0])
    one = gauss_newton_project(circle, x, num_iters=1)
    many = gauss_newton_project(circle, x, num_iters=25)
    assert jnp.abs(circle(one))[0] > jnp.abs(circle(many))[0]
    assert float(jnp.abs(circle(many))[0]) < 1e-5


def test_zero_iterations_is_a_no_op():
    x = jnp.array([3.0, 4.0])
    assert jnp.array_equal(gauss_newton_project(circle, x, num_iters=0), x)


def test_idempotent_on_the_manifold():
    x = jnp.array([1.0, 0.0])
    out = gauss_newton_project(circle, x, num_iters=5)
    assert jnp.allclose(out, x, atol=1e-6)


def test_active_set_leaves_feasible_points_untouched():
    x = jnp.array([2.0, 1.0])  # right_half(x) = -2 < 0, satisfied
    out = gauss_newton_project(right_half, x, num_iters=5, active_only=True)
    assert jnp.allclose(out, x, atol=1e-12)


def test_active_set_lands_on_the_boundary():
    """A violated inequality is projected onto h(x) = 0, not past it."""
    x = jnp.array([-1.5, 0.4])  # right_half(x) = 1.5 > 0, violated
    out = gauss_newton_project(right_half, x, num_iters=5, active_only=True)
    assert float(right_half(out)[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(out[1]) == pytest.approx(float(x[1]), abs=1e-9)


def test_unmasked_form_moves_feasible_points():
    """The equality form pulls onto the surface from either side."""
    x = jnp.array([2.0, 1.0])
    out = gauss_newton_project(right_half, x, num_iters=5, active_only=False)
    assert not jnp.allclose(out, x, atol=1e-6)


def test_step_matches_the_closed_form():
    """One step is exactly x - J^T (J J^T)^{-1} r."""
    x = jnp.array([2.0, 0.5])
    r = circle(x)
    J = jax.jacobian(circle)(x)
    expected = x - J.T @ jnp.linalg.solve(J @ J.T + 1e-10 * jnp.eye(1), r)
    assert jnp.allclose(gauss_newton_step(circle, x), expected, atol=1e-9)


def test_project_batch_preserves_shape():
    x = jax.random.normal(jax.random.key(0), (5, 3, 2))

    def residual(flat):
        return jnp.atleast_1d(jnp.sum(flat**2) - 1.0)

    out = project_batch(residual, x, num_iters=10)
    assert out.shape == x.shape
    norms = jnp.linalg.norm(out.reshape(5, -1), axis=-1)
    assert jnp.allclose(norms, 1.0, atol=1e-5)


def test_project_batch_zero_iters_is_a_no_op():
    x = jax.random.normal(jax.random.key(0), (4, 2))
    assert jnp.array_equal(project_batch(circle, x, num_iters=0), x)


def test_projection_is_jittable():
    fn = jax.jit(lambda x: gauss_newton_project(circle, x, num_iters=5))
    out = fn(jnp.array([2.0, 2.0]))
    assert float(jnp.abs(circle(out))[0]) < 1e-6
