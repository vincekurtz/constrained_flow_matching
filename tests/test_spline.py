"""Bezier splines: the jax (constraint) and torch (dataset) versions agree."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from cfm.core.spline import bezier_spline as bezier_jax
from cfm.datasets.obstacle_paths import bezier_spline as bezier_torch

KNOTS = np.array([
    [-1.0, 0.0], [-0.6, 0.4], [-0.1, -0.3], [0.4, 0.5], [1.0, 0.0],
])

# Each distinct num_sub is a separate compilation, so keep this list short.
SUBDIVISIONS = [1, 8]


@pytest.mark.parametrize("num_sub", SUBDIVISIONS)
def test_implementations_agree(num_sub):
    got = np.asarray(bezier_jax(jnp.array(KNOTS), num_sub))
    want = bezier_torch(torch.tensor(KNOTS), num_sub).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


def test_implementations_agree_on_a_batch():
    batch = np.stack([KNOTS, KNOTS * 0.5, KNOTS + 0.2])
    got = np.asarray(bezier_jax(jnp.array(batch), 8))
    want = bezier_torch(torch.tensor(batch), 8).numpy()
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("num_sub", SUBDIVISIONS)
def test_output_shape(num_sub):
    out = bezier_jax(jnp.array(KNOTS), num_sub)
    assert out.shape == (num_sub * (len(KNOTS) - 1) + 1, 2)


def test_interpolates_every_knot():
    num_sub = 8
    out = np.asarray(bezier_jax(jnp.array(KNOTS), num_sub))
    for i, knot in enumerate(KNOTS):
        np.testing.assert_allclose(out[i * num_sub], knot, atol=1e-5)


def test_endpoints_are_pinned():
    out = bezier_jax(jnp.array(KNOTS), 8)
    np.testing.assert_allclose(np.asarray(out[0]), KNOTS[0], atol=1e-6)
    np.testing.assert_allclose(np.asarray(out[-1]), KNOTS[-1], atol=1e-6)


def test_is_smooth():
    """No kinks: consecutive segment directions change gradually."""
    out = np.asarray(bezier_jax(jnp.array(KNOTS), 40))
    deltas = np.diff(out, axis=0)
    lengths = np.linalg.norm(deltas, axis=-1)
    assert np.all(lengths > 0)
    units = deltas / lengths[:, None]
    turns = np.sum(units[1:] * units[:-1], axis=-1)
    assert np.all(turns > 0.9), f"sharpest turn cos = {turns.min():.4f}"


def test_batched_knots():
    batch = jnp.stack([jnp.array(KNOTS), jnp.array(KNOTS) * 0.5])
    out = bezier_jax(batch, 8)
    assert out.shape == (2, 8 * (len(KNOTS) - 1) + 1, 2)
    single = bezier_jax(jnp.array(KNOTS), 8)
    np.testing.assert_allclose(
        np.asarray(out[0]), np.asarray(single), atol=1e-6
    )
