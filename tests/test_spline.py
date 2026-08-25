"""The two Bezier spline implementations must agree.

``cfm.core.spline.bezier_spline`` (jax, used by the constraint) and
``cfm.datasets.obstacle_paths.bezier_spline`` (torch, used to build the
training data) are separate implementations of the same curve. Only the torch
one was covered before, so a divergence between them would have meant the
model was trained on different paths than the constraint was imposed on.
"""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from cfm.core.spline import bezier_spline as bezier_jax
from cfm.datasets.obstacle_paths import bezier_spline as bezier_torch

KNOTS = np.array([
    [-1.0, 0.0], [-0.6, 0.4], [-0.1, -0.3], [0.4, 0.5], [1.0, 0.0],
])

# Every distinct num_sub is a separate XLA compilation, so the tests below
# share these two: 1 is the degenerate case where the curve is just its
# knots, 8 is the general one. Only test_is_smooth needs a denser sampling.
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
    """The curve passes through the knots, not merely near them."""
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
    """A leading batch dimension is carried through."""
    batch = jnp.stack([jnp.array(KNOTS), jnp.array(KNOTS) * 0.5])
    out = bezier_jax(batch, 8)
    assert out.shape == (2, 8 * (len(KNOTS) - 1) + 1, 2)
    single = bezier_jax(jnp.array(KNOTS), 8)
    np.testing.assert_allclose(
        np.asarray(out[0]), np.asarray(single), atol=1e-6
    )
