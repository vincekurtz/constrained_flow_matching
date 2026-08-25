"""Forward-pass behavior of the two model architectures.

Building and running a model costs an XLA compilation, and for the UNet that
dominates the whole suite. So each architecture is built once per session at
one input shape, and every assertion below reuses it: sensitivity checks put
the two inputs they want to compare in a single batch rather than making a
second call at a new shape.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from cfm.models.flow import FlowMLP, SinusoidalPosEmb
from cfm.models.unet import FlowUNet

BATCH = 4


# ---------------------------------------------------------------------------
# SinusoidalPosEmb
# ---------------------------------------------------------------------------


def test_sinusoidal_output_shape():
    """Output shape is (batch, dim) for a batch of scalar time values."""
    out = SinusoidalPosEmb(dim=16)(jnp.linspace(0, 1, 8))
    assert out.shape == (8, 16)


def test_sinusoidal_values_are_bounded_and_finite():
    """Sines and cosines, so every value lies in [-1, 1]."""
    out = SinusoidalPosEmb(dim=16)(jnp.linspace(0, 1, 100))
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out >= -1.0) and jnp.all(out <= 1.0)


def test_sinusoidal_distinguishes_times():
    """Different time values produce different embeddings."""
    out = SinusoidalPosEmb(dim=16)(jnp.array([0.0, 1.0]))
    assert not jnp.allclose(out[0], out[1])


# ---------------------------------------------------------------------------
# FlowMLP
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def mlp():
    return FlowMLP(
        data_shape=(4,),
        time_embedding_size=16,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(0),
    )


def test_flowmlp_output_matches_input_shape(mlp):
    y = mlp(jnp.ones((BATCH, 4)), jnp.linspace(0, 1, BATCH))
    assert y.shape == (BATCH, 4)
    assert jnp.all(jnp.isfinite(y))


def test_flowmlp_is_sensitive_to_time(mlp):
    """The same x at different t must give different velocities."""
    y = mlp(jnp.ones((BATCH, 4)), jnp.linspace(0, 1, BATCH))
    assert not jnp.allclose(y[0], y[-1])


def test_flowmlp_is_sensitive_to_input(mlp):
    """Different x at the same t must give different velocities."""
    x = jnp.stack([jnp.zeros(4), jnp.ones(4), jnp.full((4,), 2.0),
                   jnp.full((4,), 3.0)])
    y = mlp(x, jnp.full((BATCH,), 0.5))
    assert not jnp.allclose(y[0], y[1])


def test_flowmlp_handles_image_shaped_data():
    """Flattening and unflattening round-trips for multi-axis data."""
    model = FlowMLP(
        data_shape=(4, 4, 2),
        time_embedding_size=16,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((BATCH, 4, 4, 2))
    y = model(x, jnp.linspace(0, 1, BATCH))
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# FlowUNet
#
# Three resolution levels, matching the shape of the MNIST configuration in
# problems/mnist.py, but at the smallest size that still downsamples twice.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def unet():
    return FlowUNet(
        data_shape=(8, 8, 1),
        time_embedding_size=16,
        channels=(8, 16, 32),
        rngs=nnx.Rngs(0),
    )


def test_flowunet_output_matches_input_shape(unet):
    """Down- and up-sampling must land back on the input resolution."""
    x = jnp.ones((BATCH, 8, 8, 1))
    y = unet(x, jnp.linspace(0, 1, BATCH))
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_flowunet_is_sensitive_to_time(unet):
    y = unet(jnp.ones((BATCH, 8, 8, 1)), jnp.linspace(0, 1, BATCH))
    assert not jnp.allclose(y[0], y[-1])


def test_flowunet_is_sensitive_to_input(unet):
    x = jnp.stack([jnp.full((8, 8, 1), float(i)) for i in range(BATCH)])
    y = unet(x, jnp.full((BATCH,), 0.5))
    assert not jnp.allclose(y[0], y[1])
