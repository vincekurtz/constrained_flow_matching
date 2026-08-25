"""The plain MLP used as a building block by FlowMLP.

Each distinct layer configuration is a separate XLA compilation, so the
shape checks share one model; only the degenerate no-hidden-layers case
needs a second.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from cfm.models.mlp import MLP


@pytest.fixture(scope="module")
def mlp():
    return MLP(
        input_size=4, output_size=3, hidden_sizes=(8, 8), rngs=nnx.Rngs(0)
    )


def test_output_shape(mlp):
    """Output shape matches output_size for a single input."""
    y = mlp(jnp.ones((4,)))
    assert y.shape == (3,)
    assert jnp.all(jnp.isfinite(y))


def test_batched_output_shape(mlp):
    """Output shape matches (batch, output_size) for batched input."""
    assert mlp(jnp.ones((16, 4))).shape == (16, 3)


def test_layer_count(mlp):
    """Number of Dense layers equals len(hidden_sizes) + 1."""
    assert len(mlp.layers) == 3


def test_no_hidden_layers():
    """Network with no hidden layers acts as a single linear map."""
    model = MLP(input_size=4, output_size=2, hidden_sizes=(), rngs=nnx.Rngs(0))
    assert len(model.layers) == 1
    assert model(jnp.ones((4,))).shape == (2,)
