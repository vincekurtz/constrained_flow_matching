import jax.numpy as jnp
from flax import nnx

from architectures.mlp import MLP


def test_output_shape():
    """Output shape matches output_size for a single input."""
    model = MLP(
        input_size=4, output_size=3, hidden_sizes=(8,), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((4,))
    y = model(x)
    assert y.shape == (3,)


def test_batched_output_shape():
    """Output shape matches (batch, output_size) for batched input."""
    model = MLP(
        input_size=4, output_size=3, hidden_sizes=(8,), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((16, 4))
    y = model(x)
    assert y.shape == (16, 3)


def test_no_hidden_layers():
    """Network with no hidden layers acts as a single linear map."""
    model = MLP(input_size=4, output_size=2, hidden_sizes=(), rngs=nnx.Rngs(0))
    x = jnp.ones((4,))
    y = model(x)
    assert y.shape == (2,)


def test_multiple_hidden_layers():
    """Network with multiple hidden layers produces correct output shape."""
    model = MLP(
        input_size=8, output_size=1, hidden_sizes=(64, 64, 32), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((8,))
    y = model(x)
    assert y.shape == (1,)


def test_layer_count():
    """Number of Dense layers equals len(hidden_sizes) + 1."""
    hidden_sizes = (16, 16)
    model = MLP(
        input_size=4, output_size=2, hidden_sizes=hidden_sizes, rngs=nnx.Rngs(0)
    )
    assert len(model.layers) == len(hidden_sizes) + 1


def test_output_is_finite():
    """Forward pass produces finite values for standard inputs."""
    model = MLP(
        input_size=6, output_size=3, hidden_sizes=(32, 32), rngs=nnx.Rngs(0)
    )
    x = jnp.ones((6,))
    y = model(x)
    assert jnp.all(jnp.isfinite(y))
