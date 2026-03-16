import jax
import jax.numpy as jnp
import pytest

from architectures.mlp import MLP


@pytest.fixture
def key():
    return jax.random.key(0)


def test_output_shape(key):
    """Output shape matches output_size for a single input."""
    model = MLP(output_size=3, hidden_sizes=(8,))
    x = jnp.ones((4,))
    params = model.init(key, x)
    y = model.apply(params, x)
    assert y.shape == (3,)


def test_batched_output_shape(key):
    """Output shape matches (batch, output_size) for batched input."""
    model = MLP(output_size=3, hidden_sizes=(8,))
    x = jnp.ones((16, 4))
    params = model.init(key, x)
    y = model.apply(params, x)
    assert y.shape == (16, 3)


def test_no_hidden_layers(key):
    """Network with no hidden layers acts as a single linear map."""
    model = MLP(output_size=2, hidden_sizes=())
    x = jnp.ones((4,))
    params = model.init(key, x)
    y = model.apply(params, x)
    assert y.shape == (2,)


def test_multiple_hidden_layers(key):
    """Network with multiple hidden layers produces correct output shape."""
    model = MLP(output_size=1, hidden_sizes=(64, 64, 32))
    x = jnp.ones((8,))
    params = model.init(key, x)
    y = model.apply(params, x)
    assert y.shape == (1,)


def test_layer_count(key):
    """Number of Dense layers equals len(hidden_sizes) + 1."""
    hidden_sizes = (16, 16)
    model = MLP(output_size=2, hidden_sizes=hidden_sizes)
    x = jnp.ones((4,))
    params = model.init(key, x)
    assert len(params["params"]) == len(hidden_sizes) + 1


def test_output_is_finite(key):
    """Forward pass produces finite values for standard inputs."""
    model = MLP(output_size=3, hidden_sizes=(32, 32))
    x = jnp.ones((6,))
    params = model.init(key, x)
    y = model.apply(params, x)
    assert jnp.all(jnp.isfinite(y))
