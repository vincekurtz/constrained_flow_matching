import jax
import jax.numpy as jnp
from architectures.flow import FlowMLP
from flax import nnx

from training import loss_fn
import pytest

@pytest.fixture
def model():
    return FlowMLP(
        data_size=4,
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )

@pytest.fixture
def key():
    return jax.random.key(0)


def test_loss_fn_output_shape(model, key):
    """Loss function returns a scalar from a batch of data."""
    batch_size = 5
    data_size = 4
    key0, key1, key2 = jax.random.split(key, 3)
    x0 = jax.random.normal(key0, (batch_size, data_size))
    x1 = jax.random.normal(key1, (batch_size, data_size))
    t = jax.random.uniform(key2, (batch_size))

    loss = loss_fn(model, x1, x0, t)
    assert jnp.isscalar(loss)
