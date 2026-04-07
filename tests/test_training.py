import jax
import jax.numpy as jnp
from architectures.flow import FlowMLP
from datasets.bimodal_distribution import BimodalDataset
from flax import nnx
import optax

from training import loss_fn, train_step, train
import pytest


@pytest.fixture
def model():
    return FlowMLP(
        data_shape=(4,),
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def key():
    return jax.random.key(0)


@pytest.fixture
def optimizer(model):
    return nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)


def test_loss_fn(model, key):
    """Loss function returns a finite scalar from a batch of data."""
    batch_size = 5
    data_size = 4
    key0, key1, key2 = jax.random.split(key, 3)
    x0 = jax.random.normal(key0, (batch_size, data_size))
    x1 = jax.random.normal(key1, (batch_size, data_size))
    t = jax.random.uniform(key2, (batch_size))

    loss = loss_fn(model, x1, x0, t)
    assert jnp.isscalar(loss)
    assert jnp.isfinite(loss)


def test_train_step_returns_scalar(model, optimizer, key):
    """train_step returns a finite scalar loss."""
    batch = jax.random.normal(key, (5, 4))
    loss = train_step(model, optimizer, batch, key)
    assert jnp.isscalar(loss)
    assert jnp.isfinite(loss)
    assert loss >= 0.0, "Loss should be non-negative (mean squared error)"


def test_train_step_updates_parameters(model, optimizer, key):
    """train_step modifies model parameters in-place."""
    batch = jax.random.normal(key, (5, 4))

    # Capture parameter values before the step
    params_before = jax.tree.map(lambda x: x.copy(), nnx.state(model))

    train_step(model, optimizer, batch, key)

    params_after = nnx.state(model)
    any_changed = any(
        not jnp.array_equal(b, a)
        for b, a in zip(
            jax.tree.leaves(params_before), jax.tree.leaves(params_after)
        )
    )
    assert any_changed, "Model parameters were not updated after train_step"


def test_train_step_reduces_loss_over_iterations(model, optimizer, key):
    """Loss decreases after many train_step calls on a fixed batch."""
    batch = jax.random.normal(key, (32, 4))

    first_loss = train_step(model, optimizer, batch, key)
    for i in range(200):
        step_key = jax.random.fold_in(key, i)
        train_step(model, optimizer, batch, step_key)
    last_loss = train_step(model, optimizer, batch, key)

    assert last_loss < first_loss


def test_full_training():
    """Test the full training loop with a simple example."""
    dataset = BimodalDataset(num_samples=64)
    model = FlowMLP(
        data_shape=(2,),
        time_embedding_size=4,
        hidden_sizes=(8, 8),
        rngs=nnx.Rngs(0),
    )

    original_params = jax.tree.map(lambda x: x.copy(), nnx.state(model))

    trained_model, normalizer = train(
        dataset=dataset,
        model=model,
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-3,
        seed=0,
    )

    assert isinstance(trained_model, nnx.Module)

    trained_params = nnx.state(trained_model)
    any_changed = any(
        not jnp.array_equal(b, a)
        for b, a in zip(
            jax.tree.leaves(original_params), jax.tree.leaves(trained_params)
        )
    )
    assert any_changed, "Model parameters were not updated after training"

    # Normalizer should be in eval mode (parameters frozen) after training
    assert isinstance(normalizer, nnx.Module)

    # Normalizer stats should roughly match the data
    raw_data = jnp.array(dataset.data)
    normalized_data = normalizer(raw_data)
    assert jnp.allclose(
        jnp.mean(normalized_data, axis=0), 0.0, atol=0.5
    ), "Normalized data mean should be close to 0"
    assert jnp.allclose(
        jnp.std(normalized_data, axis=0), 1.0, atol=0.5
    ), "Normalized data std should be close to 1"
