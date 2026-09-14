import jax
import jax.numpy as jnp
from cfm.models.flow import FlowMLP
from cfm.datasets.bimodal_distribution import BimodalDataset
from flax import nnx
import optax

from cfm.training import (
    ema_update,
    loss_fn,
    make_schedule,
    train,
    train_step,
)
from cfm.models.normalizer import Normalizer
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

    # Normalizer should have valid stats after training
    assert isinstance(normalizer, Normalizer)

    # Normalizer stats should roughly match the data
    raw_data = jnp.array(dataset.data)
    normalized_data = normalizer.normalize(raw_data)
    assert jnp.allclose(
        jnp.mean(normalized_data, axis=0), 0.0, atol=0.5
    ), "Normalized data mean should be close to 0"
    assert jnp.allclose(
        jnp.std(normalized_data, axis=0), 1.0, atol=0.5
    ), "Normalized data std should be close to 1"


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------


def test_constant_schedule_never_moves():
    schedule = make_schedule(1e-3, num_steps=100, schedule="constant")
    assert schedule(0) == pytest.approx(1e-3)
    assert schedule(99) == pytest.approx(1e-3)


def test_cosine_schedule_warms_up_then_decays():
    """Starts at zero, peaks at the requested rate, ends far below it."""
    schedule = make_schedule(1e-3, num_steps=1000, schedule="cosine")
    rates = jnp.array([schedule(i) for i in range(1000)])
    assert rates[0] == pytest.approx(0.0, abs=1e-9)
    assert float(rates.max()) == pytest.approx(1e-3, rel=1e-3)
    assert float(rates[-1]) < 0.1 * 1e-3


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError, match="unknown schedule"):
        make_schedule(1e-3, num_steps=10, schedule="linear")


# ---------------------------------------------------------------------------
# Parameter averaging
# ---------------------------------------------------------------------------


def test_ema_update_moves_toward_the_new_parameters():
    averaged = {"w": jnp.zeros(3)}
    params = {"w": jnp.ones(3)}
    once = ema_update(averaged, params, 0.9)
    twice = ema_update(once, params, 0.9)
    assert jnp.allclose(once["w"], 0.1)
    assert jnp.all(twice["w"] > once["w"])
    assert jnp.all(twice["w"] < 1.0)


def test_training_with_ema_returns_the_average_not_the_last_iterate():
    """The returned weights must differ from the ones the last step left."""
    dataset = BimodalDataset(num_samples=64)

    def run(ema_decay):
        model = FlowMLP(
            data_shape=(2,),
            time_embedding_size=4,
            hidden_sizes=(8, 8),
            rngs=nnx.Rngs(0),
        )
        trained, _ = train(
            dataset=dataset,
            model=model,
            num_epochs=5,
            batch_size=16,
            learning_rate=1e-2,
            seed=0,
            ema_decay=ema_decay,
        )
        return jax.tree.leaves(nnx.state(trained, nnx.Param))

    last, averaged = run(None), run(0.9)
    assert all(jnp.all(jnp.isfinite(p)) for p in averaged)
    assert any(
        not jnp.allclose(a, b) for a, b in zip(last, averaged)
    ), "EMA weights should not equal the final iterate"


def test_training_with_a_cosine_schedule_still_learns():
    dataset = BimodalDataset(num_samples=64)
    model = FlowMLP(
        data_shape=(2,),
        time_embedding_size=4,
        hidden_sizes=(8, 8),
        rngs=nnx.Rngs(0),
    )
    before = jax.tree.map(lambda x: x.copy(), nnx.state(model))
    trained, _ = train(
        dataset=dataset,
        model=model,
        num_epochs=5,
        batch_size=16,
        learning_rate=1e-2,
        seed=0,
        schedule="cosine",
    )
    after = nnx.state(trained)
    assert any(
        not jnp.array_equal(b, a)
        for b, a in zip(jax.tree.leaves(before), jax.tree.leaves(after))
    )
