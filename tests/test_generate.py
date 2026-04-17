import jax.numpy as jnp
import pytest
from flax import nnx

from architectures.flow import FlowMLP
from architectures.normalizer import Normalizer
from generation import generate


@pytest.fixture
def model_2d():
    return FlowMLP(
        data_shape=(2,),
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def model_4d():
    return FlowMLP(
        data_shape=(4,),
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def identity_normalizer():
    """Normalizer with mean=0 and std=1 (no-op)."""
    return Normalizer(mean=jnp.zeros(2), std=jnp.ones(2))


@pytest.fixture
def normalizer_4d():
    return Normalizer(mean=jnp.zeros(4), std=jnp.ones(4))


def test_generate_output_shapes(model_2d, identity_normalizer):
    """generate returns (x, xs) with correct shapes."""
    num_samples = 10
    dt = 0.1
    x, xs = generate(
        model_2d, identity_normalizer, num_samples=num_samples, dt=dt
    )

    num_steps = int(1.0 / dt)
    assert x.shape == (num_samples, 2)
    assert xs.shape == (num_steps, num_samples, 2)


def test_generate_output_shapes_4d(model_4d, normalizer_4d):
    """generate works for higher-dimensional data shapes."""
    num_samples = 5
    dt = 0.2
    x, xs = generate(model_4d, normalizer_4d, num_samples=num_samples, dt=dt)

    num_steps = int(1.0 / dt)
    assert x.shape == (num_samples, 4)
    assert xs.shape == (num_steps, num_samples, 4)


def test_generate_output_is_finite(model_2d, identity_normalizer):
    """All generated values are finite."""
    x, xs = generate(model_2d, identity_normalizer, num_samples=20, dt=0.05)
    assert jnp.all(jnp.isfinite(x))
    assert jnp.all(jnp.isfinite(xs))


def test_generate_deterministic(model_2d, identity_normalizer):
    """Same seed produces identical outputs."""
    x1, xs1 = generate(
        model_2d, identity_normalizer, num_samples=10, dt=0.1, seed=7
    )
    x2, xs2 = generate(
        model_2d, identity_normalizer, num_samples=10, dt=0.1, seed=7
    )
    assert jnp.array_equal(x1, x2)
    assert jnp.array_equal(xs1, xs2)


def test_generate_different_seeds(model_2d, identity_normalizer):
    """Different seeds produce different outputs."""
    x1, _ = generate(
        model_2d, identity_normalizer, num_samples=10, dt=0.1, seed=0
    )
    x2, _ = generate(
        model_2d, identity_normalizer, num_samples=10, dt=0.1, seed=1
    )
    assert not jnp.array_equal(x1, x2)


def test_generate_normalizer_applied(model_2d):
    """Normalizer unnormalization is applied: shifting mean moves output."""
    mean_shift = jnp.array([100.0, 200.0])
    shifted_normalizer = Normalizer(mean=mean_shift, std=jnp.ones(2))
    identity_normalizer = Normalizer(mean=jnp.zeros(2), std=jnp.ones(2))

    x_shifted, _ = generate(
        model_2d, shifted_normalizer, num_samples=50, dt=0.1
    )
    x_identity, _ = generate(
        model_2d, identity_normalizer, num_samples=50, dt=0.1
    )

    # Shifted output should have a noticeably larger mean along both axes
    assert jnp.mean(x_shifted[:, 0]) > jnp.mean(x_identity[:, 0]) + 50
    assert jnp.mean(x_shifted[:, 1]) > jnp.mean(x_identity[:, 1]) + 50


def test_generate_trajectory_endpoint_matches_final_sample(
    model_2d, identity_normalizer
):
    """The last trajectory step matches the returned final samples."""
    x, xs = generate(model_2d, identity_normalizer, num_samples=10, dt=0.1)
    assert jnp.array_equal(x, xs[-1])
