import jax
import jax.numpy as jnp

from architectures.normalizer import Normalizer


def _make_batches(key, shape, true_mean, true_std, num_batches=300):
    """Generate batches of data with a known distribution."""
    batches = []
    for _ in range(num_batches):
        key, subkey = jax.random.split(key)
        batch = jax.random.normal(subkey, shape) * true_std + true_mean
        batches.append(batch)
    return batches


def test_stats_track_data_distribution():
    """Mean and std are approximately correct after fitting."""
    true_mean = jnp.array([2.0, -1.0, 5.0])
    true_std = jnp.array([0.5, 2.0, 1.5])

    key = jax.random.key(0)
    batches = _make_batches(key, (64, 3), true_mean, true_std)

    normalizer = Normalizer.from_dataloader(batches)

    assert jnp.allclose(normalizer.mean, true_mean, atol=0.1)
    assert jnp.allclose(normalizer.std, true_std, atol=0.1)


def test_stats_track_image_data():
    """Mean and std are correct for image-shaped data."""
    true_mean = jnp.ones((4, 4, 3)) * 5.0
    true_std = jnp.ones((4, 4, 3)) * 2.0

    key = jax.random.key(0)
    batches = _make_batches(key, (32, 4, 4, 3), true_mean, true_std, num_batches=100)

    normalizer = Normalizer.from_dataloader(batches)

    assert jnp.allclose(normalizer.mean, true_mean, atol=0.2)
    assert jnp.allclose(normalizer.std, true_std, atol=0.2)


def test_normalize_and_unnormalize_are_inverse():
    """unnormalize(normalize(x)) ≈ x."""
    true_mean = jnp.array([3.0, -2.0])
    true_std = jnp.array([1.5, 0.5])

    key = jax.random.key(2)
    batches = _make_batches(key, (64, 2), true_mean, true_std, num_batches=100)

    normalizer = Normalizer.from_dataloader(batches)

    x = batches[0]
    x_reconstructed = normalizer.unnormalize(normalizer.normalize(x))
    assert jnp.allclose(x_reconstructed, x, atol=1e-5)


def test_normalized_data_has_zero_mean_unit_var():
    """Normalized training data has approximately zero mean, unit variance."""
    true_mean = jnp.array([10.0, -5.0, 3.0])
    true_std = jnp.array([2.0, 0.5, 4.0])

    key = jax.random.key(3)
    batches = _make_batches(key, (64, 3), true_mean, true_std, num_batches=200)

    normalizer = Normalizer.from_dataloader(batches)

    all_data = jnp.concatenate(batches, axis=0)
    normalized = normalizer.normalize(all_data)
    assert jnp.allclose(jnp.mean(normalized, axis=0), 0.0, atol=0.1)
    assert jnp.allclose(jnp.std(normalized, axis=0), 1.0, atol=0.1)
