import jax
import jax.numpy as jnp
from flax import nnx

from architectures.normalizer import Normalizer


def test_running_stats_track_data_distribution():
    """Running mean and std are approximately correct after batched updates."""
    data_size = 3
    true_mean = jnp.array([2.0, -1.0, 5.0])
    true_std = jnp.array([0.5, 2.0, 1.5])

    normalizer = Normalizer(data_size=data_size, rngs=nnx.Rngs(0))

    key = jax.random.key(0)
    for _ in range(300):
        key, subkey = jax.random.split(key)
        batch = (
            jax.random.normal(subkey, (64, data_size)) * true_std + true_mean
        )
        normalizer(batch)

    assert jnp.allclose(normalizer.batch_norm.mean[...], true_mean, atol=0.1)
    assert jnp.allclose(
        jnp.sqrt(normalizer.batch_norm.var[...]), true_std, atol=0.1
    )


def test_stats_not_updated_in_eval_mode():
    """Running stats are frozen when the normalizer is in eval mode."""
    data_size = 3
    normalizer = Normalizer(data_size=data_size, rngs=nnx.Rngs(0))

    # Warm up with a few batches so stats are non-trivial
    key = jax.random.key(1)
    for _ in range(50):
        key, subkey = jax.random.split(key)
        normalizer(jax.random.normal(subkey, (64, data_size)))

    mean_before = normalizer.batch_norm.mean[...]
    var_before = normalizer.batch_norm.var[...]

    # Feed very different data in eval mode
    normalizer.eval()
    for _ in range(50):
        key, subkey = jax.random.split(key)
        normalizer(jax.random.normal(subkey, (64, data_size)) * 100.0 + 50.0)

    assert jnp.array_equal(normalizer.batch_norm.mean[...], mean_before)
    assert jnp.array_equal(normalizer.batch_norm.var[...], var_before)


def test_no_learnable_parameters():
    """Normalizer has no learnable (Param) variables."""
    normalizer = Normalizer(data_size=4, rngs=nnx.Rngs(0))
    params = nnx.state(normalizer, nnx.Param)
    assert len(nnx.to_flat_state(params)) == 0
