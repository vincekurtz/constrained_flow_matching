from typing import Tuple

import jax
import jax.numpy as jnp

from architectures.normalizer import Normalizer


def generate(
    model,
    normalizer: Normalizer,
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 42,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow-matching model.

    Integrates the learned vector field from t=0 to t=1 using forward Euler,
    then unnormalizes the result.

    Args:
        model: Trained flow model xdot = v(x, t).
        normalizer: Normalizer used during training, applied in reverse to
            produced samples in the original data space.
        num_samples: Number of samples to generate.
        dt: Step size for the forward Euler integrator.
        seed: Random seed for the initial noise.

    Returns:
        x: Final generated samples of shape (num_samples, *data_shape).
        xs: Full trajectories of shape (num_steps, num_samples, *data_shape).
    """
    rng = jax.random.key(seed)
    x = jax.random.normal(rng, (num_samples,) + model.data_shape)

    def _step_fn(x, t):
        """Single forward Euler step on the flow ODE xdot = v(x, t)."""
        t_batch = jnp.full((x.shape[0],), t)
        x_next = x + dt * model(x, t_batch)
        return x_next, x_next

    timesteps = jnp.arange(0, 1.0, dt)
    x, xs = jax.lax.scan(_step_fn, x, timesteps)

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)
