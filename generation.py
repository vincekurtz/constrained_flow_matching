from typing import Callable, Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from architectures.normalizer import Normalizer


def generate(
    model: nnx.Module,
    normalizer: Normalizer,
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
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

def generate_constrained(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    penalty_weight: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow model subject to g(x) = 0.

    Uses an analytical Lagrange multiplier to project the learned velocity onto
    the constraint tangent space, plus a penalty term that pulls samples toward
    the constraint manifold.

    Args:
        model: Trained flow model xdot = v(x, t).
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable function ``g(x)`` (operating on a single
            *unnormalized* sample) whose zero-level set defines the constraint
            manifold.  May return a scalar or a 1-D array.
        num_samples: Number of samples to generate.
        dt: Step size for the forward Euler integrator.
        seed: Random seed for the initial noise.
        penalty_weight: Strength of the quadratic penalty pulling samples
            toward the constraint manifold.

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
    """
    rng = jax.random.key(seed)
    x = jax.random.normal(rng, (num_samples,) + model.data_shape)

    def _g(x_i):
        """Constraint function applied to a single normalized sample."""
        x_i = normalizer.unnormalize(x_i)
        g_i = constraint_fn(x_i)
        return penalty_weight * g_i

    def _constrain_velocity(x_i, v_i):
        """Project one sample's velocity onto the constraint tangent space."""
        g = jnp.atleast_1d(_g(x_i))
        J = jnp.atleast_2d(jax.jacobian(_g)(x_i))

        # Project: remove component along constraint gradient via analytical
        # Lagrange multiplier. 
        JJT = J @ J.T + 1e-6 * jnp.eye(g.shape[0])
        lmbda = jnp.linalg.solve(JJT, J @ v_i)
        v_proj = v_i - J.T @ lmbda

        # Penalty: pull toward the constraint manifold via a quadratic penalty
        # on ||g(x)||^2.
        v_proj = v_proj - J.T @ g

        return v_proj

    def _step_fn(x, t):
        """Single forward Euler step with constraint projection."""
        # Get the original vector field xdot = v(x, t)
        t_batch = jnp.full((x.shape[0],), t)
        v = model(x, t_batch)

        # Apply per-sample constraint correction,
        # xdot = v(x, t) - λ' v(x, t) - ∇ ||g(x)||²
        v = jax.vmap(_constrain_velocity)(x, v)

        x_next = x + dt * v
        return x_next, x_next

    timesteps = jnp.arange(0, 1.0, dt)
    x, xs = jax.lax.scan(_step_fn, x, timesteps)

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)
