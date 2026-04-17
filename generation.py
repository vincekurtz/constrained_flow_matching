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
    penalty_weight: float = 5.0,
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
        data_shape = x_i.shape
        x_flat = x_i.ravel()
        v_flat = v_i.ravel()

        def _g_flat(xf):
            return _g(xf.reshape(data_shape))

        g = jnp.atleast_1d(_g_flat(x_flat))
        J = jnp.atleast_2d(jax.jacobian(_g_flat)(x_flat))

        # Project: remove component along constraint gradient via analytical
        # Lagrange multiplier.
        JJT = J @ J.T + 1e-6 * jnp.eye(g.shape[0])
        lmbda = jnp.linalg.solve(JJT, J @ v_flat)
        v_proj = v_flat - J.T @ lmbda

        # Penalty: pull toward the constraint manifold via a quadratic penalty
        # on ||g(x)||^2.
        v_proj = v_proj - J.T @ g

        return v_proj.reshape(data_shape)

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


def generate_constrained_inverse_free(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    penalty_weight: float = 5.0,
    rescale_factor: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow model subject to g(x) = 0.

    Uses a flowed Lagrange multiplier to avoid the need for a Jacobian
    pseudoinverse.  At each time-step the constraint value ``g`` and its
    Jacobian ``J`` are computed once per sample and reused for both the
    multiplier update and the velocity projection.

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
        rescale_factor: Factor by which to rescale the time for the Lagrange
            multiplier flow. This can help enforce the constraint more strictly
            but leads to a stiffer ODE.

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
    """
    rng = jax.random.key(seed)

    def _g(x_i):
        """Constraint on a single normalized sample (any shape)."""
        return constraint_fn(normalizer.unnormalize(x_i))

    def _g_flat(x_flat, data_shape):
        """Constraint on a flattened normalized sample."""
        return jnp.atleast_1d(_g(x_flat.reshape(data_shape)))

    def _step_single(x_i, v_i, lmbda_i, t):
        """Per-sample step: compute g/J once, update lambda and project v."""
        data_shape = x_i.shape
        x_flat = x_i.ravel()
        v_flat = v_i.ravel()

        g_flat = lambda xf: _g_flat(xf, data_shape)
        g = g_flat(x_flat)
        J = jnp.atleast_2d(jax.jacobian(g_flat)(x_flat))

        # Flow the Lagrange multiplier (Platt & Barr 1987).
        dt_lmbda = rescale_factor * dt / (1 - t + 1e-8)
        lmbda_next = lmbda_i + dt_lmbda * g

        # Project velocity: remove constraint-normal component and add
        # penalty pulling toward the manifold.
        correction = lmbda_next.T @ J + penalty_weight * g.T @ J
        v_proj = (v_flat - correction).reshape(data_shape)

        x_next = x_i + dt * v_proj
        return x_next, lmbda_next

    def _step_fn(carry, t):
        """Batched forward Euler step with constraint projection."""
        x, lmbda = carry

        # Evaluate the learned vector field.
        t_batch = jnp.full((x.shape[0],), t)
        v = model(x, t_batch)

        # Per-sample constraint correction (vmapped).
        x_next, lmbda_next = jax.vmap(
            _step_single, in_axes=(0, 0, 0, None)
        )(x, v, lmbda, t)

        return (x_next, lmbda_next), x_next

    x_init = jax.random.normal(rng, (num_samples,) + model.data_shape)

    # Initialise multipliers from the constraint value at the starting noise.
    lmbda_init = jax.vmap(lambda xi: jnp.atleast_1d(_g(xi)))(x_init)
    lmbda_init = penalty_weight * lmbda_init

    timesteps = jnp.arange(0, 1.0, dt)
    (x, lmbda), xs = jax.lax.scan(_step_fn, (x_init, lmbda_init), timesteps)

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)