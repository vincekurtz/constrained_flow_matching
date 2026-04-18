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
    method: str = "flow",
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    penalty_weight: float = 5.0,
    rescale_factor: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow model subject to g(x) = 0.

    Args:
        model: Trained flow model xdot = v(x, t). Must have a ``data_shape``
            attribute.
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable function ``g(x)`` (operating on a single
            *unnormalized* sample) whose zero-level set defines the constraint
            manifold.  May return a scalar or a 1-D array.
        method: How to compute the Lagrange multipliers. Must be one of
            - "pseudoinverse": analytical solution via Jacobian pseudoinverse.
            - "flow": approximate solution via flowing a dual ODE.
            - "penalty": quadratic penalty only.
        num_samples: Number of samples to generate.
        dt: Step size for the forward Euler integrator.
        seed: Random seed for the initial noise.
        penalty_weight: Strength of the quadratic penalty pulling samples
            toward the constraint manifold.
        rescale_factor: Factor by which to rescale the time for the Lagrange
            multiplier flow. This can help enforce the constraint more strictly
            but leads to a stiffer ODE. Only used in "flow" method.

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
    """
    rng = jax.random.key(seed)
    data_shape = model.data_shape

    def _g(x):
        """Constraint g(x), applied to a single flattened normalized sample."""
        x = x.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(penalty_weight * constraint_fn(x))

    def _step_single(x, v, lmbda, t):
        """Perform an integration step for a single sample.

        Args:
            x: Current state (generated sample).
            v: Unconstrained vector field v(x, t).
            lmbda: Current Lagrange multiplier.
            t: Current time.

        Returns:
            State after the integration step.
            Lagrange multiplier after the integration step.
        """
        x_flat = x.ravel()
        v_flat = v.ravel()

        g = _g(x_flat)
        J = jnp.atleast_2d(jax.jacobian(_g)(x_flat))

        if method == "flow":
            # Flow the Lagrange multiplier (Platt & Barr 1987), but use
            # rescaled time s = -ρ log(1 - t) to reach s = ∞ at t = 1.
            dt_lmbda = rescale_factor * dt / (1 - t + 1e-8)
            lmbda = lmbda + dt_lmbda * g
        elif method == "pseudoinverse":
            # Analytical solution via Jacobian pseudoinverse.
            JJT = J @ J.T + 1e-6 * jnp.eye(g.shape[0])
            lmbda = jnp.linalg.solve(JJT, J @ v_flat)
        elif method == "penalty":
            lmbda = jnp.zeros_like(g)
        else:
            raise ValueError(f"Invalid lagrange multiplier method: {method}")

        # Constrained flow: ẋ = v(x, t) − Jᵀλ − ∇‖g(x)‖²/2
        x_dot = v_flat - J.T @ lmbda - J.T @ g

        # Forward euler integration step.
        x = x + dt * x_dot.reshape(data_shape)
        return x, lmbda

    def _step_fn(carry, t):
        """Batched forward Euler step with constraint projection."""
        x, lmbda = carry

        # Evaluate the learned vector field.
        t_batch = jnp.full((x.shape[0],), t)
        v = model(x, t_batch)

        # Integrate the state and langrange multiplier for each sample in the
        # batch, following a constrained/corrected ODE.
        x_next, lmbda_next = jax.vmap(_step_single, in_axes=(0, 0, 0, None))(
            x, v, lmbda, t
        )

        return (x_next, lmbda_next), x_next

    # Data samples are initialized as Gaussian noise.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    # Initialise multipliers as zeros of the correct shape
    lmbda_init = 0.0 * jax.vmap(lambda xi: _g(xi.ravel()))(x_init)

    # Integrate the constrained flow ODE from t=0 to t=1.
    timesteps = jnp.arange(0, 1.0, dt)
    (x, _), xs = jax.lax.scan(_step_fn, (x_init, lmbda_init), timesteps)

    # All trajectories are in normalized space, so unnormalize before returning.
    return normalizer.unnormalize(x), normalizer.unnormalize(xs)


def generate_inequality_constrained(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    method: str = "flow",
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    penalty_weight: float = 5.0,
    rescale_factor: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow model subject to h(x) <= 0.

    Introduces slack variables s <= 0 with trivial dynamics (s_dot = 0) and
    enforces h(x) = s as an equality constraint on the augmented state (x, s).

    Args:
        model: Trained flow model xdot = v(x, t). Must have a ``data_shape``
            attribute.
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable function ``h(x)`` (operating on a single
            *unnormalized* sample) defining the inequality ``h(x) <= 0``.
            May return a scalar or a 1-D array.
        method: How to compute the Lagrange multipliers. Must be one of
            - "pseudoinverse": analytical solution via Jacobian pseudoinverse.
            - "flow": approximate solution via flowing a dual ODE.
            - "penalty": quadratic penalty only.
        num_samples: Number of samples to generate.
        dt: Step size for the forward Euler integrator.
        seed: Random seed for the initial noise.
        penalty_weight: Strength of the quadratic penalty pulling samples
            toward the constraint manifold.
        rescale_factor: Factor by which to rescale the time for the Lagrange
            multiplier flow. Only used in "flow" method.

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
    """
    rng = jax.random.key(seed)
    data_shape = model.data_shape

    def _h(x_flat):
        """Inequality h(x), applied to a single flattened normalized sample."""
        x = x_flat.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(penalty_weight * constraint_fn(x))

    def _step_single(x, v, s, lmbda, t):
        """Integration step for a single sample with slack variables.

        The augmented equality constraint is G(x, s) = h(x) - s = 0 with
        augmented Jacobian J_aug = [J_h, -I].
        """
        x_flat = x.ravel()
        v_flat = v.ravel()

        h = _h(x_flat)
        J_h = jnp.atleast_2d(jax.jacobian(_h)(x_flat))

        # Augmented equality constraint: G(x, s) = h(x) - s.
        g = h - s

        if method == "flow":
            dt_lmbda = rescale_factor * dt / (1 - t + 1e-8)
            lmbda = lmbda + dt_lmbda * g
        elif method == "pseudoinverse":
            # J_aug @ J_aug^T = J_h @ J_h^T + I (the +I comes from the slack).
            # J_aug @ v_aug = J_h @ v_flat (since s_dot = 0).
            JJT = J_h @ J_h.T + jnp.eye(g.shape[0])
            lmbda = jnp.linalg.solve(JJT, J_h @ v_flat)
        elif method == "penalty":
            lmbda = jnp.zeros_like(g)
        else:
            raise ValueError(f"Invalid method: {method}")

        # x update: x_dot = v - J_h^T @ lmbda - J_h^T @ g
        x_dot = v_flat - J_h.T @ lmbda - J_h.T @ g
        x = x + dt * x_dot.reshape(data_shape)

        # s update: s_dot = 0 - (-I)^T @ lmbda - (-I)^T @ g = lmbda + g
        s = s + dt * (lmbda + g)

        # Project slack onto s <= 0.
        s = jnp.minimum(s, 0.0)

        return x, s, lmbda

    def _step_fn(carry, t):
        """Batched forward Euler step with inequality constraint projection."""
        x, s, lmbda = carry

        t_batch = jnp.full((x.shape[0],), t)
        v = model(x, t_batch)

        x_next, s_next, lmbda_next = jax.vmap(
            _step_single, in_axes=(0, 0, 0, 0, None)
        )(x, v, s, lmbda, t)

        return (x_next, s_next, lmbda_next), x_next

    # Data samples are initialized as Gaussian noise.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    # Initialize slacks: s = min(h(x_0), 0) so s <= 0.
    h_init = jax.vmap(lambda xi: _h(xi.ravel()))(x_init)
    s_init = jnp.minimum(h_init, 0.0)
    lmbda_init = jnp.zeros_like(h_init)

    # Integrate the constrained flow ODE from t=0 to t=1.
    timesteps = jnp.arange(0, 1.0, dt)
    (x, _, _), xs = jax.lax.scan(
        _step_fn, (x_init, s_init, lmbda_init), timesteps
    )

    # All trajectories are in normalized space, so unnormalize before returning.
    return normalizer.unnormalize(x), normalizer.unnormalize(xs)
