from typing import Callable, Tuple

import diffrax
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

    Integrates the learned vector field from t=0 to t=1, then unnormalizes
    the result.

    Args:
        model: Trained flow model xdot = v(x, t).
        normalizer: Normalizer used during training, applied in reverse to
            produced samples in the original data space.
        num_samples: Number of samples to generate.
        dt: Initial step size hint for the adaptive integrator.
        seed: Random seed for the initial noise.

    Returns:
        x: Final generated samples of shape (num_samples, *data_shape).
        xs: Full trajectories of shape (num_steps, num_samples, *data_shape).
    """
    rng = jax.random.key(seed)
    x_init = jax.random.normal(rng, (num_samples,) + model.data_shape)

    def _ode_fn(t, y, args):
        del args
        t_batch = jnp.full((y.shape[0],), t)
        return model(y, t_batch)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        diffrax.Dopri5(),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=x_init,
        saveat=diffrax.SaveAt(ts=jnp.arange(dt, 1.0, dt)),
        stepsize_controller=diffrax.PIDController(
            rtol=1e-3, atol=1e-3, dtmin=1e-4, dtmax=0.1
        ),
    )
    xs = solution.ys
    x = xs[-1]

    print(solution.stats["num_steps"], "steps taken")

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)


def generate_constrained(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    penalty_weight: float = 5.0,
    rescale_factor: float = 1.0,
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
        num_samples: Number of samples to generate.
        dt: Step size used for solution output times between ``t=0`` and
            ``t=1``.
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
    data_shape = model.data_shape

    def _g(x):
        """Constraint g(x), applied to a single flattened normalized sample."""
        x = x.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(penalty_weight * constraint_fn(x))

    def _ode_fn(t, y, args):
        """Batched constrained dynamics for the primal-dual flow."""
        del args
        x, lmbda = y
        t_batch = jnp.full((x.shape[0],), t)
        v = model(x, t_batch)

        x_flat = x.reshape((x.shape[0], -1))
        v_flat = v.reshape((v.shape[0], -1))

        def _vjp(x_i, lmbda_i):
            g_i, vjp_fn = jax.vjp(_g, x_i)
            return g_i, vjp_fn(lmbda_i + g_i)[0]

        g, correction = jax.vmap(_vjp)(x_flat, lmbda)
        x_dot = (v_flat - correction).reshape(x.shape)
        lmbda_dot = rescale_factor * g / (1 - t + 1e-6) ** 2

        return x_dot, lmbda_dot

    # Data samples are initialized as Gaussian noise.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    # Initialise multipliers as zeros of the correct shape
    lmbda_init = 0.0 * jax.vmap(lambda xi: _g(xi.ravel()))(x_init)

    # Integrate the constrained flow ODE from t=0 to t=1.
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        diffrax.Dopri5(),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=(x_init, lmbda_init),
        saveat=diffrax.SaveAt(ts=jnp.arange(dt, 1.0, dt)),
        # stepsize_controller=diffrax.ConstantStepSize(),
        stepsize_controller=diffrax.PIDController(
            rtol=1e-3, atol=1e-3, dtmin=1e-4, dtmax=0.1
        ),
    )
    print(solution.stats["num_steps"], "steps taken")
    xs, _ = solution.ys
    x = xs[-1]

    # All trajectories are in normalized space, so unnormalize before returning.
    return normalizer.unnormalize(x), normalizer.unnormalize(xs)


def generate_inequality_constrained(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    penalty_weight: float = 5.0,
    rescale_factor: float = 10.0,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow model subject to h(x) <= 0.

    Introduces slack variables s <= 0 and enforces h(x) = s as an equality
    constraint via the primal-dual flow on the augmented state (x, s, lmbda).

    Args:
        model: Trained flow model xdot = v(x, t). Must have a ``data_shape``
            attribute.
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable function ``h(x)`` (operating on a single
            *unnormalized* sample) defining the inequality ``h(x) <= 0``.
            May return a scalar or a 1-D array.
        num_samples: Number of samples to generate.
        dt: Initial step size hint for the adaptive integrator.
        seed: Random seed for the initial noise.
        penalty_weight: Strength of the quadratic penalty pulling samples
            toward the constraint manifold.
        rescale_factor: Factor by which to rescale the time for the Lagrange
            multiplier flow.

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

    def _ode_fn(t, y, args):
        """Batched constrained dynamics for the primal-dual flow."""
        del args
        x, s, lmbda = y
        t_batch = jnp.full((x.shape[0],), t)
        v = model(x, t_batch)

        x_flat = x.reshape((x.shape[0], -1))
        v_flat = v.reshape((v.shape[0], -1))

        def _single(x_i, v_i, s_i, lmbda_i):
            h, vjp_fn = jax.vjp(_h, x_i)
            g = h - s_i
            x_dot = (v_i - vjp_fn(lmbda_i + g)[0]).reshape(data_shape)
            lmbda_dot = rescale_factor * g / (1 - t + 1e-8)
            s_dot = jnp.minimum(lmbda_i + h, 0) - s_i
            return x_dot, s_dot, lmbda_dot

        x_dot, s_dot, lmbda_dot = jax.vmap(_single)(x_flat, v_flat, s, lmbda)
        return x_dot.reshape(x.shape), s_dot, lmbda_dot

    # Data samples are initialized as Gaussian noise.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    # Initialize slacks: s = min(h(x_0), 0) so s <= 0.
    h_init = jax.vmap(lambda xi: _h(xi.ravel()))(x_init)
    s_init = jnp.minimum(h_init, 0.0)
    lmbda_init = jnp.zeros_like(h_init)

    # Integrate the constrained flow ODE from t=0 to t=1.
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        diffrax.Dopri5(),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=(x_init, s_init, lmbda_init),
        saveat=diffrax.SaveAt(ts=jnp.arange(dt, 1.0, dt)),
        stepsize_controller=diffrax.ConstantStepSize(),
        # stepsize_controller=diffrax.PIDController(
        #     rtol=1e-3, atol=1e-3, dtmin=1e-3, dtmax=0.1
        # ),
    )
    print(solution.stats["num_steps"], "steps taken")
    xs, _, _ = solution.ys
    x = xs[-1]

    # All trajectories are in normalized space, so unnormalize before returning.
    return normalizer.unnormalize(x), normalizer.unnormalize(xs)
