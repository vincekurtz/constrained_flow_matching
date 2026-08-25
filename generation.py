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
    solver: diffrax.AbstractSolver = diffrax.Midpoint(),
    stepsize_controller:
        diffrax.AbstractStepSizeController = diffrax.ConstantStepSize(),
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow-matching model.

    Integrates the learned vector field from t=0 to t=1, then unnormalizes
    the result.

    Args:
        model: Trained flow model xdot = v(x, t).
        normalizer: Normalizer used during training, applied in reverse to
            produced samples in the original data space.
        num_samples: Number of samples to generate.
        dt: Step size (or initial step size hint for adaptive controllers).
        seed: Random seed for the initial noise.
        solver: diffrax solver to use. Defaults to ``Midpoint()``.
        stepsize_controller: diffrax step-size controller. Defaults to
            ``ConstantStepSize()``. Pass a ``PIDController`` for adaptive
            error control.

    Returns:
        x: Final generated samples of shape (num_samples, *data_shape).
        xs: Full trajectories of shape (num_steps, num_samples, *data_shape).
            Contains NaNs if integration fails.
    """
    rng = jax.random.key(seed)
    x_init = jax.random.normal(rng, (num_samples,) + model.data_shape)
    save_ts = jnp.arange(dt, 1.0, dt)

    def _ode_fn(t, y, args):
        del args
        t_batch = jnp.full((y.shape[0],), t)
        return model(y, t_batch)

    try:
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_ode_fn),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=dt,
            y0=x_init,
            saveat=diffrax.SaveAt(ts=save_ts, t0=True),
            stepsize_controller=stepsize_controller,
        )
        xs = solution.ys
        x = xs[-1]
        print(solution.stats["num_steps"], "steps taken")
    except Exception as e:
        print(f"diffeqsolve failed: {e}")
        nan = jnp.nan
        x = jnp.full((num_samples,) + model.data_shape, nan)
        xs = jnp.full((len(save_ts) + 1, num_samples) + model.data_shape, nan)

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)


def generate_constrained(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    rng: jax.Array = None,
    penalty_weight: float = 5.0,
    rescale_factor: float = 1.0,
    rescale_exponent: float = 2.0,
    num_projection_iters: int = 0,
    solver: diffrax.AbstractSolver = diffrax.Midpoint(),
    stepsize_controller: diffrax.AbstractStepSizeController = (
        diffrax.ConstantStepSize()
    ),
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
        dt: Step size (or initial step size hint for adaptive controllers).
        rng: PRNG key for the initial noise. Defaults to ``jax.random.key(0)``
            when not provided.
        penalty_weight: Strength of the quadratic penalty pulling samples
            toward the constraint manifold.
        rescale_factor: Factor by which to rescale the time for the Lagrange
            multiplier flow. This can help enforce the constraint more strictly
            but leads to a stiffer ODE.
        rescale_exponent: Exponent for the rescaling factor (p).
        num_projection_iters: Number of Gauss-Newton iterations used to
            project the final samples onto the constraint manifold. Zero
            (the default) returns the samples produced by the flow as-is.
        solver: diffrax solver to use. Defaults to ``Midpoint()``.
        stepsize_controller: diffrax step-size controller. Defaults to
            ``ConstantStepSize()``. Pass a ``PIDController`` for adaptive
            error control.

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
            Contains NaNs if integration fails.
        num_steps: Number of ODE solver steps taken. ``None`` if integration
            failed.
    """
    if rng is None:
        rng = jax.random.key(0)
    data_shape = model.data_shape

    def _g(x):
        """Constraint g(x), applied to a single flattened normalized sample."""
        x = x.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(penalty_weight * constraint_fn(x))

    def _project(x_flat):
        """Project a single flat sample onto g(x) = 0 with Gauss-Newton.

        Each iteration takes the minimum-norm step that zeroes the linearised
        residual, x <- x - J^T (J J^T)^{-1} g(x). One step overshoots badly
        for strongly nonlinear constraints, so we iterate.
        """
        eps_reg = 1e-10  # Tikhonov regulariser on J J^T.

        def _step(x_flat, _):
            r = _g(x_flat)
            J = jax.jacobian(_g)(x_flat)  # (m, n)
            z = jnp.linalg.solve(J @ J.T + eps_reg * jnp.eye(r.shape[0]), r)
            return x_flat - J.T @ z, None

        x_flat, _ = jax.lax.scan(_step, x_flat, length=num_projection_iters)
        return x_flat

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
        lmbda_dot = rescale_factor * g / (1 - t + 1e-6) ** rescale_exponent

        return x_dot, lmbda_dot

    # Data samples are initialized as Gaussian noise.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    # Initialise multipliers as zeros of the correct shape
    lmbda_init = 0.0 * jax.vmap(lambda xi: _g(xi.ravel()))(x_init)

    # Integrate the constrained flow ODE from t=0 to t=1.
    save_ts = jnp.arange(dt, 1.0, dt)
    try:
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(_ode_fn),
            solver,
            t0=0.0,
            t1=1.0,
            dt0=dt,
            y0=(x_init, lmbda_init),
            saveat=diffrax.SaveAt(ts=save_ts, t0=True),
            stepsize_controller=stepsize_controller,
            max_steps=10_000,
        )
        num_steps = solution.stats["num_steps"]
        print(num_steps, "steps taken")
        xs, _ = solution.ys
        x = xs[-1]
    except Exception as e:
        print(f"diffeqsolve failed: {e}")
        nan = jnp.nan
        num_steps = None
        x = jnp.full((num_samples,) + data_shape, nan)
        xs = jnp.full((len(save_ts) + 1, num_samples) + data_shape, nan)

    # Optionally polish the samples with a Gauss-Newton projection, removing
    # the residual constraint violation left by the finite-time flow.
    if num_projection_iters > 0 and num_steps is not None:
        x = jax.vmap(lambda x_i: _project(x_i.ravel()))(x).reshape(x.shape)
        xs = xs.at[-1].set(x)

    # All trajectories are in normalized space, so unnormalize before returning.
    return normalizer.unnormalize(x), normalizer.unnormalize(xs), num_steps


def generate_inequality_constrained(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    rng: jax.Array = None,
    penalty_weight: float = 5.0,
    rescale_factor: float = 10.0,
    slack: str = "closed_form",
    num_projection_iters: int = 0,
    solver: diffrax.AbstractSolver = diffrax.Midpoint(),
    stepsize_controller: diffrax.AbstractStepSizeController = (
        diffrax.ConstantStepSize()
    ),
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a trained flow model subject to h(x) <= 0.

    Introduces a slack variable s >= 0 and enforces g = h(x) + s = 0 as an
    equality constraint via the primal-dual flow. The ``slack`` argument
    selects how s is handled:

    ``"closed_form"``
        For fixed x and lmbda the augmented Lagrangian is minimized over
        s >= 0 by

            s* = max(-h(x) - lmbda, 0),   so   g = max(h(x), -lmbda),

        which is substituted directly. The flow state is (x, lmbda), and
        inactive constraints (h < -lmbda) exert no force.

    ``"ode"``
        s is carried as an extra state and relaxed toward the same minimizer,
        ``s_dot = max(-h - lmbda, 0) - s``, giving the flow state (x, s,
        lmbda). Note this relaxation has a time constant of 1, comparable to
        the whole [0, 1] horizon, so s lags behind h(x) and the residual
        g = h + s pushes on every constraint, active or not.

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
        seed: Random seed for the initial noise. Ignored when ``rng`` is
            given.
        rng: PRNG key for the initial noise. Defaults to
            ``jax.random.key(seed)`` when not provided.
        penalty_weight: Strength of the quadratic penalty pulling samples
            toward the constraint manifold.
        rescale_factor: Factor by which to rescale the time for the Lagrange
            multiplier flow.
        slack: How to handle the slack variable, ``"closed_form"`` (default)
            or ``"ode"``. See above.
        num_projection_iters: Number of Gauss-Newton iterations used to
            project the final samples onto the feasible set ``h(x) <= 0``.
            Zero (the default) returns the samples produced by the flow
            as-is.
        solver: diffrax solver to use. Defaults to ``Midpoint()``.
        stepsize_controller: diffrax step-size controller. Defaults to
            ``ConstantStepSize()``. Pass a ``PIDController`` for adaptive
            error control, which the stiff dynamics that come with many
            active constraints tend to need.

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
    """
    if slack not in ("closed_form", "ode"):
        raise ValueError(
            f"slack must be 'closed_form' or 'ode', got {slack!r}"
        )

    if rng is None:
        rng = jax.random.key(seed)
    data_shape = model.data_shape

    def _h(x_flat):
        """Inequality h(x), applied to a single flattened normalized sample."""
        x = x_flat.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(penalty_weight * constraint_fn(x))

    def _project(x_flat):
        """Project a single flat sample onto h(x) <= 0 with Gauss-Newton.

        Each pass linearises only the violated constraints and takes the
        min-norm step that zeroes them, landing on the boundary of the
        feasible set. Constraints that are already satisfied are masked out
        so the projection leaves feasible samples untouched.
        """
        eps_reg = 1e-10  # Tikhonov regulariser on J J^T.

        def _step(x_flat, _):
            h = _h(x_flat)
            J = jax.jacobian(_h)(x_flat)  # (m, n)
            active = h > 0.0
            J_act = jnp.where(active[:, None], J, 0.0)
            r = jnp.where(active, h, 0.0)
            z = jnp.linalg.solve(
                J_act @ J_act.T + eps_reg * jnp.eye(h.shape[0]), r
            )
            return x_flat - J_act.T @ z, None

        x_flat, _ = jax.lax.scan(_step, x_flat, length=num_projection_iters)
        return x_flat

    def _model_terms(t, x):
        """Vector field and flattened views used by both formulations."""
        v = model(x, jnp.full((x.shape[0],), t))
        return x.reshape((x.shape[0], -1)), v.reshape((v.shape[0], -1))

    def _ode_closed_form(t, y, args):
        """Primal-dual flow with the slack substituted in closed form."""
        del args
        x, lmbda = y
        x_flat, v_flat = _model_terms(t, x)

        def _single(x_i, v_i, lmbda_i):
            h, vjp_fn = jax.vjp(_h, x_i)

            # g = h + s* with the optimal slack substituted in. Inactive
            # constraints (h < -lmbda) have dg/dh = 0 and so exert no force.
            g = jnp.maximum(h, -lmbda_i)
            active = (h > -lmbda_i).astype(g.dtype)

            x_dot = (v_i - vjp_fn(active * (lmbda_i + g))[0]).reshape(
                data_shape
            )
            lmbda_dot = rescale_factor * g / (1 - t + 1e-8)
            return x_dot, lmbda_dot

        x_dot, lmbda_dot = jax.vmap(_single)(x_flat, v_flat, lmbda)
        return x_dot.reshape(x.shape), lmbda_dot

    def _ode_slack(t, y, args):
        """Primal-dual flow with the slack carried as an extra state."""
        del args
        x, s, lmbda = y
        x_flat, v_flat = _model_terms(t, x)

        def _single(x_i, v_i, s_i, lmbda_i):
            h, vjp_fn = jax.vjp(_h, x_i)
            g = h + s_i
            x_dot = (v_i - vjp_fn(lmbda_i + g)[0]).reshape(data_shape)
            s_dot = jnp.maximum(-h - lmbda_i, 0) - s_i
            lmbda_dot = rescale_factor * g / (1 - t + 1e-8)
            return x_dot, s_dot, lmbda_dot

        x_dot, s_dot, lmbda_dot = jax.vmap(_single)(x_flat, v_flat, s, lmbda)
        return x_dot.reshape(x.shape), s_dot, lmbda_dot

    # Data samples are initialized as Gaussian noise.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    # Multipliers start at zero, so the flow initially follows the prior.
    h_init = jax.vmap(lambda xi: _h(xi.ravel()))(x_init)
    lmbda_init = jnp.zeros_like(h_init)

    if slack == "closed_form":
        ode_fn = _ode_closed_form
        y_init = (x_init, lmbda_init)
    else:
        # Initialize slacks: s = max(-h(x_0), 0) so s >= 0.
        ode_fn = _ode_slack
        y_init = (x_init, jnp.maximum(-h_init, 0.0), lmbda_init)

    # Integrate the constrained flow ODE from t=0 to t=1.
    save_ts = jnp.arange(dt, 1.0, dt)
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_fn),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=y_init,
        saveat=diffrax.SaveAt(ts=save_ts, t0=True),
        stepsize_controller=stepsize_controller,
        max_steps=100_000,
    )
    print(solution.stats["num_steps"], "steps taken")
    xs = solution.ys[0]
    x = xs[-1]

    # Optionally polish the samples with a Gauss-Newton projection, removing
    # the residual constraint violation left by the finite-time flow.
    if num_projection_iters > 0:
        x = jax.vmap(lambda x_i: _project(x_i.ravel()))(x).reshape(x.shape)
        xs = xs.at[-1].set(x)

    # All trajectories are in normalized space, so unnormalize before returning.
    return normalizer.unnormalize(x), normalizer.unnormalize(xs)
