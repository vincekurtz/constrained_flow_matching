"""Lagrangian Dual Flows (LDF).

Enforces constraints on a pre-trained flow ``xdot = v(x, t)`` with Lagrangian
dual dynamics. For an equality constraint ``g(x) = 0``,

    xdot      = v(x, t) - grad g(x)^T lambda - grad g(x)^T g(x),
    lambdadot = g(x) / (1 - t)^p.

An inequality ``h(x) <= 0`` reduces to the same flow by introducing a slack
variable s >= 0 and enforcing ``g = h(x) + s = 0``. Setting
``rescale_factor = 0`` gives a pure quadratic penalty.

    Kurtz and Davydov, "Constrained Flow Matching via Lagrangian Dual Flows".
    https://arxiv.org/abs/2607.04513
"""

from typing import Optional

import diffrax
import jax
import jax.numpy as jnp
from flax import nnx

from cfm.core.constraints import EQUALITY, Constraint
from cfm.core.projection import project_batch
from cfm.core.solve import (
    Samples,
    batched_velocity,
    initial_noise,
    resolve_rng,
    solve_flow,
    wrap_constraint,
)
from cfm.models.normalizer import Normalizer

EQUALITY_EPS = 1e-6
INEQUALITY_EPS = 1e-8
INEQUALITY_EXPONENT = 1.0

SLACK_MODES = ("closed_form", "ode")


def generate_unconstrained(
    model: nnx.Module,
    normalizer: Normalizer,
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    rng: Optional[jax.Array] = None,
    solver: diffrax.AbstractSolver = None,
    stepsize_controller: diffrax.AbstractStepSizeController = None,
) -> Samples:
    """Generate samples from a trained flow model with no constraint."""
    rng = resolve_rng(rng, seed)
    x_init = initial_noise(rng, num_samples, model.data_shape)

    def _ode_fn(t, y, args):
        del args
        return model(y, jnp.full((y.shape[0],), t))

    xs, num_steps = solve_flow(
        _ode_fn, x_init, dt=dt, save_t1=False,
        solver=solver, stepsize_controller=stepsize_controller,
    )
    return Samples(
        normalizer.unnormalize(xs[-1]), normalizer.unnormalize(xs), num_steps
    )


def _equality_flow(model, _g, rescale_factor, rescale_exponent):
    """Primal-dual dynamics for g(x) = 0. State is (x, lambda)."""

    def ode_fn(t, y, args):
        del args
        x, lmbda = y
        x_flat, v_flat = batched_velocity(model, t, x)

        def _vjp(x_i, lmbda_i):
            g_i, vjp_fn = jax.vjp(_g, x_i)
            return g_i, vjp_fn(lmbda_i + g_i)[0]

        g, correction = jax.vmap(_vjp)(x_flat, lmbda)
        x_dot = (v_flat - correction).reshape(x.shape)
        lmbda_dot = (
            rescale_factor * g / (1 - t + EQUALITY_EPS) ** rescale_exponent
        )
        return x_dot, lmbda_dot

    return ode_fn


def _inequality_closed_form_flow(model, _h, rescale_factor, data_shape):
    """Primal-dual dynamics for h(x) <= 0 with the slack in closed form.

    The optimal slack ``s* = max(-h - lambda, 0)`` gives
    ``g = max(h, -lambda)``. State is (x, lambda).
    """

    def ode_fn(t, y, args):
        del args
        x, lmbda = y
        x_flat, v_flat = batched_velocity(model, t, x)

        def _single(x_i, v_i, lmbda_i):
            h, vjp_fn = jax.vjp(_h, x_i)
            g = jnp.maximum(h, -lmbda_i)
            active = (h > -lmbda_i).astype(g.dtype)

            x_dot = (v_i - vjp_fn(active * (lmbda_i + g))[0]).reshape(
                data_shape
            )
            lmbda_dot = (
                rescale_factor * g
                / (1 - t + INEQUALITY_EPS) ** INEQUALITY_EXPONENT
            )
            return x_dot, lmbda_dot

        x_dot, lmbda_dot = jax.vmap(_single)(x_flat, v_flat, lmbda)
        return x_dot.reshape(x.shape), lmbda_dot

    return ode_fn


def _inequality_slack_flow(model, _h, rescale_factor, data_shape):
    """Primal-dual dynamics for h(x) <= 0 with the slack as an ODE state.

    ``s_dot = max(-h - lambda, 0) - s``. State is (x, s, lambda).
    """

    def ode_fn(t, y, args):
        del args
        x, s, lmbda = y
        x_flat, v_flat = batched_velocity(model, t, x)

        def _single(x_i, v_i, s_i, lmbda_i):
            h, vjp_fn = jax.vjp(_h, x_i)
            g = h + s_i
            x_dot = (v_i - vjp_fn(lmbda_i + g)[0]).reshape(data_shape)
            s_dot = jnp.maximum(-h - lmbda_i, 0) - s_i
            lmbda_dot = (
                rescale_factor * g
                / (1 - t + INEQUALITY_EPS) ** INEQUALITY_EXPONENT
            )
            return x_dot, s_dot, lmbda_dot

        x_dot, s_dot, lmbda_dot = jax.vmap(_single)(x_flat, v_flat, s, lmbda)
        return x_dot.reshape(x.shape), s_dot, lmbda_dot

    return ode_fn


def generate(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint: Constraint,
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    rng: Optional[jax.Array] = None,
    penalty_weight: float = 5.0,
    rescale_factor: float = 1.0,
    rescale_exponent: float = 2.0,
    slack: str = "closed_form",
    num_projection_iters: int = 0,
    solver: diffrax.AbstractSolver = None,
    stepsize_controller: diffrax.AbstractStepSizeController = None,
    max_steps: Optional[int] = None,
) -> Samples:
    """Generate samples subject to ``constraint`` via the dual flow.

    Args:
        penalty_weight: scale on the constraint residual.
        rescale_factor: gain on the multiplier flow; zero gives a pure penalty.
        rescale_exponent: p in 1/(1 - t)^p. Equality constraints only.
        slack: ``"closed_form"`` or ``"ode"``. Inequality constraints only.
        num_projection_iters: Gauss-Newton projection steps on the final
            samples.
        max_steps: solver step cap. Defaults to 10k (equality) or 100k
            (inequality).
    """
    if slack not in SLACK_MODES:
        raise ValueError(f"slack must be one of {SLACK_MODES}, got {slack!r}")

    rng = resolve_rng(rng, seed)
    data_shape = model.data_shape
    is_equality = constraint.kind == EQUALITY

    _c = wrap_constraint(
        constraint.fn, normalizer, data_shape, penalty_weight
    )

    x_init = initial_noise(rng, num_samples, data_shape)
    c_init = jax.vmap(lambda xi: _c(xi.ravel()))(x_init)
    lmbda_init = jnp.zeros_like(c_init)

    if is_equality:
        ode_fn = _equality_flow(model, _c, rescale_factor, rescale_exponent)
        y_init = (x_init, lmbda_init)
        default_max_steps = 10_000
    elif slack == "closed_form":
        ode_fn = _inequality_closed_form_flow(
            model, _c, rescale_factor, data_shape
        )
        y_init = (x_init, lmbda_init)
        default_max_steps = 100_000
    else:
        ode_fn = _inequality_slack_flow(
            model, _c, rescale_factor, data_shape
        )
        y_init = (x_init, jnp.maximum(-c_init, 0.0), lmbda_init)
        default_max_steps = 100_000

    ys, num_steps = solve_flow(
        ode_fn, y_init, dt=dt, save_t1=False,
        solver=solver, stepsize_controller=stepsize_controller,
        max_steps=default_max_steps if max_steps is None else max_steps,
    )
    xs = ys[0]
    x = xs[-1]

    if num_projection_iters > 0:
        x = project_batch(
            _c, x, num_iters=num_projection_iters, active_only=not is_equality
        )
        xs = xs.at[-1].set(x)

    return Samples(
        normalizer.unnormalize(x), normalizer.unnormalize(xs), num_steps
    )
