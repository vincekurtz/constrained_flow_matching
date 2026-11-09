"""Quadratic penalty c² ||g(x)||² / (2 (1-t)²), without dual variables.

As for LDF and the constant penalty, ``penalty_weight`` is the residual
scale c. Inequalities penalize only their positive part. The singular
endpoint is excluded: by default return at the existing samplers' last
saved time (approximately 1-dt), and integrate only to that time.
"""

import diffrax
import jax
import jax.numpy as jnp

from cfm.core.constraints import EQUALITY
from cfm.core.solve import (
    Samples, batched_velocity, initial_noise, resolve_rng, save_times,
    wrap_constraint,
)


def penalty_flow(model, residual, *, equality=True, time_varying=True):
    """Build the penalty vector field on normalized samples."""
    def ode_fn(t, x, args):
        del args
        x_flat, v_flat = batched_velocity(model, t, x)

        def correction(x_i):
            g, pullback = jax.vjp(residual, x_i)
            return pullback(g if equality else jnp.maximum(g, 0))[0]

        force = jax.vmap(correction)(x_flat)
        scale = (1 - t) ** -2 if time_varying else 1.0
        return (v_flat - scale * force).reshape(x.shape)

    return ode_fn


def generate(
    model, normalizer, constraint, num_samples=1000, dt=0.01, seed=0,
    rng=None, penalty_weight=5.0, t1=None, solver=None,
    stepsize_controller=None, max_steps=100_000, save_trajectory=True,
    throw=True,
):
    """Integrate the unprojected penalty flow, strictly before t=1.

    ``dt`` is the initial step for an adaptive controller. Supply ``t1``
    explicitly to vary step size without changing the endpoint. No
    denominator clipping or final projection is applied. ``throw=False``
    returns NaN samples on solver failure for benchmark accounting.
    """
    if not 0 < dt < 1:
        raise ValueError("dt must lie strictly between 0 and 1")
    if t1 is None:
        t1 = 1.0 - dt
    if not 0 < t1 < 1:
        raise ValueError("t1 must lie strictly between 0 and 1")
    if penalty_weight < 0:
        raise ValueError("penalty_weight must be nonnegative")
    residual = wrap_constraint(
        constraint.fn, normalizer, model.data_shape, penalty_weight,
    )
    x0 = initial_noise(resolve_rng(rng, seed), num_samples, model.data_shape)
    ts = save_times(dt, t1) if save_trajectory else jnp.array([])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(penalty_flow(
            model, residual, equality=constraint.kind == EQUALITY,
        )),
        diffrax.Midpoint() if solver is None else solver,
        t0=0., t1=t1, dt0=dt, y0=x0,
        saveat=diffrax.SaveAt(t0=True, ts=ts, t1=True),
        stepsize_controller=(diffrax.ConstantStepSize()
                             if stepsize_controller is None
                             else stepsize_controller),
        max_steps=max_steps, throw=throw,
    )
    xs = normalizer.unnormalize(sol.ys)
    xs = jnp.where(sol.result == diffrax.RESULTS.successful, xs, jnp.nan)
    return Samples(xs[-1], xs, sol.stats["num_steps"])
