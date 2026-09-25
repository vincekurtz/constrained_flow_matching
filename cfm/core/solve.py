"""Shared scaffolding for the generation methods."""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import diffrax
import jax
import jax.numpy as jnp

from cfm.models.normalizer import Normalizer


class Samples(NamedTuple):
    """Output of a generation method.

    Attributes:
        x: final samples, ``(num_samples, *data_shape)``.
        xs: trajectories, ``(num_saved, num_samples, *data_shape)``.
        num_steps: solver steps taken, if tracked.
    """

    x: jax.Array
    xs: jax.Array
    num_steps: Optional[Any] = None


def resolve_rng(rng: Optional[jax.Array], seed: int = 0) -> jax.Array:
    """Return ``rng``, or a key from ``seed`` if it is None."""
    return jax.random.key(seed) if rng is None else rng


def initial_noise(
    rng: jax.Array, num_samples: int, data_shape: Tuple[int, ...]
) -> jax.Array:
    """Initial Gaussian noise in normalized space."""
    return jax.random.normal(rng, (num_samples,) + data_shape)


def save_times(dt: float, t1: float) -> jax.Array:
    """Times at which to record the trajectory, excluding ``t0``."""
    return jnp.arange(dt, t1, dt)


def wrap_constraint(
    constraint_fn: Callable[[jax.Array], jax.Array],
    normalizer: Normalizer,
    data_shape: Tuple[int, ...],
    weight: float = 1.0,
) -> Callable[[jax.Array], jax.Array]:
    """Map ``g`` on unnormalized samples to a 1-D, weighted residual on flat
    normalized samples."""

    def wrapped(x_flat: jax.Array) -> jax.Array:
        x = normalizer.unnormalize(x_flat.reshape(data_shape))
        return jnp.atleast_1d(weight * constraint_fn(x))

    return wrapped


def batched_velocity(
    model, t: jax.Array, x: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate the model on a batch; return ``x`` and ``v`` flattened."""
    v = model(x, jnp.full((x.shape[0],), t))
    return x.reshape((x.shape[0], -1)), v.reshape((v.shape[0], -1))


def solve_flow(
    ode_fn: Callable,
    y0,
    *,
    dt: float,
    t1: float = 1.0,
    save_ts: Optional[jax.Array] = None,
    save_t1: bool = False,
    solver: diffrax.AbstractSolver = None,
    stepsize_controller: diffrax.AbstractStepSizeController = None,
    max_steps: int = 10_000,
):
    """Integrate ``ode_fn`` from 0 to ``t1``; return ``(ys, num_steps)``.

    Defaults to ``Midpoint()`` with ``ConstantStepSize()``, saving at
    ``save_times(dt, t1)`` plus ``t0`` (and ``t1`` if ``save_t1``).
    """
    if solver is None:
        solver = diffrax.Midpoint()
    if stepsize_controller is None:
        stepsize_controller = diffrax.ConstantStepSize()
    if save_ts is None:
        save_ts = save_times(dt, t1)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_fn),
        solver,
        t0=0.0,
        t1=t1,
        dt0=dt,
        y0=y0,
        saveat=diffrax.SaveAt(ts=save_ts, t0=True, t1=save_t1),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    return solution.ys, solution.stats["num_steps"]
