"""Scaffolding shared by every generation method.

Each generator used to re-derive all of this for itself: the data shape, the
default RNG, the constraint adapter, the initial noise, the save times, the
``diffeqsolve`` call and the final unnormalize. Collecting it here is what
makes the individual methods short enough to read as algorithms.

Nothing in this module is traced-value dependent -- it composes closures that
JIT to the same graph the open-coded versions did.
"""

from typing import Any, Callable, NamedTuple, Optional, Tuple

import diffrax
import jax
import jax.numpy as jnp

from cfm.models.normalizer import Normalizer


class Samples(NamedTuple):
    """What every generation method returns.

    Methods used to return either a 2-tuple or a 3-tuple depending on which
    one you called, so every caller unpacked differently.

    Attributes:
        x: Final samples, shape ``(num_samples, *data_shape)``.
        xs: Trajectories, shape ``(num_saved, num_samples, *data_shape)``.
        num_steps: Solver steps taken, where the method tracks it.
    """

    x: jax.Array
    xs: jax.Array
    num_steps: Optional[Any] = None


def resolve_rng(rng: Optional[jax.Array], seed: int = 0) -> jax.Array:
    """Return ``rng``, or a key built from ``seed`` when it is None.

    Methods historically disagreed about whether they took ``rng``, ``seed``
    or both; they now all take both and funnel through here.
    """
    return jax.random.key(seed) if rng is None else rng


def initial_noise(
    rng: jax.Array, num_samples: int, data_shape: Tuple[int, ...]
) -> jax.Array:
    """Draw the Gaussian noise the flow starts from, in normalized space."""
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
    """Adapt a user constraint to the form the solvers use internally.

    The user writes ``g(x)`` against a single *unnormalized* sample of the
    natural shape. The solvers work on single *flat normalized* samples and
    want a 1-D result, optionally pre-scaled by the penalty weight.

    Args:
        constraint_fn: The user's ``g(x)`` (or ``h(x)``).
        normalizer: Normalizer used during training.
        data_shape: Shape of a single sample.
        weight: Scale applied to the residual, i.e. the penalty weight.

    Returns:
        A function from a flat normalized sample to a 1-D residual.
    """

    def wrapped(x_flat: jax.Array) -> jax.Array:
        x = normalizer.unnormalize(x_flat.reshape(data_shape))
        return jnp.atleast_1d(weight * constraint_fn(x))

    return wrapped


def batched_velocity(
    model, t: jax.Array, x: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Evaluate the model on a batch and return flat views of ``x`` and ``v``.

    Returns:
        x_flat: ``x`` reshaped to ``(batch, -1)``.
        v_flat: ``v(x, t)`` reshaped to ``(batch, -1)``.
    """
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
    """Integrate ``ode_fn`` from ``t = 0`` to ``t = t1``.

    Args:
        ode_fn: Right-hand side, in diffrax's ``(t, y, args)`` form.
        y0: Initial state (any pytree).
        dt: Step size, or the initial step hint for adaptive controllers.
        t1: Final time.
        save_ts: Times to record. Defaults to ``save_times(dt, t1)``.
        save_t1: Whether to additionally record ``t1``. Note the equality and
            inequality flows deliberately do *not* -- see ``methods/ldf.py``.
        solver: diffrax solver. Defaults to ``Midpoint()``.
        stepsize_controller: Defaults to ``ConstantStepSize()``.
        max_steps: Solver step cap.

    Returns:
        ys: The recorded trajectory pytree.
        num_steps: Number of solver steps taken.
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
