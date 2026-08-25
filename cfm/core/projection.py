"""Gauss-Newton projection onto a constraint set.

This iteration previously existed in four places -- the equality flow, the
inequality flow, PCFM's per-step projection, and the CBF terminal filter --
which had already drifted apart in their regularizers.

Each pass takes the minimum-norm step that zeroes the linearized residual,

    x <- x - J^T (J J^T)^{-1} r(x),

which for a nonlinear constraint overshoots badly in one step, so it is
iterated. Passing ``active_only=True`` masks out the rows that are already
satisfied, turning the equality projection into the inequality one: it lands
on the boundary of the feasible set and leaves feasible samples untouched.
"""

from typing import Callable

import jax
import jax.numpy as jnp


def gauss_newton_step(
    residual_fn: Callable[[jax.Array], jax.Array],
    x_flat: jax.Array,
    *,
    active_only: bool = False,
    eps_reg: float = 1e-10,
) -> jax.Array:
    """Take one Gauss-Newton step towards ``residual_fn(x) = 0``.

    Args:
        residual_fn: Residual on a single flat sample.
        x_flat: The sample to move.
        active_only: Zero out rows whose residual is already <= 0, giving the
            inequality (active-set) form.
        eps_reg: Tikhonov regularizer on ``J J^T``.
    """
    r = residual_fn(x_flat)
    J = jax.jacobian(residual_fn)(x_flat)  # (m, n)

    if active_only:
        active = r > 0.0
        J = jnp.where(active[:, None], J, 0.0)
        r = jnp.where(active, r, 0.0)

    z = jnp.linalg.solve(J @ J.T + eps_reg * jnp.eye(r.shape[0]), r)
    return x_flat - J.T @ z


def gauss_newton_project(
    residual_fn: Callable[[jax.Array], jax.Array],
    x_flat: jax.Array,
    *,
    num_iters: int,
    active_only: bool = False,
    eps_reg: float = 1e-10,
) -> jax.Array:
    """Iterate :func:`gauss_newton_step` ``num_iters`` times.

    Returns ``x_flat`` unchanged when ``num_iters == 0``.
    """
    if num_iters == 0:
        return x_flat

    def _step(x, _):
        return gauss_newton_step(
            residual_fn, x, active_only=active_only, eps_reg=eps_reg
        ), None

    x_flat, _ = jax.lax.scan(_step, x_flat, length=num_iters)
    return x_flat


def project_batch(
    residual_fn: Callable[[jax.Array], jax.Array],
    x: jax.Array,
    *,
    num_iters: int,
    active_only: bool = False,
    eps_reg: float = 1e-10,
) -> jax.Array:
    """Project a batch of samples of arbitrary shape, preserving that shape."""
    if num_iters == 0:
        return x
    projected = jax.vmap(
        lambda x_i: gauss_newton_project(
            residual_fn, x_i.ravel(), num_iters=num_iters,
            active_only=active_only, eps_reg=eps_reg,
        )
    )(x)
    return projected.reshape(x.shape)
