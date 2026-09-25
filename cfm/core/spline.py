"""Cubic Bezier spline in jax.

Mirrors ``cfm.datasets.obstacle_paths.bezier_spline``.
"""

import jax
import jax.numpy as jnp


def bezier_spline(knots: jax.Array, num_sub: int) -> jax.Array:
    """Evaluate a C1 cubic Bezier spline through ``knots``.

    Control points come from Catmull-Rom tangents. ``knots`` has shape
    ``(..., N, 2)``; the result has shape ``(..., num_sub * (N - 1) + 1, 2)``.
    """
    # One-sided differences at the ends.
    interior = 0.5 * (knots[..., 2:, :] - knots[..., :-2, :])
    first = knots[..., 1:2, :] - knots[..., 0:1, :]
    last = knots[..., -1:, :] - knots[..., -2:-1, :]
    tangents = jnp.concatenate([first, interior, last], axis=-2)

    b0 = knots[..., :-1, :]
    b3 = knots[..., 1:, :]
    b1 = b0 + tangents[..., :-1, :] / 3.0
    b2 = b3 - tangents[..., 1:, :] / 3.0

    u = jnp.linspace(0.0, 1.0, num_sub, endpoint=False)
    u = u.reshape((1,) * (knots.ndim - 1) + (num_sub, 1))
    points = (
        (1 - u) ** 3 * b0[..., None, :]
        + 3 * (1 - u) ** 2 * u * b1[..., None, :]
        + 3 * (1 - u) * u**2 * b2[..., None, :]
        + u**3 * b3[..., None, :]
    )

    points = points.reshape(knots.shape[:-2] + (-1, 2))
    return jnp.concatenate([points, knots[..., -1:, :]], axis=-2)
