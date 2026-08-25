"""Cubic Bezier spline through a set of knots.

The jax counterpart of ``cfm.datasets.obstacle_paths.bezier_spline``; the two
are checked against each other in ``tests/test_spline.py``.
"""

import jax
import jax.numpy as jnp


def bezier_spline(knots: jax.Array, num_sub: int) -> jax.Array:
    """Evaluate a cubic Bezier spline passing through the given knots.

    The jax counterpart of ``datasets.obstacle_paths.bezier_spline``.
    Consecutive knots are joined by a cubic Bezier segment whose control
    points are set from Catmull-Rom tangents,

        m_i = (P_{i+1} - P_{i-1}) / 2,
        B_0 = P_i,  B_1 = P_i + m_i / 3,
        B_3 = P_{i+1},  B_2 = P_{i+1} - m_{i+1} / 3,

    (one-sided differences at the ends). The result interpolates every knot
    and is C1 continuous, so the path is smooth no matter where the knots
    land.

    Args:
        knots: Knot positions, shape ``(..., N, 2)``.
        num_sub: Number of samples per segment.

    Returns:
        Points along the spline, shape ``(..., num_sub * (N - 1) + 1, 2)``.
    """
    # Catmull-Rom tangents, with one-sided differences at the two ends.
    interior = 0.5 * (knots[..., 2:, :] - knots[..., :-2, :])
    first = knots[..., 1:2, :] - knots[..., 0:1, :]
    last = knots[..., -1:, :] - knots[..., -2:-1, :]
    tangents = jnp.concatenate([first, interior, last], axis=-2)

    # Bezier control points of each segment, shape (..., N - 1, 2).
    b0 = knots[..., :-1, :]
    b3 = knots[..., 1:, :]
    b1 = b0 + tangents[..., :-1, :] / 3.0
    b2 = b3 - tangents[..., 1:, :] / 3.0

    # Bernstein basis, evaluated at num_sub points per segment.
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
