"""Shared plotting and path utilities for flow-matching examples."""

from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


def plot_2d(
    dataset,
    x: jax.Array,
    xs: jax.Array,
    plot_lims: Tuple[float, float] = (-3.0, 3.0),
):
    """Three-panel scatter / trajectory plot for 2-D generated samples.

    Args:
        dataset: Training dataset with a ``.data`` attribute.
        x: Final generated samples, shape ``(num_samples, 2)``.
        xs: Full trajectory, shape ``(num_steps, num_samples, 2)``.
        plot_lims: (lo, hi) axis limits applied to the training-data panel.
    """
    assert x.ndim == 2 and x.shape[1] == 2, "plot_2d only supports 2-D data"

    lo, hi = plot_lims
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    ax[0].set_title("Training Data")
    x_train = jnp.array(dataset.data)
    ax[0].scatter(x_train[:, 0], x_train[:, 1], alpha=0.5)
    ax[0].grid()
    ax[0].set_aspect("equal")
    ax[0].set_xlim(lo, hi)
    ax[0].set_ylim(lo, hi)

    ax[1].set_title("Generated Samples")
    ax[1].scatter(x[:, 0], x[:, 1], alpha=0.5)
    ax[1].grid()
    ax[1].set_aspect("equal")

    ax[2].set_title("Flow Trajectories")
    ax[2].scatter(xs[0, :, 0], xs[0, :, 1], alpha=0.5, label="Initial Noise")
    ax[2].scatter(xs[-1, :, 0], xs[-1, :, 1], alpha=0.5, label="Final Samples")
    ax[2].plot(xs[:, 0:50, 0], xs[:, 0:50, 1], "k--")
    ax[2].grid()
    ax[2].set_aspect("equal")
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def plot_paths(
    paths: jax.Array,
    obstacles=None,
    start=None,
    goal=None,
    ax=None,
    title: str = "",
    color: str = "C0",
    alpha: float = 0.5,
    plot_lims: Tuple[float, float] = (-1.6, 1.6),
):
    """Plot a batch of 2-D robot paths over a field of circular obstacles.

    Args:
        paths: Robot paths, shape ``(num_paths, num_points, 2)``.
        obstacles: Optional ``(centers, radii)`` describing the circular
            obstacles, with shapes ``(M, 2)`` and ``(M,)``.
        start: Optional start position to mark, shape ``(2,)``.
        goal: Optional goal position to mark, shape ``(2,)``.
        ax: Optional matplotlib axes to draw on. A new figure is created when
            this is not given.
        title: Title for the axes.
        color: Color of the paths.
        alpha: Opacity of the paths.
        plot_lims: (lo, hi) limits applied to both axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if obstacles is not None:
        centers, radii = obstacles
        for center, radius in zip(centers, radii):
            ax.add_patch(
                plt.Circle(tuple(center), float(radius), color="0.6", zorder=1)
            )

    for path in paths:
        ax.plot(path[:, 0], path[:, 1], color=color, alpha=alpha, lw=1,
                zorder=2)

    if start is not None:
        ax.plot(start[0], start[1], "o", color="C2", ms=10, zorder=3,
                label="start")
    if goal is not None:
        ax.plot(goal[0], goal[1], "*", color="C3", ms=16, zorder=3,
                label="goal")
    if start is not None or goal is not None:
        ax.legend(loc="upper right")

    lo, hi = plot_lims
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    return ax


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
