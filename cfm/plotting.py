"""Plot helpers shared by the problem definitions."""

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
