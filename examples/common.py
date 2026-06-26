"""Shared plotting utilities for flow-matching examples."""

from typing import Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


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


def _draw_avoiding_scene(
    ax,
    obstacles: Sequence[Tuple[float, float, float]],
    start: Tuple[float, float],
    goal_y: float,
    first_circle: Tuple[float, float, float],
    funnel_lines: Sequence[Tuple[float, float, float]],
    plot_lims: Tuple[float, float],
):
    """Render the shared avoiding scene: pillars, inference constraints, goal.

    ``first_circle`` is ``(cx, cy, R_big)`` for the enlarged keep-out region and
    ``funnel_lines`` is a list of ``(a0, a1, b)`` half-planes whose forbidden
    side is ``a0 * x + a1 * y - b > 0``.
    """
    lo, hi = plot_lims

    # Shade the union of forbidden regions (enlarged circle + funnel) in red.
    gx, gy = np.meshgrid(np.linspace(lo, hi, 400), np.linspace(lo, hi, 400))
    cx, cy, r_big = first_circle
    forbidden = (gx - cx) ** 2 + (gy - cy) ** 2 < r_big ** 2
    for a0, a1, b in funnel_lines:
        forbidden = forbidden | (a0 * gx + a1 * gy - b > 0)
    ax.contourf(
        gx, gy, forbidden.astype(float), levels=[0.5, 1.5],
        colors=["#d62728"], alpha=0.12,
    )

    # Original training pillars.
    for ox, oy, orad in obstacles:
        ax.add_patch(plt.Circle((ox, oy), orad, color="0.55", zorder=2))

    # Enlarged first-circle keep-out boundary.
    ax.add_patch(plt.Circle(
        (cx, cy), r_big, fill=False, edgecolor="#d62728",
        linestyle="--", linewidth=1.5, zorder=3,
    ))

    # Funnel half-plane boundaries.
    xs = np.linspace(lo, hi, 2)
    for a0, a1, b in funnel_lines:
        if a1 != 0.0:
            ax.plot(xs, (b - a0 * xs) / a1, color="#d62728",
                    linewidth=1.5, zorder=3)

    ax.axhline(goal_y, color="green", linestyle="--", linewidth=1.5,
               zorder=3, label="goal line")
    ax.plot(start[0], start[1], "ks", markersize=8, zorder=4, label="start")

    ax.set_aspect("equal")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.2)


def plot_avoiding(
    samples_by_title: Sequence[Tuple[str, jax.Array]],
    obstacles: Sequence[Tuple[float, float, float]],
    start: Tuple[float, float],
    goal_y: float,
    first_circle: Tuple[float, float, float],
    funnel_lines: Sequence[Tuple[float, float, float]],
    feasible_fn: Callable[[jax.Array], bool],
    plot_lims: Tuple[float, float] = (-1.2, 1.2),
    save_path: Optional[str] = None,
):
    """Side-by-side avoiding figure, one panel per (title, samples) pair.

    Each ``samples`` array has shape ``(N, T, 2)``. Trajectories are coloured by
    feasibility (green if ``feasible_fn(traj)`` is True, red otherwise) over the
    shared scene drawn by ``_draw_avoiding_scene``. Mirrors HardFlow's Fig. 4.
    """
    n = len(samples_by_title)
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, 7), sharex=True, sharey=True, squeeze=False,
    )
    axes = axes[0]

    for ax, (title, samples) in zip(axes, samples_by_title):
        _draw_avoiding_scene(
            ax, obstacles, start, goal_y, first_circle, funnel_lines, plot_lims,
        )
        samples = np.asarray(samples)
        n_ok = 0
        for traj in samples:
            ok = bool(feasible_fn(traj))
            n_ok += ok
            ax.plot(traj[:, 0], traj[:, 1],
                    color="#2ca02c" if ok else "#d62728",
                    alpha=0.35, linewidth=1.0, zorder=5)
        rate = n_ok / max(len(samples), 1)
        ax.set_title(f"{title}\nfeasible: {n_ok}/{len(samples)} ({rate:.0%})")

    # De-duplicated legend on the first axis.
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {save_path}")
    plt.show()
