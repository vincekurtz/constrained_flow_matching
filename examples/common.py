"""Shared plotting utilities for flow-matching examples."""

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
