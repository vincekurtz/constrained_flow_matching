"""Constraints and plots reused across the 2-D problems."""

import jax.numpy as jnp
from flax import nnx

from cfm.core.constraints import equality, inequality
from cfm.models.flow import FlowMLP
from cfm.plotting import plot_2d


def mlp(hidden_sizes, time_embedding_size=4, data_shape=(2,)):
    """Factory for the small MLP flow model the 2-D problems share."""
    def make():
        return FlowMLP(
            data_shape=data_shape,
            time_embedding_size=time_embedding_size,
            hidden_sizes=hidden_sizes,
            rngs=nnx.Rngs(0),
        )
    return make


def unit_circle_constraint():
    """g(x) = ||x||^2 - 1, satisfied on the unit circle.

    Written once here rather than six times across the example scripts, so
    the benchmark, the figures and the examples cannot drift apart.
    """
    return equality(
        lambda x: jnp.sum(x**2, axis=-1) - 1.0, name="||x|| = 1"
    )


def right_half_constraint():
    """h(x) = -x[0] <= 0, satisfied on the right half plane."""
    return inequality(lambda x: jnp.atleast_1d(-x[0]), name="x[0] >= 0")


def scatter_plot(plot_lims):
    """A three-panel scatter/trajectory plot with fixed axis limits."""
    def plot(problem, samples, constraint=None, **_):
        plot_2d(
            problem.make_dataset(), samples.x, samples.xs,
            plot_lims=plot_lims,
        )
    return plot
