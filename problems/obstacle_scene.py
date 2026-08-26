"""Geometry shared by the obstacle-avoidance example and the benchmark.

The path-planning scene lives here rather than in ``examples/obstacles.py`` so
that anything wanting the same constraint -- the example script, the
benchmark -- builds it from one definition. A scene is a field of circular
obstacles between a fixed start and goal; the decision variables are the
interior knots of a cubic Bezier spline, and the constraint keeps points
sampled along that spline out of every obstacle.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cfm.core.spline import bezier_spline

# Start and goal are the same for every path, so they are not decision
# variables: the model only generates the interior spline knots.
START = jnp.array([-1.0, 0.0])
GOAL = jnp.array([1.0, 0.0])
NUM_KNOTS = 10
CLEARANCE = 0.02

COLLISION_SUBSAMPLE = 8  # collision checks per spline segment
PLOT_SUBSAMPLE = 25  # samples per spline segment when drawing a path

# Gains tuned per slack mode. The closed-form solver stays stable at much
# stiffer settings than the slack ODE, which diverges well before it gets
# there.
SLACK_GAINS = {
    "closed_form": {"penalty_weight": 40.0, "rescale_factor": 10.0},
    "ode": {"penalty_weight": 5.0, "rescale_factor": 20.0},
}


def full_knots(knots: jax.Array) -> jax.Array:
    """Prepend the start and append the goal to a batch of interior knots."""
    lead = knots.shape[:-2]
    start = jnp.broadcast_to(START, lead + (1, 2))
    goal = jnp.broadcast_to(GOAL, lead + (1, 2))
    return jnp.concatenate([start, knots, goal], axis=-2)


def path(knots: jax.Array, num_sub: int) -> jax.Array:
    """The Bezier spline the robot follows, given a batch of interior knots.

    Args:
        knots: Interior knots, shape ``(..., T, 2)``.
        num_sub: Samples per spline segment.

    Returns:
        Points along the path, shape ``(..., num_sub * (T + 1) + 1, 2)``.
    """
    return bezier_spline(full_knots(knots), num_sub)


def sample_scene(seed: int, num_obstacles: int) -> Tuple[jax.Array, jax.Array]:
    """Sample a new field of obstacles between the start and the goal.

    Obstacles are rejection-sampled so that they do not overlap each other and
    do not crowd the start or the goal. This keeps the scene solvable: there is
    always a gap wide enough for the robot to slip through. A layout that
    cannot be completed is abandoned and the whole scene is resampled.

    Returns:
        centers: Obstacle centers, shape ``(M, 2)``.
        radii: Obstacle radii, shape ``(M,)``.
    """
    gap = 0.22  # minimum free space around each obstacle
    rng = np.random.default_rng(seed)
    start, goal = np.array(START), np.array(GOAL)

    for _ in range(100):  # scene attempts
        centers, radii = [], []
        for _ in range(200 * num_obstacles):  # placement attempts
            radius = rng.uniform(0.05, 0.3)
            s = rng.uniform(0.15, 0.85)
            offset = np.array(
                [rng.uniform(-0.1, 0.1), rng.uniform(-0.35, 0.35)]
            )
            center = start + s * (goal - start) + offset

            clear_of_endpoints = all(
                np.linalg.norm(center - p) > radius + gap
                for p in (start, goal)
            )
            clear_of_others = all(
                np.linalg.norm(center - c) > radius + r + gap
                for c, r in zip(centers, radii)
            )
            if clear_of_endpoints and clear_of_others:
                centers.append(center)
                radii.append(radius)
            if len(centers) == num_obstacles:
                return jnp.array(np.stack(centers)), jnp.array(radii)

    raise RuntimeError(
        f"Could not fit {num_obstacles} obstacles into the scene."
    )


def make_constraint_fn(
    centers: jax.Array, radii: jax.Array
) -> Callable[[jax.Array], jax.Array]:
    """Build the obstacle-avoidance constraint h(x) <= 0 for one scene."""

    def h(knots: jax.Array) -> jax.Array:
        """Signed penetration depth of the path into every obstacle.

        The constraint is imposed on points sampled along the spline, not on
        the knots: the curve between two knots is what the robot follows.
        """
        pts = path(knots, COLLISION_SUBSAMPLE)  # (P, 2)
        deltas = pts[:, None, :] - centers[None, :, :]  # (P, M, 2)

        # Smoothed norm: ||d|| is not differentiable at an obstacle center.
        dists = jnp.sqrt(jnp.sum(deltas**2, axis=-1) + 1e-8)  # (P, M)
        return (radii[None, :] + CLEARANCE - dists).ravel()

    return h
