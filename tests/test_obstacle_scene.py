"""Invariants of the randomly generated obstacle scenes.

``sample_scene`` rejection-samples a layout that is meant to stay solvable --
obstacles that do not overlap each other and do not crowd the start or goal.
None of that was covered before, so a scene that quietly boxed in the robot
would have looked like a failure of the planner.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from problems.obstacle_scene import (
    CLEARANCE,
    COLLISION_SUBSAMPLE,
    GOAL,
    NUM_KNOTS,
    START,
    full_knots,
    make_constraint_fn,
    path,
    sample_scene,
)

SEEDS = [0, 1, 7, 13]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("num_obstacles", [1, 2, 3])
def test_scene_shapes(seed, num_obstacles):
    centers, radii = sample_scene(seed, num_obstacles)
    assert centers.shape == (num_obstacles, 2)
    assert radii.shape == (num_obstacles,)
    assert np.all(np.asarray(radii) > 0)


@pytest.mark.parametrize("seed", SEEDS)
def test_scene_is_deterministic(seed):
    a_c, a_r = sample_scene(seed, 3)
    b_c, b_r = sample_scene(seed, 3)
    np.testing.assert_array_equal(np.asarray(a_c), np.asarray(b_c))
    np.testing.assert_array_equal(np.asarray(a_r), np.asarray(b_r))


def test_different_seeds_give_different_scenes():
    a, _ = sample_scene(0, 2)
    b, _ = sample_scene(1, 2)
    assert not np.allclose(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("seed", SEEDS)
def test_obstacles_do_not_overlap(seed):
    centers, radii = sample_scene(seed, 3)
    centers, radii = np.asarray(centers), np.asarray(radii)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            gap = np.linalg.norm(centers[i] - centers[j])
            assert gap > radii[i] + radii[j], (
                f"obstacles {i} and {j} overlap in scene {seed}"
            )


@pytest.mark.parametrize("seed", SEEDS)
def test_start_and_goal_stay_clear(seed):
    """The endpoints are never swallowed by an obstacle."""
    centers, radii = sample_scene(seed, 3)
    for endpoint in (np.asarray(START), np.asarray(GOAL)):
        dists = np.linalg.norm(np.asarray(centers) - endpoint, axis=-1)
        assert np.all(dists > np.asarray(radii))


def test_raises_when_the_scene_cannot_be_filled():
    """An impossible request fails loudly rather than looping forever."""
    with pytest.raises(RuntimeError, match="Could not fit"):
        sample_scene(0, 16)


def test_fits_the_largest_feasible_scene():
    """The rejection sampler is not giving up early on a solvable layout."""
    centers, radii = sample_scene(0, 5)
    assert centers.shape == (5, 2)


def test_full_knots_pins_the_endpoints():
    interior = jnp.zeros((NUM_KNOTS, 2))
    knots = full_knots(interior)
    assert knots.shape == (NUM_KNOTS + 2, 2)
    np.testing.assert_allclose(np.asarray(knots[0]), np.asarray(START))
    np.testing.assert_allclose(np.asarray(knots[-1]), np.asarray(GOAL))


def test_full_knots_batched():
    interior = jnp.zeros((4, NUM_KNOTS, 2))
    knots = full_knots(interior)
    assert knots.shape == (4, NUM_KNOTS + 2, 2)
    np.testing.assert_allclose(np.asarray(knots[:, 0]),
                               np.broadcast_to(np.asarray(START), (4, 2)))


def test_path_starts_and_ends_at_the_endpoints():
    interior = jnp.zeros((NUM_KNOTS, 2))
    pts = path(interior, 8)
    np.testing.assert_allclose(np.asarray(pts[0]), np.asarray(START),
                               atol=1e-6)
    np.testing.assert_allclose(np.asarray(pts[-1]), np.asarray(GOAL),
                               atol=1e-6)


def test_constraint_is_positive_inside_an_obstacle():
    """h > 0 exactly where the path penetrates an obstacle."""
    centers = jnp.array([[0.0, 0.0]])
    radii = jnp.array([0.5])
    h = make_constraint_fn(centers, radii)

    # A straight line from start to goal runs through the origin.
    through = jnp.linspace(-0.8, 0.8, NUM_KNOTS).reshape(-1, 1)
    knots = jnp.concatenate([through, jnp.zeros_like(through)], axis=-1)
    assert float(jnp.max(h(knots))) > 0.0

    # Detouring far above it clears the obstacle.
    detour = knots.at[:, 1].set(3.0)
    assert float(jnp.max(h(detour))) < 0.0


def test_constraint_matches_penetration_plus_clearance():
    """h is exactly r + clearance - distance, on the path the robot follows.

    Checked against the sampled path rather than the knots, since that is
    what the constraint is imposed on.
    """
    centers = jnp.array([[0.0, 0.6]])
    radii = jnp.array([0.25])
    h = make_constraint_fn(centers, radii)

    knots = jnp.zeros((NUM_KNOTS, 2))
    worst = float(jnp.max(h(knots)))

    pts = np.asarray(path(knots, COLLISION_SUBSAMPLE))
    closest = np.min(np.linalg.norm(pts - np.asarray(centers)[0], axis=-1))
    expected = float(radii[0]) + CLEARANCE - closest

    assert worst == pytest.approx(expected, abs=1e-5)


def test_clearance_pushes_the_boundary_out():
    """The margin makes a just-touching path count as violating."""
    centers = jnp.array([[0.0, 0.6]])
    h = make_constraint_fn(centers, jnp.array([0.25]))
    knots = jnp.zeros((NUM_KNOTS, 2))
    pts = np.asarray(path(knots, COLLISION_SUBSAMPLE))
    closest = np.min(np.linalg.norm(pts - np.asarray(centers)[0], axis=-1))

    # Shrink the obstacle so the path exactly touches its surface; the
    # clearance margin means h is still positive, by exactly CLEARANCE.
    touching = make_constraint_fn(centers, jnp.array([closest]))
    assert float(jnp.max(touching(knots))) == pytest.approx(
        CLEARANCE, abs=1e-5
    )
    assert float(jnp.max(h(knots))) < float(jnp.max(touching(knots)))
