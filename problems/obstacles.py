"""Path planning around circular obstacles.

The model is trained unconditionally on start-to-goal paths. At inference,

    h(x) = r_j + clearance - ||p_i - c_j|| <= 0

for points p_i along the spline and obstacles (c_j, r_j).
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from cfm.core.constraints import inequality
from cfm.datasets.obstacle_paths import ObstaclePathDataset
from cfm.models.flow import FlowMLP
from cfm.plotting import plot_paths
from problems import Problem, TrainConfig
from problems.obstacle_scene import (
    GOAL,
    NUM_KNOTS,
    PLOT_SUBSAMPLE,
    SLACK_GAINS,
    START,
    make_constraint_fn,
    path,
    sample_scene,
)

# Last scene built by make_constraint, for plotting.
_LAST_SCENE = {}


def make_constraint(scene_seed=0, num_obstacles=2, **_):
    """Sample a fresh scene and build its avoidance constraint."""
    centers, radii = sample_scene(scene_seed, num_obstacles)
    _LAST_SCENE["obstacles"] = (centers, radii)
    return inequality(
        make_constraint_fn(centers, radii),
        name=f"{num_obstacles} obstacles (seed {scene_seed})",
    )


def make_model():
    return FlowMLP(
        data_shape=(NUM_KNOTS, 2),
        time_embedding_size=16,
        hidden_sizes=(256, 256, 256),
        rngs=nnx.Rngs(0),
    )


def report_violations(knots, centers, radii):
    """Print collision stats, checked more densely than the constraint."""
    dense = path(knots, 50)
    deltas = dense[:, :, None, :] - centers[None, None, :, :]
    dists = jnp.linalg.norm(deltas, axis=-1)
    penetration = radii[None, None, :] - dists
    worst = jnp.max(penetration, axis=(1, 2))

    num_nan = int(jnp.sum(jnp.isnan(worst)))
    if num_nan:
        print(f"  integration diverged for {num_nan}/{knots.shape[0]} paths")
    print(f"  paths hitting an obstacle: "
          f"{int(jnp.sum(worst > 0.0))}/{knots.shape[0]}")
    print(f"  deepest penetration:       {float(jnp.nanmax(worst)):.5f}")
    print(f"  mean penetration:          "
          f"{float(jnp.nanmean(jnp.maximum(worst, 0.0))):.5f}")


def plot(problem, samples, constraint=None, title="Constrained", **_):
    """Draw generated paths, with the obstacle scene if constrained."""
    obstacles = _LAST_SCENE.get("obstacles") if constraint else None

    if constraint is None:
        dataset = problem.make_dataset()
        _, ax = plt.subplots(1, 2, figsize=(11, 5.5))
        plot_paths(
            path(jnp.array(dataset.data.numpy()[:samples.x.shape[0]]),
                 PLOT_SUBSAMPLE),
            start=START, goal=GOAL, ax=ax[0], title="Training Data",
        )
        plot_paths(
            path(samples.x, PLOT_SUBSAMPLE),
            start=START, goal=GOAL, ax=ax[1], title="Generated Paths",
        )
    else:
        centers, radii = obstacles
        report_violations(samples.x, centers, radii)
        _, ax = plt.subplots(figsize=(6, 6))
        plot_paths(
            path(samples.x, PLOT_SUBSAMPLE), obstacles=obstacles,
            start=START, goal=GOAL, ax=ax, title=title,
        )
    plt.tight_layout()
    plt.show()


PROBLEM = Problem(
    name="obstacles",
    label="Obstacle avoidance",
    make_dataset=lambda: ObstaclePathDataset(
        num_samples=4096,
        num_knots=NUM_KNOTS,
        start=tuple(START.tolist()),
        goal=tuple(GOAL.tolist()),
    ),
    make_model=make_model,
    train=TrainConfig(num_epochs=1000, batch_size=256, print_frequency=50),
    make_constraint=make_constraint,
    plot=plot,
    method_gains={
        "ldf": dict(SLACK_GAINS["closed_form"], num_projection_iters=5),
        "penalty": {
            "penalty_weight": SLACK_GAINS["closed_form"]["penalty_weight"],
            "num_projection_iters": 5,
        },
        "cbf": {"qp": "exact"},
    },
    options={
        "--num-obstacles": {
            "type": int, "default": 2,
            "help": "Number of obstacles in the test scene.",
        },
        "--scene-seed": {
            "type": int, "default": 0,
            "help": "Random seed for the test scene.",
        },
    },
    default_num_samples=64,
    checkpoint="data/obstacle_model.pkl",
)
