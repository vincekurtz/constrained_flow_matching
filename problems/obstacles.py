"""Point-mass path planning around a field of circular obstacles.

The flow model is trained *unconditionally* on wiggly start-to-goal paths,
with no knowledge of any particular obstacle field. At inference time a new
scene is sampled and obstacle avoidance is imposed as inequalities

    h(x) = r_j + clearance - ||p_i - c_j|| <= 0

for every point p_i sampled along the path and every obstacle (c_j, r_j).

The decision variables are the interior knots of a cubic Bezier spline, so
the robot's path stays smooth however the constraints push the knots around.
The scene geometry lives in ``problems/obstacle_scene.py``.
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

# Remembered from the last make_constraint call so plotting can draw the same
# scene the samples were generated against.
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
    """Print how badly the generated paths hit the obstacles.

    Collisions are checked densely along the path, independently of the
    points the constraint is imposed on, so the numbers reflect the robot's
    actual swept path rather than the constraint residual.
    """
    dense = path(knots, 50)
    deltas = dense[:, :, None, :] - centers[None, None, :, :]
    dists = jnp.linalg.norm(deltas, axis=-1)
    penetration = radii[None, None, :] - dists  # > 0 means inside an obstacle
    worst = jnp.max(penetration, axis=(1, 2))  # per path

    num_nan = int(jnp.sum(jnp.isnan(worst)))
    if num_nan:
        print(f"  integration diverged for {num_nan}/{knots.shape[0]} paths")
    print(f"  paths hitting an obstacle: "
          f"{int(jnp.sum(worst > 0.0))}/{knots.shape[0]}")
    print(f"  deepest penetration:       {float(jnp.nanmax(worst)):.5f}")
    print(f"  mean penetration:          "
          f"{float(jnp.nanmean(jnp.maximum(worst, 0.0))):.5f}")


def plot(problem, samples, constraint=None, title="Constrained", **_):
    """Draw generated paths, against the obstacle scene when there is one."""
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
        # The obstacle scene puts many constraints in play at once and needs
        # much stiffer gains than the 2-D problems. SLACK_GAINS holds the
        # per-slack-mode values; closed_form is the default.
        "ldf": dict(SLACK_GAINS["closed_form"], num_projection_iters=5),
        # The ablation shares the penalty weight but not the multiplier
        # rescaling, which it fixes at zero by definition.
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
