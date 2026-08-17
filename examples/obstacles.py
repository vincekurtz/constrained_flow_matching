"""Point-mass path planning around a field of circular obstacles.

The flow model is trained *unconditionally* on wiggly start-to-goal paths, with
no knowledge of any particular obstacle field. At inference time a brand-new
scene is sampled and obstacle avoidance is imposed with inequality constraints

    h(x) = r_j + clearance - ||p_i - c_j|| <= 0

for every point p_i sampled along the path and every obstacle (c_j, r_j).

The decision variables are the interior knots of a cubic Bezier spline, so
the robot's path is smooth however the constraints push the knots around.
"""

import argparse
from pathlib import Path

import cloudpickle
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from architectures.flow import FlowMLP
from datasets.obstacle_paths import ObstaclePathDataset
from examples.common import bezier_spline, plot_paths
from generation import generate, generate_inequality_constrained
import training

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--generate_constrained", action="store_true")
parser.add_argument(
    "--num-obstacles", type=int, default=2,
    help="Number of obstacles in the test scene."
)
parser.add_argument(
    "--scene-seed", type=int, default=0,
    help="Random seed for the test scene."
)
parser.add_argument(
    "--slack", choices=["closed_form", "ode"], default="closed_form",
    help="How the inequality solver handles the slack variable."
)
parser.add_argument(
    "--penalty-weight", type=float, default=None,
    help="Constraint penalty weight. Defaults to a per-slack-mode value."
)
parser.add_argument(
    "--rescale-factor", type=float, default=None,
    help="Multiplier flow rescaling. Defaults to a per-slack-mode value."
)
parser.add_argument("--save-path", type=str, default="data/obstacle_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = ObstaclePathDataset(num_samples=4096, num_knots=10)
model = FlowMLP(
    data_shape=(dataset.num_knots, 2),
    time_embedding_size=16,
    hidden_sizes=(256, 256, 256),
    rngs=nnx.Rngs(0),
)

# Start and goal are the same for every path, so they are not decision
# variables: the model only generates the interior spline knots.
START = jnp.array(dataset.start.numpy())
GOAL = jnp.array(dataset.goal.numpy())
CLEARANCE = 0.02

# Gains tuned per slack mode. The closed-form solver stays stable at much
# stiffer settings than the slack ODE, which diverges well before it gets
# there.
SLACK_GAINS = {
    "closed_form": {"penalty_weight": 40.0, "rescale_factor": 10.0},
    "ode": {"penalty_weight": 5.0, "rescale_factor": 20.0},
}
COLLISION_SUBSAMPLE = 8  # collision checks per spline segment
PLOT_SUBSAMPLE = 25  # samples per spline segment when drawing a path


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


def sample_scene(seed: int, num_obstacles: int):
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
            radius = rng.uniform(0.15, 0.3)
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


def make_constraint_fn(centers: jax.Array, radii: jax.Array):
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


def report_violations(
    knots: jax.Array, centers: jax.Array, radii: jax.Array
) -> None:
    """Print how badly the generated paths hit the obstacles.

    Collisions are checked densely along the path, independently of the points
    the constraint is imposed on, so the numbers reflect the robot's actual
    swept path rather than the constraint residual.
    """
    dense = path(knots, 50)
    deltas = dense[:, :, None, :] - centers[None, None, :, :]
    dists = jnp.linalg.norm(deltas, axis=-1)
    penetration = radii[None, None, :] - dists  # > 0 means inside an obstacle
    worst = jnp.max(penetration, axis=(1, 2))  # per path

    num_nan = int(jnp.sum(jnp.isnan(worst)))
    if num_nan:
        print(f"  integration diverged for {num_nan}/{knots.shape[0]} paths")

    num_hit = int(jnp.sum(worst > 0.0))
    print(f"  paths hitting an obstacle: {num_hit}/{knots.shape[0]}")
    print(f"  deepest penetration:       {float(jnp.nanmax(worst)):.5f}")
    print(
        f"  mean penetration:          "
        f"{float(jnp.nanmean(jnp.maximum(worst, 0.0))):.5f}"
    )


def load_model():
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    return data["model"], data["normalizer"]


if args.train:
    model, normalizer = training.train(
        dataset=dataset,
        model=model,
        num_epochs=1000,
        batch_size=256,
        learning_rate=1e-3,
        seed=0,
        print_frequency=50,
    )
    print("Saving trained model and normalizer to", save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        cloudpickle.dump({"model": model, "normalizer": normalizer}, f)

if args.generate:
    model, normalizer = load_model()

    print("Generating unconstrained paths...")
    x, _ = generate(model, normalizer, num_samples=64, dt=0.01)

    _, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_paths(
        path(jnp.array(dataset.data.numpy()[:64]), PLOT_SUBSAMPLE),
        start=START, goal=GOAL, ax=ax[0], title="Training Data",
    )
    plot_paths(
        path(x, PLOT_SUBSAMPLE),
        start=START, goal=GOAL, ax=ax[1], title="Generated Paths",
    )
    plt.tight_layout()
    plt.show()

if args.generate_constrained:
    model, normalizer = load_model()

    centers, radii = sample_scene(args.scene_seed, args.num_obstacles)
    h = make_constraint_fn(centers, radii)

    gains = SLACK_GAINS[args.slack]
    penalty_weight = args.penalty_weight or gains["penalty_weight"]
    rescale_factor = args.rescale_factor or gains["rescale_factor"]

    print(
        f"Generating paths for a new {args.num_obstacles}-obstacle scene "
        f"(slack={args.slack}, penalty_weight={penalty_weight}, "
        f"rescale_factor={rescale_factor})..."
    )
    x, _ = generate_inequality_constrained(
        model,
        normalizer,
        h,
        num_samples=64,
        dt=0.002,
        penalty_weight=penalty_weight,
        rescale_factor=rescale_factor,
        slack=args.slack,
    )
    print("With obstacle constraints:")
    report_violations(x, centers, radii)

    x_unconstrained, _ = generate(model, normalizer, num_samples=64, dt=0.002)
    print("Unconstrained baseline:")
    report_violations(x_unconstrained, centers, radii)

    _, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_paths(
        path(x_unconstrained, PLOT_SUBSAMPLE), obstacles=(centers, radii),
        start=START, goal=GOAL, ax=ax[0], title="Unconstrained",
    )
    plot_paths(
        path(x, PLOT_SUBSAMPLE), obstacles=(centers, radii),
        start=START, goal=GOAL, ax=ax[1], title="Obstacle Constraints",
    )
    plt.tight_layout()
    plt.show()

if not (args.train or args.generate or args.generate_constrained):
    parser.print_help()
