"""Point-mass path planning around a field of circular obstacles.

The flow model is trained *unconditionally* on wiggly start-to-goal paths, with
no knowledge of any particular obstacle field. At inference time a brand-new
scene is sampled and obstacle avoidance is imposed with inequality constraints

    h(x) = r_j + clearance - ||p_i - c_j|| <= 0

for every point p_i on the path and every obstacle (c_j, r_j).
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
from examples.common import plot_paths
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
parser.add_argument("--save-path", type=str, default="data/obstacle_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = ObstaclePathDataset(num_samples=4096, num_waypoints=24)
model = FlowMLP(
    data_shape=(dataset.num_waypoints, 2),
    time_embedding_size=16,
    hidden_sizes=(256, 256, 256),
    rngs=nnx.Rngs(0),
)

# Start and goal are the same for every path, so they are not decision
# variables: the model only generates the interior waypoints.
START = jnp.array(dataset.start.numpy())
GOAL = jnp.array(dataset.goal.numpy())
CLEARANCE = 0.02
COLLISION_SUBSAMPLE = 8  # collision checks per path segment


def full_path(waypoints: jax.Array) -> jax.Array:
    """Prepend the start and append the goal to a batch of waypoints."""
    lead = waypoints.shape[:-2]
    start = jnp.broadcast_to(START, lead + (1, 2))
    goal = jnp.broadcast_to(GOAL, lead + (1, 2))
    return jnp.concatenate([start, waypoints, goal], axis=-2)


def interpolate(path: jax.Array, num_sub: int) -> jax.Array:
    """Resample a path with ``num_sub`` evenly spaced points per segment.

    Args:
        path: Path points, shape ``(..., N, 2)``.
        num_sub: Number of samples per segment. ``1`` returns the input.

    Returns:
        The densified path, shape ``(..., num_sub * (N - 1) + 1, 2)``.
    """
    alphas = jnp.linspace(0.0, 1.0, num_sub, endpoint=False)
    alphas = alphas.reshape((1,) * (path.ndim - 2) + (1, num_sub, 1))
    starts = path[..., :-1, None, :]
    ends = path[..., 1:, None, :]
    segments = starts * (1 - alphas) + ends * alphas
    segments = segments.reshape(path.shape[:-2] + (-1, 2))
    return jnp.concatenate([segments, path[..., -1:, :]], axis=-2)


def collision_points(waypoints: jax.Array) -> jax.Array:
    """Path points where obstacle avoidance is enforced.

    The path is densified before the check, because avoidance at the waypoints
    alone would let a straight segment between two of them cut straight
    through an obstacle.
    """
    return interpolate(full_path(waypoints), COLLISION_SUBSAMPLE)


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

    def h(waypoints: jax.Array) -> jax.Array:
        """Signed penetration depth of every path point into every obstacle."""
        pts = collision_points(waypoints)  # (P, 2)
        deltas = pts[:, None, :] - centers[None, :, :]  # (P, M, 2)

        # Smoothed norm: ||d|| is not differentiable at an obstacle center.
        dists = jnp.sqrt(jnp.sum(deltas**2, axis=-1) + 1e-8)  # (P, M)
        return (radii[None, :] + CLEARANCE - dists).ravel()

    return h


def report_violations(
    waypoints: jax.Array, centers: jax.Array, radii: jax.Array
) -> None:
    """Print how badly the generated paths hit the obstacles.

    Collisions are checked densely along the path, independently of the points
    the constraint is imposed on, so the numbers reflect the robot's actual
    swept path rather than the constraint residual.
    """
    dense = interpolate(full_path(waypoints), 50)
    deltas = dense[:, :, None, :] - centers[None, None, :, :]
    dists = jnp.linalg.norm(deltas, axis=-1)
    penetration = radii[None, None, :] - dists  # > 0 means inside an obstacle
    worst = jnp.max(penetration, axis=(1, 2))  # per path

    num_hit = int(jnp.sum(worst > 0.0))
    print(f"  paths hitting an obstacle: {num_hit}/{waypoints.shape[0]}")
    print(f"  deepest penetration:       {float(jnp.max(worst)):.5f}")
    print(
        f"  mean penetration:          "
        f"{float(jnp.mean(jnp.maximum(worst, 0.0))):.5f}"
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
        full_path(jnp.array(dataset.data.numpy()[:64])),
        start=START, goal=GOAL, ax=ax[0], title="Training Data",
    )
    plot_paths(
        full_path(x),
        start=START, goal=GOAL, ax=ax[1], title="Generated Paths",
    )
    plt.tight_layout()
    plt.show()

if args.generate_constrained:
    model, normalizer = load_model()

    centers, radii = sample_scene(args.scene_seed, args.num_obstacles)
    h = make_constraint_fn(centers, radii)

    print(f"Generating paths for a new {args.num_obstacles}-obstacle scene...")
    x, _ = generate_inequality_constrained(
        model,
        normalizer,
        h,
        num_samples=64,
        dt=0.002,
        penalty_weight=40.0,
        rescale_factor=10.0,
    )
    print("With obstacle constraints:")
    report_violations(x, centers, radii)

    x_unconstrained, _ = generate(model, normalizer, num_samples=64, dt=0.002)
    print("Unconstrained baseline:")
    report_violations(x_unconstrained, centers, radii)

    _, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    plot_paths(
        full_path(x_unconstrained), obstacles=(centers, radii),
        start=START, goal=GOAL, ax=ax[0], title="Unconstrained",
    )
    plot_paths(
        full_path(x), obstacles=(centers, radii),
        start=START, goal=GOAL, ax=ax[1], title="Obstacle Constraints",
    )
    plt.tight_layout()
    plt.show()

if not (args.train or args.generate or args.generate_constrained):
    parser.print_help()
