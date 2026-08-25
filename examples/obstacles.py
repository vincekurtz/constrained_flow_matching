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
import diffrax
from flax import nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architectures.flow import FlowMLP
from cbf import generate_cbf
from datasets.obstacle_paths import ObstaclePathDataset
from examples.common import plot_paths
from examples.obstacle_scene import (
    GOAL,
    NUM_KNOTS,
    PLOT_SUBSAMPLE,
    SLACK_GAINS,
    START,
    make_constraint_fn,
    path,
    sample_scene,
)
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
    "--method", choices=["dual", "cbf"], default="dual",
    help="Constrained sampler: Lagrangian dual flow, or the SafeFlow CBF "
         "safety filter baseline."
)
parser.add_argument(
    "--slack", choices=["closed_form", "ode"], default="closed_form",
    help="How the inequality solver handles the slack variable (--method "
         "dual only)."
)
parser.add_argument(
    "--phi0", type=float, default=1.0,
    help="CBF class-K gain used where the sample is feasible."
)
parser.add_argument(
    "--omega", type=float, default=4.0,
    help="CBF blow-up gain omega / (1 - t)^2 used where the sample is "
         "infeasible."
)
parser.add_argument(
    "--qp", choices=["exact", "elastic"], default="exact",
    help="Solve the CBF quadratic program as written, or in its elastic "
         "relaxation, which tolerates mutually infeasible barrier conditions."
)
parser.add_argument(
    "--qp-penalty", type=float, default=1e4,
    help="Price per unit of barrier-condition violation (--qp elastic only)."
)
parser.add_argument(
    "--adaptive", action="store_true",
    help="Integrate the CBF flow with Tsit5 and PID error control instead of "
         "fixed midpoint steps, as in SafeFlow's Algorithm 1."
)
parser.add_argument(
    "--penalty-weight", type=float, default=None,
    help="Constraint penalty weight. Defaults to a per-slack-mode value "
         "(--method dual only)."
)
parser.add_argument(
    "--rescale-factor", type=float, default=None,
    help="Multiplier flow rescaling. Defaults to a per-slack-mode value "
         "(--method dual only)."
)
parser.add_argument("--save-path", type=str, default="data/obstacle_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = ObstaclePathDataset(
    num_samples=4096,
    num_knots=NUM_KNOTS,
    start=tuple(START.tolist()),
    goal=tuple(GOAL.tolist()),
)
model = FlowMLP(
    data_shape=(dataset.num_knots, 2),
    time_embedding_size=16,
    hidden_sizes=(256, 256, 256),
    rngs=nnx.Rngs(0),
)


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

    if args.method == "cbf":
        print(
            f"Generating paths for a new {args.num_obstacles}-obstacle scene "
            f"(method=cbf, qp={args.qp}, phi0={args.phi0}, "
            f"omega={args.omega})..."
        )
        # SafeFlow's Algorithm 1 integrates with an embedded RK pair and
        # error control, which --adaptive reproduces. The omega / (1 - t)^2
        # schedule is genuinely stiff at the end of the horizon, so the fixed
        # midpoint steps used elsewhere in the repository are not always
        # enough to hold on to it.
        integrator = (
            {
                "solver": diffrax.Tsit5(),
                "stepsize_controller": diffrax.PIDController(
                    rtol=1e-5, atol=1e-7
                ),
            }
            if args.adaptive
            else {}
        )
        x, _ = generate_cbf(
            model,
            normalizer,
            h,
            num_samples=64,
            dt=0.01,
            phi0=args.phi0,
            omega=args.omega,
            qp=args.qp,
            qp_penalty=args.qp_penalty,
            **integrator,
        )
        title = "CBF Safety Filter"
    else:
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
        title = "Obstacle Constraints"
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
        start=START, goal=GOAL, ax=ax[1], title=title,
    )
    plt.tight_layout()
    plt.show()

if not (args.train or args.generate or args.generate_constrained):
    parser.print_help()
