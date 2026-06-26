"""D3IL-style "Avoiding" task for constrained flow matching.

A 2D surrogate of the D3IL avoiding task (see ``datasets/avoiding.py``): we
train an unconditional flow-matching model over whole demonstration
*trajectories* that weave from a fixed start to a goal line around six pillars.
At inference we then impose a **new** hard constraint set the demonstrations
never respected (per HardFlow, arXiv:2511.08425, Fig. 4):

  1. an *enlarged* keep-out circle around the first pillar, and
  2. two half-plane "funnel" constraints that cut off the last two pillars.

``--generate`` samples from the unconstrained model (many trajectories violate
the new constraints); ``--generate_constrained`` adds smooth constraint guidance
so most trajectories satisfy them (enforcement is soft, not guaranteed 100%).

Note: we guide only the inequality (avoidance) set and let the learned model
handle start/goal.
"""

import argparse
from pathlib import Path
import time

import cloudpickle
import jax
import jax.numpy as jnp
from flax import nnx

from architectures.flow import FlowMLP
from datasets.avoiding import (
    AvoidingDataset,
    GOAL_Y,
    OBSTACLES,
    START,
    TRAJ_LEN,
)
from examples.common import plot_avoiding
from generation import generate, generate_constrained
import training

# --- Inference-time constraint set (NOT seen during training) ---------------
# Enlarged keep-out region around the first pillar (obstacle 0). The original
# pillar radius is 0.18; we expand the forbidden region to R_BIG.
_C0 = OBSTACLES[0][:2]
R_BIG = 0.20
FIRST_CIRCLE = (_C0[0], _C0[1], R_BIG)

# Two half-plane "funnel" constraints. Each is (a0, a1, b); the forbidden side
# is a0 * x + a1 * y - b > 0. Together they exclude the last two pillars (4 and
# 5) and narrow the admissible corridor toward the goal.
FUNNEL_LINES = ((1.0, 0.45, 0.9), (-1.0, 0.45, 0.9))

# Tolerance for declaring an inequality satisfied / the goal reached.
FEASIBILITY_TOL = 1e-3
GOAL_TOL = 0.1


def avoiding_constraint(x: jax.Array) -> jax.Array:
    """Inference-time inequality constraint h(x) <= 0 for a single trajectory.

    Args:
        x: A single *unnormalized* trajectory of shape ``(TRAJ_LEN, 2)``.

    Returns:
        A 1-D array of shape ``(TRAJ_LEN * 3,)`` stacking, per waypoint:
        the enlarged-circle keep-out and the two funnel half-planes. Each entry
        is ``<= 0`` exactly when that waypoint satisfies the constraint.
    """
    cx, cy, r_big = FIRST_CIRCLE
    circle = r_big ** 2 - ((x[:, 0] - cx) ** 2 + (x[:, 1] - cy) ** 2)
    parts = [circle]
    for a0, a1, b in FUNNEL_LINES:
        parts.append(a0 * x[:, 0] + a1 * x[:, 1] - b)
    return jnp.concatenate(parts)


def avoiding_penalty(x: jax.Array) -> jax.Array:
    """Soft-penalty *equality* form of the avoidance constraint.

    The inequality ``h(x) <= 0`` is equivalent to the equality ``relu(h(x)) =
    0``: the penalty is zero wherever the constraint is satisfied and equals the
    violation amount where it is not. Feeding this to ``generate_constrained``
    (the equality primal-dual flow) drives only the *violated* waypoints toward
    feasibility, which keeps the sampled trajectories smooth.
    """
    return jnp.maximum(avoiding_constraint(x), 0.0)


def _violation(x: jax.Array) -> jax.Array:
    """Scalar constraint violation max(0, max h) for one trajectory."""
    return jnp.maximum(jnp.max(avoiding_constraint(x)), 0.0)


def report_metrics(label: str, samples: jax.Array) -> None:
    """Print feasibility, violation, and goal-reach stats for a sample batch."""
    viol = jax.vmap(_violation)(samples)
    feasible = viol <= FEASIBILITY_TOL
    reached = jnp.abs(samples[:, -1, 1] - GOAL_Y) <= GOAL_TOL
    n = samples.shape[0]
    print(f"\n[{label}] over {n} trajectories")
    print(f"  feasibility rate: {float(feasible.mean()):.2%}")
    print(f"  max violation:    {float(viol.max()):.3e}  "
          f"(mean {float(viol.mean()):.3e})")
    print(f"  goal-reach rate:  {float(reached.mean()):.2%}")


def is_feasible(traj) -> bool:
    """Whether a single trajectory satisfies the inference constraint set."""
    return float(_violation(jnp.asarray(traj))) <= FEASIBILITY_TOL


def load_model(save_path: Path):
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    return data["model"], data["normalizer"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--generate_constrained", action="store_true")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--save-path", type=str,
                        default="data/avoiding_model.pkl")
    parser.add_argument("--fig-path", type=str, default="plots/avoiding.png")
    args = parser.parse_args()

    save_path = Path(args.save_path)

    if args.train:
        dataset = AvoidingDataset(num_samples=4096)
        model = FlowMLP(
            data_shape=(TRAJ_LEN, 2),
            time_embedding_size=16,
            hidden_sizes=(256, 256, 256),
            rngs=nnx.Rngs(0),
        )
        model, normalizer = training.train(
            dataset=dataset,
            model=model,
            num_epochs=300,
            batch_size=128,
            learning_rate=1e-3,
            seed=0,
            print_frequency=10,
        )
        print("Saving trained model and normalizer to", save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "wb") as f:
            cloudpickle.dump({"model": model, "normalizer": normalizer}, f)

    if args.generate:
        model, normalizer = load_model(save_path)
        print("Generating unconstrained trajectories...")
        start_time = time.time()
        x, _ = generate(model, normalizer, num_samples=args.num_samples,
                        dt=0.01)
        jax.block_until_ready(x)
        print(f"Generation took {time.time() - start_time:.2f} seconds")
        report_metrics("unconstrained", x)
        plot_avoiding(
            [("Unconstrained", x)],
            OBSTACLES, START, GOAL_Y, FIRST_CIRCLE, FUNNEL_LINES,
            feasible_fn=is_feasible, save_path=args.fig_path,
        )

    if args.generate_constrained:
        model, normalizer = load_model(save_path)

        print("Generating unconstrained trajectories (baseline panel)...")
        x_unc, _ = generate(model, normalizer, num_samples=args.num_samples,
                            dt=0.01)
        jax.block_until_ready(x_unc)
        report_metrics("unconstrained", x_unc)

        print("\nGenerating constrained trajectories "
              "(funnel + enlarged circle)...")
        start_time = time.time()
        x_con, _, _ = generate_constrained(
            model,
            normalizer,
            avoiding_penalty,
            num_samples=args.num_samples,
            dt=0.01,
            penalty_weight=20.0,
            rescale_factor=1.0,
            rescale_exponent=2.0,
        )
        jax.block_until_ready(x_con)
        print(f"Constrained generation took "
              f"{time.time() - start_time:.2f} seconds")
        report_metrics("constrained", x_con)

        plot_avoiding(
            [("Unconstrained", x_unc), ("Constrained", x_con)],
            OBSTACLES, START, GOAL_Y, FIRST_CIRCLE, FUNNEL_LINES,
            feasible_fn=is_feasible, save_path=args.fig_path,
        )

    if not (args.train or args.generate or args.generate_constrained):
        parser.print_help()


if __name__ == "__main__":
    main()
