import argparse
from pathlib import Path
import time

import cloudpickle
import jax
import jax.numpy as jnp
from flax import nnx

from architectures.flow import FlowMLP
from datasets.unit_circle import UnitCircleDataset
from examples.common import plot_2d
from generation import (
    generate,
    generate_constrained,
    generate_inequality_constrained,
)
from pi_gdm import generate_pigdm
import training

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--generate_constrained", action="store_true")
parser.add_argument("--generate_inequality", action="store_true")
parser.add_argument("--generate_pigdm", action="store_true")
parser.add_argument(
    "--save-path", type=str, default="data/unit_circle_model.pkl"
)
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = UnitCircleDataset(num_samples=1024)
model = FlowMLP(
    data_shape=(2,),
    time_embedding_size=4,
    hidden_sizes=(64, 64),
    rngs=nnx.Rngs(0),
)

if args.train:
    model, normalizer = training.train(
        dataset=dataset,
        model=model,
        num_epochs=500,
        batch_size=64,
        learning_rate=1e-3,
        seed=0,
        print_frequency=10,
    )
    print("Saving trained model and normalizer to", save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        cloudpickle.dump({"model": model, "normalizer": normalizer}, f)

if args.generate:
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    print("Generating samples...")
    x, xs = generate(
        model, normalizer, num_samples=1000, dt=0.01
    )
    plot_2d(dataset, x, xs, plot_lims=(-2, 2))

if args.generate_constrained:
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    def unit_circle_constraint(x):
        """Constraint g(x) = ||x||^2 - 1, satisfied on the unit circle."""
        return jnp.sum(x**2, axis=-1) - 1.0

    print("Generating samples with unit circle constraint...")
    start_time = time.time()
    x, xs = generate_constrained(
        model,
        normalizer,
        unit_circle_constraint,
        num_samples=1000,
        dt=0.01,
        penalty_weight=5.0,
        rescale_factor=1.0,
    )
    jax.block_until_ready(x)
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds")

    # Report constraint violation statistics.
    g = jnp.abs(unit_circle_constraint(x))
    print(f"Constraint violation: mean={g.mean()}, max={g.max()}")

    plot_2d(dataset, x, xs, plot_lims=(-2, 2))

if args.generate_inequality:
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    def right_half_constraint(x):
        """Inequality h(x) = -x[0] <= 0, satisfied when x[0] > 0."""
        return -x[0]

    print("Generating samples with x[0] > 0 inequality constraint...")
    start_time = time.time()
    x, xs = generate_inequality_constrained(
        model,
        normalizer,
        right_half_constraint,
        num_samples=1000,
        dt=0.02,
        penalty_weight=5.0,
        rescale_factor=1.0,
    )
    jax.block_until_ready(x)
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds")

    # Report constraint violation statistics.
    h = jax.vmap(right_half_constraint)(x)
    n_violated = int(jnp.sum(h > 0))
    print(f"Samples violating x[0] > 0: {n_violated}/{x.shape[0]}")
    print(f"h(x): mean={float(h.mean()):.4f}, max={float(h.max()):.4f}")

    plot_2d(dataset, x, xs, plot_lims=(-2, 2))

if args.generate_pigdm:
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    def unit_circle_constraint(x):
        """Constraint g(x) = ||x||^2 - 1, satisfied on the unit circle."""
        return jnp.sum(x**2, axis=-1) - 1.0

    print("Generating samples with PiGDM unit circle guidance...")
    start_time = time.time()
    x, xs = generate_pigdm(
        model,
        normalizer,
        unit_circle_constraint,
        num_samples=1000,
        dt=0.01,
        guidance_scale=1.0,
        eps_reg=1e-4,
    )
    jax.block_until_ready(x)
    end_time = time.time()
    print(f"Generation took {end_time - start_time:.2f} seconds")

    g = jnp.abs(unit_circle_constraint(x))
    print(f"Constraint violation: mean={g.mean()}, max={g.max()}")

    plot_2d(dataset, x, xs, plot_lims=(-2, 2))

if not (args.train or args.generate or args.generate_constrained
        or args.generate_inequality or args.generate_pigdm):
    parser.print_help()
