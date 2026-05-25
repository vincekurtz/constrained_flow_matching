import argparse
from pathlib import Path

import cloudpickle
from flax import nnx
import jax.numpy as jnp

from architectures.flow import FlowMLP
from datasets.star import StarDataset
from examples.common import plot_2d
from generation import generate, generate_constrained
from pcfm import generate_pcfm
from pi_gdm import generate_pigdm
import training

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--generate_constrained", action="store_true")
parser.add_argument("--generate_pigdm", action="store_true")
parser.add_argument("--generate_pcfm", action="store_true")
parser.add_argument("--save-path", type=str, default="data/star_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = StarDataset(num_samples=1024)
model = FlowMLP(
    data_shape=(2,),
    time_embedding_size=4,
    hidden_sizes=(64, 64, 64, 64),
    rngs=nnx.Rngs(0),
)

if args.train:
    model, normalizer = training.train(
        dataset=dataset,
        model=model,
        num_epochs=5000,
        batch_size=256,
        learning_rate=1e-3,
        seed=0,
        print_frequency=100,
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
    x, xs = generate(model, normalizer, num_samples=1000, dt=0.01)
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
    x, xs = generate_constrained(
        model,
        normalizer,
        unit_circle_constraint,
        num_samples=1000,
        dt=0.01,
        penalty_weight=5.0,
        rescale_factor=1.0,
    )

    # Report constraint violation statistics.
    g = unit_circle_constraint(x)
    print(f"Constraint violation: mean={g.mean()}, max={g.max()}")

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
    x, xs = generate_pigdm(
        model,
        normalizer,
        unit_circle_constraint,
        num_samples=1000,
        dt=0.01,
        guidance_scale=1.0,
        eps_reg=1e-4,
    )

    g = jnp.abs(unit_circle_constraint(x))
    print(f"Constraint violation: mean={g.mean()}, max={g.max()}")

    plot_2d(dataset, x, xs, plot_lims=(-2, 2))

if args.generate_pcfm:
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    def unit_circle_constraint(x):
        """Constraint g(x) = ||x||^2 - 1, satisfied on the unit circle."""
        return jnp.sum(x**2, axis=-1) - 1.0

    print("Generating samples with PCFM unit circle projection...")
    x, xs = generate_pcfm(
        model,
        normalizer,
        unit_circle_constraint,
        num_samples=1000,
        num_steps=100,
    )

    g = jnp.abs(unit_circle_constraint(x))
    print(f"Constraint violation: mean={g.mean()}, max={g.max()}")

    plot_2d(dataset, x, xs, plot_lims=(-2, 2))

if not (args.train or args.generate or args.generate_constrained
        or args.generate_pigdm or args.generate_pcfm):
    parser.print_help()
