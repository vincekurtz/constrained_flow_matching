import argparse
import math
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from architectures.unet import FlowUNet
from datasets.mnist import MNISTDataset
from generation import generate, generate_constrained
import training

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--generate_constrained", action="store_true")
parser.add_argument("--save-path", type=str, default="data/mnist_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)

if args.train:
    dataset = MNISTDataset(train=True)
    model = FlowUNet(
        data_shape=(28, 28, 1),
        time_embedding_size=128,
        channels=(64, 128, 256),
        rngs=nnx.Rngs(0),
    )
    model, normalizer = training.train(
        dataset=dataset,
        model=model,
        num_epochs=100,
        batch_size=128,
        learning_rate=1e-4,
        seed=0,
        print_frequency=1,
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
    x, xs = generate(model, normalizer, num_samples=25, dt=0.01)

    x = jnp.clip(x, 0.0, 1.0)
    n = math.isqrt(x.shape[0])
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for ax, i in zip(axes.flat, range(n * n)):
        ax.imshow(x[i].squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle("Generated MNIST Digits")
    plt.tight_layout()
    plt.show()

if args.generate_constrained:
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    # Pick a reference image from the test set as the inpainting target.
    dataset = MNISTDataset(train=False, digit=5)
    reference = jnp.array(dataset[0])  # (28, 28, 1)

    # Fix the top half of the image (rows 0-13).
    mask = jnp.zeros((28, 28, 1), dtype=bool).at[:14, :, :].set(True)
    observed_indices = jnp.where(mask.ravel())[0]
    y = reference.ravel()[observed_indices]

    # Build a selection matrix A of shape (n_observed, 784).
    n_pixels = 28 * 28 * 1
    A = jnp.eye(n_pixels)[observed_indices]

    def inpainting_constraint(x):
        """g(x) = A @ flatten(x) - y: observed pixels must match."""
        return A @ x.ravel() - y

    print("Generating constrained (inpainted) samples...")
    num_samples = 25
    x, xs = generate_constrained(
        model,
        normalizer,
        inpainting_constraint,
        num_samples=num_samples,
        dt=0.01,
        penalty_weight=20.0,
    )

    # Report constraint violation.
    violations = jnp.abs(jax.vmap(inpainting_constraint)(x))
    print(f"Constraint violation: mean={float(jnp.mean(violations)):.6f}, "
          f"max={float(jnp.max(violations)):.6f}")

    x = jnp.clip(x, 0.0, 1.0)
    n = math.isqrt(num_samples)
    fig, axes = plt.subplots(n, n + 1, figsize=(n + 1, n))

    # First column: reference image with the observed region highlighted.
    for row in range(n):
        ax = axes[row, 0]
        if row == 0:
            vis = jnp.where(mask, reference, 0.5 * reference)
            ax.imshow(vis.squeeze(-1), cmap="gray", vmin=0, vmax=1)
            ax.set_title("Ref", fontsize=7)
        else:
            ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])

    # Remaining columns: generated inpainted samples.
    for idx, ax in enumerate(axes[:, 1:].flat):
        if idx < num_samples:
            ax.imshow(x[idx].squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.suptitle("Inpainted MNIST (top half fixed)")
    plt.tight_layout()
    plt.show()

if not (args.train or args.generate or args.generate_constrained):
    parser.print_help()
