import argparse
import jax
import jax.numpy as jnp
from pathlib import Path

from datasets.bimodal_distribution import BimodalDataset
from architectures.flow import FlowMLP
from training import train

import matplotlib.pyplot as plt
from flax import nnx
import cloudpickle

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train",
    action="store_true",
    help="Train the flow-matching model and save the trained parameters.",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="Load the trained model parameters and visualize generated samples.",
)
parser.add_argument(
    "--save-path",
    type=str,
    default="data/bimodal_model.pkl",
)
args = parser.parse_args()

if args.train:
    # Load the dataset
    dataset = BimodalDataset(num_samples=1024)

    # Create the flow model xdot = v(x, t)
    model = FlowMLP(
        data_size=2,
        time_embedding_size=4,
        hidden_sizes=(64, 64),
        rngs=nnx.Rngs(0),
    )

    # Train the model on the dataset
    model, normalizer = train(
        dataset=dataset,
        model=model,
        num_epochs=500,
        batch_size=64,
        learning_rate=1e-3,
        seed=0,
        print_frequency=10,
    )

    # Save the trained model parameters and normalizer stats
    print("Saving trained model and normalizer to", args.save_path)
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        cloudpickle.dump({"model": model, "normalizer": normalizer}, f)

if args.test:
    print("Loading trained model and normalizer from", args.save_path)
    with open(args.save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    print("Generating samples...")
    num_samples = 1000
    rng = jax.random.key(42)
    x = jax.random.normal(rng, (num_samples, 2))  # Initial gaussian noise
    dt = 0.01

    def _step_fn(x, t):
        """Single forward Euler step on the flow ODE xdot = v(x, t)."""
        t_reshaped = jnp.full((x.shape[0],), t)
        x_next = x + dt * model(x, t_reshaped)
        return x_next, x_next

    # Generate samples by integrating the flow ODE. This is a fast (compiled,
    # batched) JAX version of
    # for t in timesteps:
    #     x += dt * model(x, t)
    #     xs.append(x)
    timesteps = jnp.arange(0, 1.0, dt)
    x, xs = jax.lax.scan(_step_fn, x, timesteps)

    # Push data samples back to the original scale
    x = normalizer.unnormalize(x)

    # Plot the generated samples alongside training data
    print("Plotting")
    fig, ax = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    ax[0].set_title("Training Data")
    x_train = jnp.array(BimodalDataset().data)
    ax[0].scatter(x_train[:, 0], x_train[:, 1], alpha=0.5)
    ax[0].grid()
    ax[0].set_aspect("equal")
    ax[0].set_xlim(-10, 10)
    ax[0].set_ylim(-10, 10)

    ax[1].set_title("Generated Samples")
    ax[1].scatter(x[:, 0], x[:, 1], alpha=0.5)
    ax[1].grid()
    ax[1].set_aspect("equal")

    ax[2].set_title("Flow Trajectories")
    xs = normalizer.unnormalize(xs)
    ax[2].scatter(xs[0, :, 0], xs[0, :, 1], alpha=0.5, label="Initial Noise")
    ax[2].scatter(xs[-1, :, 0], xs[-1, :, 1], alpha=0.5, label="Final Samples")
    ax[2].plot(xs[:, 0:50, 0], xs[:, 0:50, 1], "k--")
    ax[2].grid()
    ax[2].set_aspect("equal")
    ax[2].legend()

    plt.tight_layout()
    plt.show()

if not args.train and not args.test:
    parser.print_help()
