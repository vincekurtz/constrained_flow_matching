from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from architectures.flow import FlowMLP
from datasets.mnist import MNISTDataset
from examples.example_base import FlowExample

# Parse command line arguments (use --help to see options)
parser = FlowExample.build_arg_parser("data/mnist_model.pkl")
args = parser.parse_args()

# Define the architecture of the flow model we'll train.
model = FlowMLP(
    data_shape=(28, 28, 1),
    time_embedding_size=32,
    hidden_sizes=(512, 512, 512, 512),
    rngs=nnx.Rngs(0),
)

# Define training hyperparameters
hyperparams = {
    "num_epochs": 50,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "seed": 0,
    "print_frequency": 1,
}


def generate(save_path: str, num_samples: int = 25, dt: float = 0.01):
    """Load the saved model and display a grid of generated digit images."""
    print("Loading trained model and normalizer from", save_path)
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    print("Generating samples...")
    rng = jax.random.key(42)
    x = jax.random.normal(rng, (num_samples, 28, 28, 1))

    def _step_fn(x, t):
        """Single forward Euler step on the flow ODE xdot = v(x, t)."""
        t_batch = jnp.full((x.shape[0],), t)
        return x + dt * model(x, t_batch), None

    timesteps = jnp.arange(0.0, 1.0, dt)
    x, _ = jax.lax.scan(_step_fn, x, timesteps)

    x = normalizer.unnormalize(x)
    x = jnp.clip(x, 0.0, 1.0)

    print("Plotting")
    n = int(num_samples**0.5)
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    for ax, i in zip(axes.flat, range(num_samples)):
        ax.imshow(x[i].squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    plt.suptitle("Generated MNIST Digits")
    plt.tight_layout()
    plt.show()


if args.train:
    example = FlowExample(
        dataset=MNISTDataset(train=True),
        model=model,
        save_path=Path(args.save_path),
    )
    example.train(**hyperparams)

if args.generate:
    generate(args.save_path)

if not args.train and not args.generate:
    parser.print_help()
