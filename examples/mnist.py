import math

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from architectures.flow import FlowMLP
from architectures.unet import FlowUNet
from datasets.mnist import MNISTDataset
from examples.example_base import FlowExample

# Parse command line arguments (use --help to see options)
parser = FlowExample.build_arg_parser("data/mnist_model.pkl")
args = parser.parse_args()

# Define the architecture of the flow model we'll train.
model = FlowUNet(
    data_shape=(28, 28, 1),
    time_embedding_size=16,
    channels=(16, 32, 64),
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

# Override the base example's simple 2D plots
class MNISTExample(FlowExample):
    def plot(self, x: jax.Array, xs: jax.Array):
        """Visualize generated MNIST digits."""
        x = jnp.clip(x, 0.0, 1.0)
        n = math.isqrt(x.shape[0])
        fig, axes = plt.subplots(n, n, figsize=(n, n))
        for ax, i in zip(axes.flat, range(n * n)):
            ax.imshow(x[i].squeeze(-1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.suptitle("Generated MNIST Digits")
        plt.tight_layout()
        plt.show()


example = MNISTExample(
    dataset=MNISTDataset(train=True),
    model=model,
    save_path=args.save_path,
)
example.run(
    args,
    generate_num_samples=25,
    generate_dt=0.1,
    parser=parser,
    **hyperparams,
)
