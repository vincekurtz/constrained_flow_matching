"""MNIST digits, with top-half inpainting as the constraint.

The inpainting setup -- reference digit, mask, selection matrix -- was
previously written out three times in ``examples/mnist.py`` and twice more in
the benchmark and the figures. It is defined once here.
"""

import math

import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from cfm.core.constraints import equality
from cfm.datasets.mnist import MNISTDataset
from cfm.models.unet import FlowUNet
from problems import Problem, TrainConfig

IMAGE_SHAPE = (28, 28, 1)
NUM_PIXELS = math.prod(IMAGE_SHAPE)
OBSERVED_ROWS = 14  # the top half is held fixed
REFERENCE_DIGIT = 5


def _reference_and_mask():
    """The digit being inpainted, and which of its pixels are observed."""
    dataset = MNISTDataset(train=False, digit=REFERENCE_DIGIT)
    reference = jnp.array(dataset[0])  # (28, 28, 1)
    mask = (
        jnp.zeros(IMAGE_SHAPE, dtype=bool).at[:OBSERVED_ROWS, :, :].set(True)
    )
    return reference, mask


def make_constraint(**_):
    """g(x) = A x - y: the observed pixels must match the reference."""
    reference, mask = _reference_and_mask()
    observed = jnp.where(mask.ravel())[0]
    y = reference.ravel()[observed]
    A = jnp.eye(NUM_PIXELS)[observed]

    return equality(
        lambda x: A @ x.ravel() - y,
        name=f"top {OBSERVED_ROWS} rows fixed",
    )


def make_model():
    return FlowUNet(
        data_shape=IMAGE_SHAPE,
        time_embedding_size=128,
        channels=(64, 128, 256),
        rngs=nnx.Rngs(0),
    )


def plot(problem, samples, constraint=None, **_):
    """Grid of generated digits, with the reference alongside when inpainting.

    This grid was copy-pasted three times in the old example script, once per
    method; it is drawn from ``samples`` here regardless of which method
    produced them.
    """
    x = jnp.clip(samples.x, 0.0, 1.0)
    num = x.shape[0]
    n = math.isqrt(num)

    if constraint is None:
        _, axes = plt.subplots(n, n, figsize=(n, n))
        for ax, i in zip(axes.flat, range(n * n)):
            ax.imshow(x[i].squeeze(-1), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
        plt.suptitle("Generated MNIST Digits")
        plt.tight_layout()
        plt.show()
        return

    reference, mask = _reference_and_mask()
    _, axes = plt.subplots(n, n + 1, figsize=(n + 1, n))

    # First column: the reference, with the unobserved half dimmed.
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

    for idx, ax in enumerate(axes[:, 1:].flat):
        if idx < num:
            ax.imshow(x[idx].squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    plt.suptitle("Inpainted MNIST (top half fixed)")
    plt.tight_layout()
    plt.show()


PROBLEM = Problem(
    name="mnist",
    label="MNIST",
    make_dataset=lambda: MNISTDataset(train=True),
    make_model=make_model,
    train=TrainConfig(
        num_epochs=100, batch_size=128, learning_rate=1e-4, print_frequency=1
    ),
    make_constraint=make_constraint,
    plot=plot,
    method_gains={
        "ldf": {"penalty_weight": 10.0, "num_projection_iters": 2},
        "penalty": {"penalty_weight": 10.0, "num_projection_iters": 2},
        "pcfm": {"correction_weight": 0.0},
    },
    default_num_samples=25,
)
