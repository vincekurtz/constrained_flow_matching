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
from generation import generate
import training

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--save-path", type=str, default="data/mnist_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = MNISTDataset(train=True)
model = FlowUNet(
    data_shape=(28, 28, 1),
    time_embedding_size=128,
    channels=(64, 128, 256),
    rngs=nnx.Rngs(0),
)

if args.train:
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

if not args.train and not args.generate:
    parser.print_help()
