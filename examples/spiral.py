import argparse
from pathlib import Path

import cloudpickle
from flax import nnx

from architectures.flow import FlowMLP
from datasets.spiral import SpiralDataset
from examples.common import plot_2d
from generation import generate
import training

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--generate", action="store_true")
parser.add_argument("--save-path", type=str, default="data/spiral_model.pkl")
args = parser.parse_args()

save_path = Path(args.save_path)
dataset = SpiralDataset(num_samples=1024)
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
    plot_2d(dataset, x, xs, plot_lims=(-4, 4))

if not args.train and not args.generate:
    parser.print_help()
