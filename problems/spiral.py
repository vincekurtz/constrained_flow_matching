"""A 2-D spiral. Unconstrained generation only."""

from cfm.datasets.spiral import SpiralDataset
from problems import Problem, TrainConfig
from problems._shared import mlp, scatter_plot

PROBLEM = Problem(
    name="spiral",
    label="Spiral",
    make_dataset=lambda: SpiralDataset(num_samples=1024),
    make_model=mlp((64, 64, 64, 64)),
    train=TrainConfig(num_epochs=5000, batch_size=256, print_frequency=100),
    plot=scatter_plot((-4, 4)),
)
