"""A mixture of two Gaussians. Unconstrained generation only."""

from cfm.datasets.bimodal_distribution import BimodalDataset
from problems import Problem, TrainConfig
from problems._shared import mlp, scatter_plot

PROBLEM = Problem(
    name="bimodal",
    label="Bimodal",
    make_dataset=lambda: BimodalDataset(num_samples=1024),
    make_model=mlp((64, 64)),
    train=TrainConfig(num_epochs=500, batch_size=64, print_frequency=10),
    plot=scatter_plot((-10, 10)),
)
