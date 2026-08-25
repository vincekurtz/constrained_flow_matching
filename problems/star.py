"""A five-pointed star, constrained onto the unit circle."""

from cfm.datasets.star import StarDataset
from problems import Problem, TrainConfig
from problems._shared import mlp, scatter_plot, unit_circle_constraint

PROBLEM = Problem(
    name="star",
    label="Star",
    make_dataset=lambda: StarDataset(num_samples=1024),
    make_model=mlp((64, 64, 64, 64)),
    train=TrainConfig(num_epochs=5000, batch_size=256, print_frequency=100),
    make_constraint=lambda **_: unit_circle_constraint(),
    plot=scatter_plot((-2, 2)),
    method_gains={
        "ldf": {"penalty_weight": 5.0, "num_projection_iters": 2},
        "penalty": {"penalty_weight": 5.0, "num_projection_iters": 2},
    },
)
