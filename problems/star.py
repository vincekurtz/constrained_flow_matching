"""A five-pointed star.

Supports both constraint kinds: the unit-norm equality that pulls every
sample onto the circumscribed circle, and a right-half-plane inequality,
selected with ``--constraint``.
"""

from cfm.datasets.star import StarDataset
from problems import Problem, TrainConfig
from problems._shared import (
    mlp,
    right_half_constraint,
    scatter_plot,
    unit_circle_constraint,
)

CONSTRAINTS = {
    "circle": unit_circle_constraint,
    "right_half": right_half_constraint,
}


def make_constraint(constraint="circle", **_):
    """Pick between this problem's equality and inequality constraints."""
    try:
        return CONSTRAINTS[constraint]()
    except KeyError:
        raise KeyError(
            f"unknown constraint {constraint!r}; available: "
            f"{', '.join(sorted(CONSTRAINTS))}"
        ) from None


PROBLEM = Problem(
    name="star",
    label="Star",
    make_dataset=lambda: StarDataset(num_samples=1024),
    make_model=mlp((64, 64, 64, 64)),
    train=TrainConfig(num_epochs=5000, batch_size=256, print_frequency=100),
    make_constraint=make_constraint,
    plot=scatter_plot((-2, 2)),
    method_gains={
        "ldf": {"penalty_weight": 5.0, "num_projection_iters": 2},
        "penalty": {"penalty_weight": 5.0, "num_projection_iters": 2},
    },
    options={
        "--constraint": {
            "choices": sorted(CONSTRAINTS),
            "default": "circle",
            "help": "Which constraint to impose.",
        },
    },
)
