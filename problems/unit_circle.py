"""Points on the unit circle.

Supports both constraint kinds: the unit-norm equality the other 2-D
problems use, and a right-half-plane inequality, selected with
``--constraint``.
"""

from cfm.datasets.unit_circle import UnitCircleDataset
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
    name="unit_circle",
    label="Unit circle",
    make_dataset=lambda: UnitCircleDataset(num_samples=1024),
    make_model=mlp((64, 64)),
    train=TrainConfig(num_epochs=500, batch_size=64, print_frequency=10),
    make_constraint=make_constraint,
    plot=scatter_plot((-2, 2)),
    method_gains={
        "ldf": {"penalty_weight": 4.0},
        "penalty": {"penalty_weight": 4.0},
    },
    options={
        "--constraint": {
            "choices": sorted(CONSTRAINTS),
            "default": "circle",
            "help": "Which constraint to impose.",
        },
    },
)
