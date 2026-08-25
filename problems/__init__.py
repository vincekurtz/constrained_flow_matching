"""The registry of example problems.

A problem is everything needed to train a model and then constrain it: the
dataset, the architecture, the training schedule, the constraint, how to draw
the result, and any gains that differ from a method's defaults.

Adding an example used to mean copying a ~200-line argparse script. It now
means adding one :class:`Problem` here; the CLI, the benchmark, the sweep and
the tests all read this registry.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from cfm.core.constraints import Constraint


@dataclass(frozen=True)
class TrainConfig:
    """Hyperparameters for one problem's training run."""

    num_epochs: int
    batch_size: int
    learning_rate: float = 1e-3
    seed: int = 0
    print_frequency: int = 10


@dataclass(frozen=True)
class Problem:
    """One example: a dataset, a model, and optionally a constraint.

    Attributes:
        name: Key used on the command line and in result files.
        label: Human-readable name for tables and figures.
        make_dataset: Builds the training dataset.
        make_model: Builds an untrained model.
        train: Training schedule.
        make_constraint: Builds the constraint, given any problem-specific
            options (e.g. the obstacle scene's seed). None for problems that
            only demonstrate unconstrained generation.
        plot: ``(problem, samples, constraint, **opts) -> None``. Draws the
            result.
        method_gains: Per-method gain overrides, keyed by method name.
        options: Extra CLI options this problem accepts, as
            ``{flag: {argparse kwargs}}``.
        default_num_samples: How many samples to draw when generating.
        checkpoint: Where the trained model lives.
    """

    name: str
    label: str
    make_dataset: Callable[[], Any]
    make_model: Callable[[], Any]
    train: TrainConfig
    plot: Callable[..., None]
    make_constraint: Optional[Callable[..., Constraint]] = None
    method_gains: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_num_samples: int = 1000
    checkpoint: str = ""

    @property
    def checkpoint_path(self) -> str:
        return self.checkpoint or f"data/{self.name}_model.pkl"

    @property
    def constrained(self) -> bool:
        return self.make_constraint is not None

    def gains_for(self, method_name: str) -> Dict[str, Any]:
        """Gain overrides for one method, empty when it uses the defaults."""
        return dict(self.method_gains.get(method_name, {}))


def _registry() -> Dict[str, Problem]:
    # Imported lazily and inside the function so that importing one problem
    # does not drag in every dataset (MNIST in particular touches the disk).
    from problems import bimodal, mnist, obstacles, spiral, star, unit_circle

    entries = (
        bimodal.PROBLEM,
        spiral.PROBLEM,
        star.PROBLEM,
        unit_circle.PROBLEM,
        mnist.PROBLEM,
        obstacles.PROBLEM,
    )
    return {p.name: p for p in entries}


_CACHE: Dict[str, Problem] = {}


def all_problems() -> Dict[str, Problem]:
    """Every registered problem, keyed by name."""
    if not _CACHE:
        _CACHE.update(_registry())
    return _CACHE


def get(name: str) -> Problem:
    """Look up a problem by name, with a helpful error for typos."""
    problems = all_problems()
    try:
        return problems[name]
    except KeyError:
        raise KeyError(
            f"unknown problem {name!r}; available: "
            f"{', '.join(sorted(problems))}"
        ) from None


def names():
    """Sorted names of every registered problem."""
    return sorted(all_problems())


__all__ = ["Problem", "TrainConfig", "all_problems", "get", "names"]
