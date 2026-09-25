"""Registry of example problems."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from cfm.core.constraints import Constraint


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters.

    Attributes:
        schedule: Learning-rate schedule, one of ``cfm.training.SCHEDULES``.
        ema_decay: Parameter EMA decay, or None to keep the last iterate.
    """

    num_epochs: int
    batch_size: int
    learning_rate: float = 1e-3
    seed: int = 0
    print_frequency: int = 10
    schedule: str = "constant"
    ema_decay: Optional[float] = None


@dataclass(frozen=True)
class Problem:
    """One example: a dataset, a model, and optionally a constraint.

    Attributes:
        make_constraint: Builds the constraint from problem-specific options.
            None for unconstrained problems.
        plot: ``(problem, samples, constraint, **opts) -> None``.
        method_gains: Per-method gain overrides, keyed by method name.
        options: Extra CLI options, as ``{flag: {argparse kwargs}}``.
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
    # Lazy, so importing one problem doesn't import every dataset.
    from problems import (
        bimodal,
        locomotion,
        mnist,
        obstacles,
        spiral,
        star,
        unit_circle,
    )

    entries = (
        bimodal.PROBLEM,
        spiral.PROBLEM,
        star.PROBLEM,
        unit_circle.PROBLEM,
        mnist.PROBLEM,
        obstacles.PROBLEM,
        *locomotion.PROBLEMS,
    )
    return {p.name: p for p in entries}


_CACHE: Dict[str, Problem] = {}


def all_problems() -> Dict[str, Problem]:
    """Every registered problem, keyed by name."""
    if not _CACHE:
        _CACHE.update(_registry())
    return _CACHE


def get(name: str) -> Problem:
    """Look up a problem by name."""
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
