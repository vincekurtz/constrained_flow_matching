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
        unconstrained_gains: Defaults for the unconstrained sampler.
        method_constraint_options: Per-method constraint representations.
        options: Extra CLI options, as ``{flag: {argparse kwargs}}``.
        train: None for inference-only pretrained examples.
        supported_methods: Optional restriction for expensive examples.
        save_figure: Optional ``(problem, samples, constraint, **opts)``
            exporter accepting ``output_path`` and ``show=False``.
        retain_trajectory: Whether the CLI should retain trajectories.
            False keeps only a final snapshot for large image examples.
    """

    name: str
    label: str
    make_dataset: Optional[Callable[[], Any]]
    make_model: Callable[[], Any]
    train: Optional[TrainConfig]
    plot: Callable[..., None]
    make_constraint: Optional[Callable[..., Constraint]] = None
    method_gains: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    unconstrained_gains: Dict[str, Any] = field(default_factory=dict)
    method_constraint_options: Dict[str, Dict[str, Any]] = field(
        default_factory=dict
    )
    options: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    default_num_samples: int = 1000
    checkpoint: str = ""
    supported_methods: Optional[tuple[str, ...]] = None
    save_figure: Optional[Callable[..., None]] = None
    retain_trajectory: bool = True

    @property
    def checkpoint_path(self) -> str:
        return self.checkpoint or f"data/{self.name}_model.pkl"

    @property
    def constrained(self) -> bool:
        return self.make_constraint is not None

    def gains_for(self, method_name: str) -> Dict[str, Any]:
        """Gain overrides for one method, empty when it uses the defaults."""
        return dict(self.method_gains.get(method_name, {}))

    def constraint_for(self, method_name: str, **options) -> Constraint:
        """Build a constraint with the method's representation defaults."""
        return self.make_constraint(**{
            **self.method_constraint_options.get(method_name, {}), **options,
        })



def _registry() -> Dict[str, Problem]:
    # Lazy, so importing one problem doesn't import every dataset.
    from problems import (
        bimodal,
        imagenet,
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
        imagenet.PROBLEM,
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
