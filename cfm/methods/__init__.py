"""Registry of constrained generation methods."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet

from cfm.core.constraints import EQUALITY, INEQUALITY, Constraint
from cfm.core.solve import Samples
from cfm.methods import cbf, ldf, pcfm, pigdm

BOTH = frozenset({EQUALITY, INEQUALITY})


@dataclass(frozen=True)
class Method:
    """One constrained-generation algorithm.

    Attributes:
        name: Key used on the command line and in result files.
        label: Human-readable name for tables and figures.
        generate: ``(model, normalizer, constraint, **cfg) -> Samples``.
        supports: Constraint kinds this method can handle.
        defaults: Default hyperparameters.
        locked: Hyperparameters that config may not override.
    """

    name: str
    label: str
    generate: Callable[..., Samples]
    supports: FrozenSet[str]
    defaults: Dict[str, Any] = field(default_factory=dict)
    locked: Dict[str, Any] = field(default_factory=dict)

    def supports_constraint(self, constraint: Constraint) -> bool:
        return constraint.kind in self.supports

    def config(self, **overrides) -> Dict[str, Any]:
        """Resolve hyperparameters: defaults, then overrides, then locked.

        ``None`` overrides are dropped so argparse defaults mean "unset".
        """
        cfg = dict(self.defaults)
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        conflicts = [k for k in self.locked if cfg.get(k) != self.locked[k]
                     and k in overrides and overrides[k] is not None]
        cfg.update(self.locked)
        if conflicts:
            fixed = ", ".join(f"{k}={self.locked[k]}" for k in conflicts)
            raise ValueError(
                f"method {self.name!r} fixes {fixed} and cannot be overridden; "
                f"use --method ldf if you want to vary it."
            )
        return cfg

    def run(self, model, normalizer, constraint: Constraint, **overrides):
        """Generate samples, after checking the constraint kind is supported."""
        if not self.supports_constraint(constraint):
            raise ValueError(
                f"method {self.name!r} does not support {constraint.kind} "
                f"constraints (supports: {', '.join(sorted(self.supports))})"
            )
        return self.generate(
            model, normalizer, constraint, **self.config(**overrides)
        )


_LDF_DEFAULTS = {
    "penalty_weight": 5.0,
    "rescale_factor": 1.0,
    "rescale_exponent": 2.0,
    "slack": "closed_form",
    "num_projection_iters": 0,
}

LDF = Method(
    name="ldf",
    label="LDF (ours)",
    generate=ldf.generate,
    supports=BOTH,
    defaults=dict(_LDF_DEFAULTS),
)

# LDF with frozen (zero) multipliers, i.e. a pure quadratic penalty.
PENALTY = Method(
    name="penalty",
    label="Penalty only",
    generate=ldf.generate,
    supports=BOTH,
    defaults=dict(_LDF_DEFAULTS, rescale_factor=0.0),
    locked={"rescale_factor": 0.0},
)

PCFM = Method(
    name="pcfm",
    label="PCFM",
    generate=pcfm.generate,
    supports=frozenset({EQUALITY}),
    defaults={
        "num_steps": 100,
        "correction_weight": 0.1,
        "num_correction_iters": 10,
        "correction_lr": 0.1,
        "num_final_projection_iters": 20,
        "eps_reg": 1e-10,
    },
)

PIGDM = Method(
    name="pigdm",
    label="ΠGDM",
    generate=pigdm.generate,
    supports=frozenset({EQUALITY}),
    defaults={"guidance_scale": 1.0, "eps_reg": 1e-4},
)

CBF = Method(
    name="cbf",
    label="CBF safety filter (SafeFlow)",
    generate=cbf.generate,
    supports=frozenset({INEQUALITY}),
    defaults={
        "phi0": 1.0,
        "omega": 4.0,
        "qp": "elastic",
        "qp_penalty": 1e4,
        "num_terminal_iters": 20,
    },
)

METHODS: Dict[str, Method] = {
    m.name: m for m in (LDF, PENALTY, PCFM, PIGDM, CBF)
}

# Methods that take `dt`; PCFM takes `num_steps` instead.
USES_DT = frozenset({"ldf", "penalty", "pigdm", "cbf"})


def get(name: str) -> Method:
    """Look up a method by name."""
    try:
        return METHODS[name]
    except KeyError:
        raise KeyError(
            f"unknown method {name!r}; available: "
            f"{', '.join(sorted(METHODS))}"
        ) from None


def supporting(constraint: Constraint):
    """All methods that support this constraint."""
    return [m for m in METHODS.values() if m.supports_constraint(constraint)]


__all__ = [
    "METHODS", "Method", "LDF", "PENALTY", "PCFM", "PIGDM", "CBF",
    "USES_DT", "get", "supporting",
]
