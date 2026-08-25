"""Constraints, and how to measure violating one.

A constraint used to be just a bare function, with the question of whether it
was an equality or an inequality answered by a hand-maintained table in
``benchmark.py``, and the question of how to score a violation answered
separately (and differently) in ``benchmark.py`` and ``plots.py``.

Bundling the three together means a problem states its constraint once and
the runner, the figures and the tests all agree by construction.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp

EQUALITY = "equality"
INEQUALITY = "inequality"
KINDS = (EQUALITY, INEQUALITY)


@dataclass(frozen=True)
class Constraint:
    """A constraint on generated samples.

    Attributes:
        fn: Differentiable residual on a single *unnormalized* sample. May
            return a scalar or a 1-D array.
        kind: ``"equality"`` for ``fn(x) == 0``, ``"inequality"`` for
            ``fn(x) <= 0``.
        violation: Maps a single sample to a scalar violation magnitude.
        name: Human-readable description, used in reports and figures.
    """

    fn: Callable[[jax.Array], jax.Array]
    kind: str
    violation: Callable[[jax.Array], jax.Array]
    name: str = ""

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")

    def violations(self, x: jax.Array) -> jax.Array:
        """Per-sample violation for a batch of samples."""
        return jax.vmap(self.violation)(x)


def equality(
    fn: Callable[[jax.Array], jax.Array],
    name: str = "",
    violation: Optional[Callable] = None,
) -> Constraint:
    """An equality constraint ``fn(x) = 0``.

    Violation is the largest absolute residual, which for a scalar residual
    is just its magnitude.
    """
    if violation is None:
        def violation(x):
            return jnp.max(jnp.abs(jnp.atleast_1d(fn(x))))

    return Constraint(fn=fn, kind=EQUALITY, violation=violation, name=name)


def inequality(
    fn: Callable[[jax.Array], jax.Array],
    name: str = "",
    violation: Optional[Callable] = None,
) -> Constraint:
    """An inequality constraint ``fn(x) <= 0``.

    Violation is the amount by which the constraint is exceeded, and is zero
    for any feasible sample.
    """
    if violation is None:
        def violation(x):
            return jnp.max(jnp.maximum(jnp.atleast_1d(fn(x)), 0.0))

    return Constraint(fn=fn, kind=INEQUALITY, violation=violation, name=name)
