"""Constraint definitions and violation metrics."""

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
        fn: residual on a single unnormalized sample, scalar or 1-D.
        kind: ``"equality"`` (``fn(x) = 0``) or ``"inequality"``
            (``fn(x) <= 0``).
        violation: scalar violation of a single sample.
        name: label for reports and figures.
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
    """Equality constraint ``fn(x) = 0``, scored by max absolute residual."""
    if violation is None:
        def violation(x):
            return jnp.max(jnp.abs(jnp.atleast_1d(fn(x))))

    return Constraint(fn=fn, kind=EQUALITY, violation=violation, name=name)


def inequality(
    fn: Callable[[jax.Array], jax.Array],
    name: str = "",
    violation: Optional[Callable] = None,
) -> Constraint:
    """Inequality constraint ``fn(x) <= 0``, scored by max positive part."""
    if violation is None:
        def violation(x):
            return jnp.max(jnp.maximum(jnp.atleast_1d(fn(x)), 0.0))

    return Constraint(fn=fn, kind=INEQUALITY, violation=violation, name=name)
