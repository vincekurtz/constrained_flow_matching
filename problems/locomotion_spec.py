"""D4RL Walker2D/Hopper specs and the SafeFlowMatcher roof constraint.

    h(x)[t] = z_t + phi * vz_t - h_r <= 0

with torso height z and vertical velocity vz (SafeFlowMatcher App. D.2).
h_r and phi come from SafeDiffuser (arXiv 2306.00148). Unlike SafeDiffuser,
the residual is in raw units, so the numbers are not directly comparable.

Samples are ``(horizon, action_dim + obs_dim)`` windows, actions first.
"""

from dataclasses import dataclass

import jax

from cfm.core.constraints import Constraint, inequality

# Matches SafeFlowMatcher's config/locomotion.py. Must be a multiple of four.
HORIZON = 32


@dataclass(frozen=True)
class LocomotionSpec:
    """Per-environment settings.

    Attributes:
        z_obs_index: Torso height index within the observation.
        vz_obs_index: Torso vertical velocity index within the observation.
        height_limit: Roof ``h_r``, in metres.
        phi: Weight on the vertical velocity, in seconds.
    """

    name: str
    label: str
    env: str
    filename: str
    action_dim: int
    obs_dim: int
    z_obs_index: int
    vz_obs_index: int
    height_limit: float
    phi: float

    @property
    def transition_dim(self) -> int:
            return self.action_dim + self.obs_dim

    @property
    def z_index(self) -> int:
        """Column of the torso height in a sample."""
        return self.action_dim + self.z_obs_index

    @property
    def vz_index(self) -> int:
        """Column of the vertical velocity in a sample."""
        return self.action_dim + self.vz_obs_index


# Gymnasium observation layouts (x-position excluded).
WALKER2D = LocomotionSpec(
    name="walker2d",
    label="Walker2D",
    env="walker2d-medium-expert-v2",
    filename="walker2d_medium_expert-v2.hdf5",
    action_dim=6,
    obs_dim=17,
    z_obs_index=0,
    vz_obs_index=9,
    height_limit=1.4,
    phi=0.1,
)

HOPPER = LocomotionSpec(
    name="hopper",
    label="Hopper",
    env="hopper-medium-expert-v2",
    filename="hopper_medium_expert-v2.hdf5",
    action_dim=3,
    obs_dim=11,
    z_obs_index=0,
    vz_obs_index=6,
    height_limit=1.6,
    phi=0.1,
)

SPECS = {spec.name: spec for spec in (WALKER2D, HOPPER)}


def height_residual(
    x: jax.Array,
    z_index: int,
    vz_index: int,
    height_limit: float,
    phi: float,
) -> jax.Array:
    """Roof residual ``z + phi * vz - h_r``, shape ``(..., horizon)``."""
    return x[..., z_index] + phi * x[..., vz_index] - height_limit


def resolve(spec: LocomotionSpec, height_limit=None, phi=None):
    """Fill in spec defaults for roof parameters left as None."""
    limit = spec.height_limit if height_limit is None else float(height_limit)
    weight = spec.phi if phi is None else float(phi)
    return limit, weight


def make_constraint_fn(spec: LocomotionSpec, height_limit=None, phi=None):
    """Build ``h(x)`` for one sample, of shape ``(horizon,)``."""
    limit, weight = resolve(spec, height_limit, phi)

    def constraint_fn(x: jax.Array) -> jax.Array:
        return height_residual(x, spec.z_index, spec.vz_index, limit, weight)

    return constraint_fn


def make_constraint(
    spec: LocomotionSpec, height_limit=None, phi=None
) -> Constraint:
    """The roof as an inequality constraint, one row per timestep."""
    limit, weight = resolve(spec, height_limit, phi)
    return inequality(
        make_constraint_fn(spec, limit, weight),
        name=f"z + {weight:g} vz <= {limit:g}",
    )


def barrier(
    spec: LocomotionSpec, x: jax.Array, height_limit=None, phi=None
) -> jax.Array:
    """CBF-convention barrier ``h_r - z - phi * vz``, >= 0 when safe."""
    limit, weight = resolve(spec, height_limit, phi)
    return -height_residual(x, spec.z_index, spec.vz_index, limit, weight)


def heights(spec: LocomotionSpec, x: jax.Array) -> jax.Array:
    """Torso height traces, shape ``(..., horizon)``."""
    return x[..., spec.z_index]


def vertical_velocities(spec: LocomotionSpec, x: jax.Array) -> jax.Array:
    """Torso vertical velocity traces, shape ``(..., horizon)``."""
    return x[..., spec.vz_index]


__all__ = [
    "HORIZON",
    "HOPPER",
    "SPECS",
    "WALKER2D",
    "LocomotionSpec",
    "barrier",
    "height_residual",
    "heights",
    "make_constraint",
    "make_constraint_fn",
    "resolve",
    "vertical_velocities",
]
