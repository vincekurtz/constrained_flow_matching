"""The D4RL locomotion tasks and their roof constraint.

Two examples from the SafeFlowMatcher paper (arXiv 2509.24243), Walker2D and
Hopper. A flow model is trained *unconditionally* on windows of the
medium-expert demonstrations, and at inference time a single speed-dependent
ceiling is imposed at every timestep of the window:

    h(x)[t] = z_t + phi * vz_t - h_r <= 0,

where ``z`` is the torso height, ``vz`` its vertical velocity, ``h_r`` the
roof and ``phi`` a lookahead weight. That is the barrier of SafeFlowMatcher
Appendix D.2. The paper states the form but no numbers, so ``h_r`` and ``phi``
come from SafeDiffuser (arXiv 2306.00148), whose locomotion setup
SafeFlowMatcher says it reuses: ``invariance_cpx`` and
``invariance_hopper_cpx`` in its ``diffuser/models/diffusion.py``.

One deliberate difference from SafeDiffuser: they convert ``h_r`` and ``z``
into normalized units but then apply ``phi`` to an already-normalized ``vz``,
which mixes units. We write the whole residual in raw metres and metres per
second -- ``cfm.core.solve.wrap_constraint`` unnormalizes before calling it --
so our numbers are not directly comparable to theirs.

A sample is a trajectory window in the Diffuser layout: ``(horizon,
action_dim + obs_dim)``, actions first, then the observation. That is why the
column a state lives in is ``action_dim`` plus its index in the observation.
"""

from dataclasses import dataclass

import jax

from cfm.core.constraints import Constraint, inequality

# Window length, matching SafeFlowMatcher's own config/locomotion.py. The
# H=600 quoted in the paper's tables would also fit the temporal U-Net the
# problem trains, but costs proportionally more to train and to integrate.
# Any multiple of four works; see cfm/models/temporal_unet.py.
HORIZON = 32


@dataclass(frozen=True)
class LocomotionSpec:
    """Everything that differs between the Walker2D and Hopper examples.

    Attributes:
        name: Problem name, used on the command line.
        label: Human-readable name for figures.
        env: D4RL environment id, for reference.
        filename: Name of the dataset file on the mirror.
        action_dim: Width of the action block, which comes first in a sample.
        obs_dim: Width of the observation block.
        z_obs_index: Index of the torso height *within the observation*.
        vz_obs_index: Index of the torso vertical velocity within the
            observation.
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
        """Width of one timestep of a sample."""
        return self.action_dim + self.obs_dim

    @property
    def z_index(self) -> int:
        """Column of the torso height in a sample, actions included."""
        return self.action_dim + self.z_obs_index

    @property
    def vz_index(self) -> int:
        """Column of the vertical velocity in a sample."""
        return self.action_dim + self.vz_obs_index


# Observation layouts are Gymnasium's, with the x-position excluded: obs[0] is
# rootz for both, obs[9] is the Walker2D torso z-velocity and obs[6] the
# Hopper one.
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
    """The roof residual ``z + phi * vz - h_r``, one entry per timestep.

    Indexed on the last axis, so this serves both the per-sample constraint
    and batched reporting.

    Args:
        x: Trajectory windows, shape ``(..., horizon, transition_dim)``.
        z_index: Column holding the torso height.
        vz_index: Column holding the vertical velocity.
        height_limit: Roof ``h_r``.
        phi: Weight on the vertical velocity.

    Returns:
        Residuals of shape ``(..., horizon)``, non-positive where safe.
    """
    return x[..., z_index] + phi * x[..., vz_index] - height_limit


def resolve(spec: LocomotionSpec, height_limit=None, phi=None):
    """Fill in a spec's defaults for any roof parameter left as None."""
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
    """The roof as an inequality constraint, one row per timestep.

    Pure: no disk, no model, no download. The problem registry builds this on
    every CLI invocation, so it has to stay that way.
    """
    limit, weight = resolve(spec, height_limit, phi)
    return inequality(
        make_constraint_fn(spec, limit, weight),
        name=f"z + {weight:g} vz <= {limit:g}",
    )


def barrier(
    spec: LocomotionSpec, x: jax.Array, height_limit=None, phi=None
) -> jax.Array:
    """The CBF-convention barrier ``h_r - z - phi * vz``, non-negative when
    safe. The negation of the residual, kept for figures that follow the
    paper's sign convention."""
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
