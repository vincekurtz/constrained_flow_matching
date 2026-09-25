"""Walker2D and Hopper problems from SafeFlowMatcher.

Trained unconditionally on D4RL medium-expert windows; the roof constraint
(see ``locomotion_spec``) is imposed at inference.
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

from cfm.core.constraints import Constraint
from cfm.datasets.d4rl_locomotion import D4RLWindowDataset
from cfm.models.temporal_unet import FlowTemporalUNet
from problems import Problem, TrainConfig
from problems.locomotion_spec import (
    HORIZON,
    SPECS,
    LocomotionSpec,
    height_residual,
    make_constraint,
    resolve,
)

MAX_WINDOWS = 32768
PLOT_WINDOWS = 64
REFERENCE_WINDOWS = 512
# Active rows land exactly on the boundary, so ignore rounding noise.
FEASIBLE_TOL = 1e-6

# Last roof built by build_constraint, for plotting.
_LAST_ROOF = {}


def make_dataset(spec: LocomotionSpec, max_windows=MAX_WINDOWS):
    """Training windows for one environment."""
    return D4RLWindowDataset(
        spec.filename,
        horizon=HORIZON,
        max_windows=max_windows,
        action_dim=spec.action_dim,
        obs_dim=spec.obs_dim,
    )


def make_model(spec: LocomotionSpec) -> FlowTemporalUNet:
    """Temporal U-Net over the horizon, transition entries as channels."""
    return FlowTemporalUNet(
        data_shape=(HORIZON, spec.transition_dim),
        time_embedding_size=32,
        channels=(64, 128, 256),
        rngs=nnx.Rngs(0),
    )


def build_constraint(
    spec: LocomotionSpec, height_limit=None, phi=None, **_
) -> Constraint:
    """Build the roof constraint and remember it for the plot."""
    limit, weight = resolve(spec, height_limit, phi)
    _LAST_ROOF[spec.name] = (limit, weight)
    return make_constraint(spec, limit, weight)


def report_violations(spec, x, height_limit, phi, reference=None):
    """Print roof violation stats.

    If ``reference`` training windows are given, also report how often the
    data breaks the roof and how many sample entries leave the data range.
    """
    x = np.asarray(x)
    residual = height_residual(
        x, spec.z_index, spec.vz_index, height_limit, phi
    )
    worst = np.max(residual, axis=-1)

    num_nan = int(np.sum(np.isnan(worst)))
    if num_nan:
        print(f"  integration diverged for {num_nan}/{len(worst)} windows")
    print(f"  windows above the roof:  {int(np.sum(worst > FEASIBLE_TOL))}"
          f"/{len(worst)}   (tolerance {FEASIBLE_TOL:g})")
    print(f"  timesteps above:         "
          f"{int(np.sum(residual > FEASIBLE_TOL))}/{residual.size}")
    print(f"  worst residual:          {float(np.nanmax(worst)):+.3e}")
    print(f"  mean positive residual:  "
          f"{float(np.nanmean(np.maximum(worst, 0.0))):.3e}")

    if reference is None:
        return
    ref = np.asarray(reference)
    ref_worst = np.max(
        height_residual(ref, spec.z_index, spec.vz_index, height_limit, phi),
        axis=-1,
    )
    print(f"  training windows above:  {np.mean(ref_worst > 0):.1%}"
          f"   (how hard the constraint is working)")
    lo, hi = ref.min(axis=0), ref.max(axis=0)
    outside = (x < lo) | (x > hi)
    print(f"  entries outside the training range: {np.mean(outside):.1%}"
          f"   (compare the unconstrained model)")


def _trace_panel(ax, traces, color, alpha, label=None):
    """One line per window against timestep."""
    steps = np.arange(traces.shape[1])
    for i, trace in enumerate(traces):
        ax.plot(steps, trace, color=color, alpha=alpha, lw=1.0,
                label=label if i == 0 else None)


def plot(problem, samples, constraint=None, spec=None, title="Constrained",
         **_):
    """Torso-height traces, the residual, and the (z, vz) phase plane.

    The roof on the height panel is not a cap on z alone; check the residual.
    """
    x = np.asarray(samples.x)
    reference = make_dataset(
        spec, max_windows=REFERENCE_WINDOWS
    ).windows().numpy()
    training = reference[:PLOT_WINDOWS]
    z_gen, z_ref = x[..., spec.z_index], training[..., spec.z_index]

    if constraint is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        _trace_panel(ax[0], z_ref, "0.4", 0.6)
        ax[0].set_title("Training Data")
        _trace_panel(ax[1], z_gen, "C0", 0.6)
        ax[1].set_title("Generated Windows")
        for a in ax:
            a.set_xlabel("timestep")
            a.grid(alpha=0.3)
        ax[0].set_ylabel("torso height $z$ (m)")
        plt.tight_layout()
        plt.show()
        return

    limit, weight = _LAST_ROOF.get(spec.name, resolve(spec))
    report_violations(spec, x, limit, weight, reference=reference)

    vz_gen, vz_ref = x[..., spec.vz_index], training[..., spec.vz_index]
    residual = height_residual(
        x, spec.z_index, spec.vz_index, limit, weight
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

    _trace_panel(ax[0], z_ref, "0.6", 0.4, label="training")
    _trace_panel(ax[0], z_gen, "C0", 0.7, label="generated")
    ax[0].axhline(limit, color="C3", ls="--", label=f"$h_r = {limit:g}$")
    ax[0].set_xlabel("timestep")
    ax[0].set_ylabel("torso height $z$ (m)")
    ax[0].set_title(f"{spec.label}: {title}")
    ax[0].legend(loc="lower right", fontsize=8)

    _trace_panel(ax[1], residual, "C0", 0.6)
    ax[1].axhline(0.0, color="C3", ls="--")
    ax[1].set_xlabel("timestep")
    ax[1].set_ylabel(r"$h(x)_t = z_t + \phi v_{z,t} - h_r$")
    ax[1].set_title("Constraint residual")

    ax[2].scatter(z_ref.ravel(), vz_ref.ravel(), s=3, c="0.6", alpha=0.3,
                  label="training")
    ax[2].scatter(z_gen.ravel(), vz_gen.ravel(), s=5, c="C0", alpha=0.6,
                  label="generated")
    # Boundary z + phi vz = h_r.
    vz_line = np.linspace(*ax[2].get_ylim(), 2)
    ax[2].plot(limit - weight * vz_line, vz_line, color="C3", ls="--")
    ax[2].set_xlabel("torso height $z$ (m)")
    ax[2].set_ylabel("vertical velocity $v_z$ (m/s)")
    ax[2].set_title("Phase plane")
    ax[2].legend(loc="upper right", fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def make_problem(spec: LocomotionSpec) -> Problem:
    """Registry entry for one environment. Must not touch the disk."""
    return Problem(
        name=spec.name,
        label=f"{spec.label} (medium-expert)",
        make_dataset=lambda: make_dataset(spec),
        make_model=lambda: make_model(spec),
        train=TrainConfig(
            num_epochs=200,
            batch_size=256,
            # Cosine decay gives noticeably smoother height traces.
            learning_rate=2e-4,
            schedule="cosine",
            ema_decay=0.999,
            print_frequency=10,
        ),
        make_constraint=partial(build_constraint, spec),
        plot=partial(plot, spec=spec),
        method_gains={
            # Affine residual: one Gauss-Newton step projects exactly.
            "ldf": {
                "penalty_weight": 20.0,
                "rescale_factor": 10.0,
                "num_projection_iters": 1,
            },
            "penalty": {
                "penalty_weight": 20.0,
                "num_projection_iters": 1,
            },
            "cbf": {"qp": "elastic"},
        },
        options={
            "--height-limit": {
                "type": float, "default": None,
                "help": "Roof h_r in z + phi*vz <= h_r. Defaults to the "
                        f"paper's {spec.height_limit:g} m.",
            },
            "--phi": {
                "type": float, "default": None,
                "help": "Vertical-velocity weight phi, in seconds. "
                        f"Defaults to {spec.phi:g}.",
            },
        },
        default_num_samples=64,
    )


PROBLEMS = tuple(make_problem(spec) for spec in SPECS.values())

__all__ = ["PROBLEMS", "build_constraint", "make_dataset", "make_model",
           "make_problem", "plot", "report_violations"]
