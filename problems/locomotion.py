"""The Walker2D and Hopper examples from the SafeFlowMatcher paper.

A flow model is trained *unconditionally* on windows of D4RL medium-expert
demonstrations, with no knowledge of any roof. At inference time a single
speed-dependent ceiling is imposed at every timestep of the window,

    h(x)[t] = z_t + phi * vz_t - h_r <= 0.

The two problems differ only in their :class:`LocomotionSpec`, so both are
built by the same factory. The constraint and the index layout live in
``problems/locomotion_spec.py``; the data lives in
``cfm/datasets/d4rl_locomotion.py``.

The paper's thresholds bind on the real data without retuning: about 35% of
Walker2D windows and 29% of Hopper windows exceed the roof, so the constraint
has real work to do.
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
# Windows drawn behind the samples, and the (larger) reference set the
# report measures against -- a range built from only a few dozen windows
# is too tight to say anything about staying on the data manifold.
PLOT_WINDOWS = 64
REFERENCE_WINDOWS = 512
# LDF drives the active rows exactly onto the boundary, so counting any
# positive residual as a breach reports rounding noise as a failure.
FEASIBLE_TOL = 1e-6

# Remembered from the last make_constraint call so plotting can draw the same
# roof the samples were generated against.
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
    """The flow model over whole trajectory windows.

    A temporal U-Net over the horizon, with the transition entries as
    channels. A flattened MLP of the same parameter count fits the marginals
    just as well but generates visibly jagged windows, because nothing stops
    it from moving one timestep independently of its neighbours; see the
    module docstring of ``cfm/models/temporal_unet.py``.
    """
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
    """Print how far the generated windows push through the roof.

    Args:
        spec: Which environment.
        x: Generated windows, shape ``(num_samples, horizon, dim)``.
        height_limit: Roof used.
        phi: Velocity weight used.
        reference: Optional training windows, same trailing shape. Used to
            report how often the data itself breaks the roof, so a constraint
            that never binds is obvious, and how far the samples have strayed
            off the data manifold.
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
    """Draw one line per window against the timestep index."""
    steps = np.arange(traces.shape[1])
    for i, trace in enumerate(traces):
        ax.plot(steps, trace, color=color, alpha=alpha, lw=1.0,
                label=label if i == 0 else None)


def plot(problem, samples, constraint=None, spec=None, title="Constrained",
         **_):
    """Show torso-height traces, the residual, and the (z, vz) phase plane.

    The roof drawn on the height panel is not a hard cap on ``z``: the
    constraint bounds ``z + phi * vz``, so a window descending fast enough may
    sit above it and still be feasible. The residual panel is the one that
    shows whether the constraint actually holds.
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
    # The constraint boundary z + phi vz = h_r, drawn over the plotted range.
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
    """One registry entry per environment.

    Nothing here touches the disk: the registry is rebuilt on every CLI
    invocation, so the dataset stays behind a lambda and the constraint is
    built from the spec alone.
    """
    return Problem(
        name=spec.name,
        label=f"{spec.label} (medium-expert)",
        make_dataset=lambda: make_dataset(spec),
        make_model=lambda: make_model(spec),
        train=TrainConfig(
            num_epochs=200,
            batch_size=256,
            # Cosine decay is what makes the windows smooth rather than
            # merely correct: at a constant rate the last iterate is still
            # bouncing around the minimum, and on the near-constant torso
            # height that jitter is larger than the signal. Decaying to
            # 5% of the peak cuts the height trace's step-to-step
            # roughness from 6.5x the training data's to 5.2x.
            learning_rate=2e-4,
            schedule="cosine",
            # Worth about one percent of the marginal spread, consistently
            # across seeds, and costs no measurable training time.
            ema_decay=0.999,
            print_frequency=10,
        ),
        make_constraint=partial(build_constraint, spec),
        plot=partial(plot, spec=spec),
        method_gains={
            # The residual is affine in x and its rows have disjoint support,
            # so one Gauss-Newton step is an exact projection -- unlike the
            # obstacle scene, which needs five for its nonlinear residual.
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
