"""Plots for the constrained flow-matching paper figures.

Each ``plot_*`` function takes a ``regenerate`` flag. When ``regenerate=False``
and a cached raw-data file exists in ``plots/data/``, the function loads it
and only redraws the figure. When ``regenerate=True`` (or the cache is
missing), the function re-runs the underlying generation/benchmark, writes
the raw data, and then draws.

Usage:
    python plots.py --plot all
    python plots.py --plot constrained_star --regenerate
"""

import argparse
import json
import pickle
from functools import partial
from pathlib import Path
import diffrax

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import problems
from cfm import methods
from cfm.core import checkpoint
from cfm.methods import ldf, pcfm, pigdm
from cfm.methods.ldf import generate_unconstrained


DATA_DIR = Path("plots/data")
FIG_DIR = Path("plots/figures")

# Which methods appear in the comparison figures, in plotting order. Labels
# come from the registry so a rename lands here too.
METHODS = ("ldf", "pigdm", "pcfm")
METHOD_COLORS = {
    "ldf": "C0", "pigdm": "C1", "pcfm": "C2", "penalty": "C3", "cbf": "C4",
}
METHOD_NAMES = {name: methods.get(name).label for name in METHODS}


def _cached(container, method):
    """Read a method's entry from cached raw data.

    Data written before the ours -> ldf rename is keyed on the old name;
    fall back to it so existing caches still render. Regenerating writes the
    new key, after which the fallback is dead.
    """
    if method in container:
        return container[method]
    raise KeyError(
        f"no cached data for method {method!r} (have: "
        f"{', '.join(sorted(container))}); re-run with --regenerate"
    )

# Set uniform font size and serif font style
plt.rcParams.update(
    {
        "font.size": 14,
        "font.family": "serif",
    }
)


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_model(example: str):
    """Load a trained model by problem name."""
    return checkpoint.load(problems.get(example).checkpoint_path)


# Constraints come from the problem registry, so the figures, the benchmark
# and the examples all impose exactly the same thing. These used to be three
# separate definitions.
def _constraint(example):
    return problems.get(example).make_constraint()


CIRCLE = None  # built lazily; see _circle()
RIGHT_HALF = None


def _circle():
    """The unit-norm equality constraint used by the star figures."""
    global CIRCLE
    if CIRCLE is None:
        CIRCLE = problems.get("star").make_constraint()
    return CIRCLE


def _right_half():
    """The right-half-plane inequality used by the inequality figure."""
    global RIGHT_HALF
    if RIGHT_HALF is None:
        RIGHT_HALF = problems.get("unit_circle").make_constraint(
            constraint="right_half"
        )
    return RIGHT_HALF


# ============================================================================
# Constraint violation vs penalty weight
# ============================================================================


def plot_violation_vs_penalty(
    regenerate: bool = False,
    num_samples: int = 200,
    dt: float = 0.01,
):
    """Mean constraint violation vs penalty weight for two rescale exponents."""
    _ensure_dirs()
    data_file = DATA_DIR / "violation_vs_penalty.json"

    if regenerate or not data_file.exists():
        print("[violation_vs_penalty] regenerating raw data ...")
        model, normalizer = _load_model("star")
        penalties = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        results = {"penalties": penalties, "dt": dt, "num_samples": num_samples}
        for exp_val, key in ((1.0, "exp1"), (2.0, "exp2")):
            violations = []
            for pw in penalties:
                try:
                    x, _, _ = ldf.generate(
                        model,
                        normalizer,
                        _circle(),
                        num_samples=num_samples,
                        dt=dt,
                        penalty_weight=pw,
                        rescale_factor=1.0,
                        rescale_exponent=exp_val,
                        solver=diffrax.Dopri5(),
                        stepsize_controller=diffrax.PIDController(
                            rtol=1e-5,
                            atol=1e-5,
                            dtmin=1e-5,
                        ),
                    )
                    violation = float(jnp.mean(jnp.abs(_circle().fn(x))))
                    print(
                        f"  exp={exp_val}, penalty={pw}, violation={violation}"
                    )
                except Exception:
                    print(f"  exp={exp_val}, penalty={pw}, Failed!")
                    violation = float("nan")
                violations.append(violation)
            results[key] = violations
        with open(data_file, "w") as f:
            json.dump(results, f, indent=2)

    with open(data_file) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        results["penalties"],
        results["exp1"],
        "o-",
        label="p = 1",
    )
    ax.plot(
        results["penalties"],
        results["exp2"],
        "s-",
        label="p = 2",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Penalty Weight (c)")
    ax.set_ylabel("Mean Constraint Violation (|g(x)|)")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "violation_vs_penalty.png"
    fig.savefig(out, dpi=150)
    print(f"[violation_vs_penalty] wrote {out}")
    plt.close(fig)


# ============================================================================
# MNIST constraint violation vs penalty weight (solver configs)
# ============================================================================


def plot_mnist_violation_vs_penalty(
    regenerate: bool = False,
    num_samples: int = 50,
    dt: float = 0.01,
):
    """MNIST inpainting: max constraint violation vs penalty weight.

    Illustrates the importance of error-controlled integration at high penalty
    values.
    """
    _ensure_dirs()
    data_file = DATA_DIR / "mnist_violation_vs_penalty.json"

    penalties = [1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 30.0, 50.0, 100.0]
    configs = [
        (
            "midpoint",
            "Midpoint (dt=0.01)",
            diffrax.Midpoint(),
            diffrax.ConstantStepSize(),
        ),
        (
            "heun",
            "Heun-Euler (adaptive)",
            diffrax.Heun(),
            diffrax.PIDController(rtol=1e-3, atol=1e-3, dtmin=1e-5),
        ),
    ]

    if regenerate or not data_file.exists():
        print("[mnist_violation_vs_penalty] regenerating raw data ...")
        model, normalizer = _load_model("mnist")
        inpaint = _constraint("mnist")

        def _violation(x):
            return float(jnp.mean(inpaint.violations(x)))

        results = {
            "penalties": penalties,
            "dt": dt,
            "num_samples": num_samples,
        }
        for key, _, solver, controller in configs:
            violations = []
            for pw in penalties:
                x, _, _ = ldf.generate(
                    model,
                    normalizer,
                    inpaint,
                    num_samples=num_samples,
                    dt=dt,
                    penalty_weight=pw,
                    rescale_factor=1.0,
                    rescale_exponent=2.0,
                    solver=solver,
                    stepsize_controller=controller,
                )
                violations.append(_violation(x))
                print(f"  {key}, penalty={pw}, violation={violations[-1]}")
            results[key] = violations
        with open(data_file, "w") as f:
            json.dump(results, f, indent=2)

    with open(data_file) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    markers = {"midpoint": "o-", "heun": "s-"}
    for key, label, _, _ in configs:
        ax.plot(
            results["penalties"],
            results[key],
            markers[key],
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Penalty Weight (c)")
    ax.set_ylabel("Constraint Violation ($\\|g(x)\\|_{\\infty}$)")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "mnist_violation_vs_penalty.png"
    fig.savefig(out, dpi=150)
    print(f"[mnist_violation_vs_penalty] wrote {out}")
    plt.close(fig)


# ============================================================================
# Constrained star + representative trajectories
# ============================================================================


def plot_constrained_star(
    regenerate: bool = False,
    num_samples: int = 500,
    num_paths: int = 20,
):
    """For each method, scatter of constrained samples + a few flow paths."""
    _ensure_dirs()
    data_file = DATA_DIR / "constrained_star.pkl"

    if regenerate or not data_file.exists():
        print("[constrained_star] regenerating raw data ...")
        model, normalizer = _load_model("star")
        data = {}
        x_unc, xs_unc, _ = generate_unconstrained(
            model, normalizer, num_samples=num_samples, dt=0.01
        )
        data["unconstrained"] = {
            "x": np.asarray(x_unc),
            "xs": np.asarray(xs_unc),
        }
        x, xs, _ = ldf.generate(
            model,
            normalizer,
            _circle(),
            num_samples=num_samples,
            dt=0.01,
            penalty_weight=5.0,
            rescale_factor=1.0,
        )
        data["ldf"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        x, xs, _ = pigdm.generate(
            model,
            normalizer,
            _circle(),
            num_samples=num_samples,
            dt=0.01,
            guidance_scale=1.0,
            eps_reg=1e-4,
        )
        data["pigdm"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        x, xs, _ = pcfm.generate(
            model,
            normalizer,
            _circle(),
            num_samples=num_samples,
            num_steps=100,
        )
        data["pcfm"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    theta = np.linspace(0, 2 * np.pi, 200)
    all_keys = [("unconstrained", "Unconstrained", "gray")] + [
        (m, METHOD_NAMES[m], METHOD_COLORS[m]) for m in METHODS
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    for col, (key, title, color) in enumerate(all_keys):
        d = _cached(data, key)
        x, xs = d["x"], d["xs"]

        ax = axes[0, col]
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4)
        ax.scatter(x[:, 0], x[:, 1], alpha=0.5, s=8, color=color)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.4)

        ax = axes[1, col]
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4)
        n_show = min(num_paths, x.shape[0])
        for i in range(n_show):
            ax.plot(xs[:, i, 0], xs[:, i, 1], lw=1.0, alpha=0.7, color=color)
        ax.scatter(
            xs[0, :n_show, 0],
            xs[0, :n_show, 1],
            s=15,
            color="k",
            alpha=0.5,
            label="start",
        )
        ax.scatter(
            xs[-1, :n_show, 0],
            xs[-1, :n_show, 1],
            s=15,
            color=color,
            label="end",
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.grid(linestyle=":", alpha=0.4)
    axes[0, 0].set_ylabel("Generated Samples")
    axes[1, 0].set_ylabel("Flow Paths")
    fig.tight_layout()
    out = FIG_DIR / "constrained_star.png"
    fig.savefig(out, dpi=150)
    print(f"[constrained_star] wrote {out}")
    plt.close(fig)


# ============================================================================
# Constrained MNIST inpainting
# ============================================================================


def plot_constrained_mnist(
    regenerate: bool = False,
    num_samples: int = 25,
    grid: int = 5,
):
    """Reference image once (top-left); 5x5 grids per method + unconstrained."""
    _ensure_dirs()
    data_file = DATA_DIR / "constrained_mnist.pkl"

    if regenerate or not data_file.exists():
        print("[constrained_mnist] regenerating raw data ...")
        model, normalizer = _load_model("mnist")
        from problems.mnist import _reference_and_mask

        reference, mask = _reference_and_mask()
        data = {"reference": np.asarray(reference), "mask": np.asarray(mask)}
        x, _, _ = generate_unconstrained(
            model, normalizer, num_samples=num_samples, dt=0.01
        )
        data["unconstrained"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        x, _, _ = ldf.generate(
            model,
            normalizer,
            _constraint("mnist"),
            num_samples=num_samples,
            dt=0.01,
            penalty_weight=10.0,
            rescale_factor=1.0,
        )
        data["ldf"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        x, _, _ = pigdm.generate(
            model,
            normalizer,
            _constraint("mnist"),
            num_samples=num_samples,
            dt=0.01,
            guidance_scale=1.0,
            eps_reg=1e-4,
        )
        data["pigdm"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        x, _, _ = pcfm.generate(
            model,
            normalizer,
            _constraint("mnist"),
            num_samples=num_samples,
            num_steps=100,
        )
        data["pcfm"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    ref = data["reference"]
    mask = data["mask"]
    masked_ref = np.where(mask, ref, 0.5 * ref)

    panels = [
        ("unconstrained", "Unconstrained"),
        ("ldf", METHOD_NAMES["ldf"]),
        ("pigdm", METHOD_NAMES["pigdm"]),
        ("pcfm", METHOD_NAMES["pcfm"]),
    ]

    # Layout: narrow reference column on the left + 2x2 grid of panels.
    fig = plt.figure(figsize=(12, 9))
    left_fig, right_fig = fig.subfigures(
        1, 2, width_ratios=[1, 2 * grid], wspace=0.06
    )

    # Reference image (left column, centred vertically).
    ax_ref = left_fig.subplots(1, 1)
    ax_ref.imshow(masked_ref.squeeze(-1), cmap="gray", vmin=0, vmax=1)
    ax_ref.set_title("Reference")
    ax_ref.axis("off")

    # 2x2 grid of sample panels with consistent margins.
    m = 0.0  # equal margin fraction on all four sides of the image grid
    title_h = 0.10  # fraction of panel height reserved for the title above
    panel_figs = right_fig.subfigures(2, 2, wspace=0.08, hspace=0.08)
    for i, (key, title) in enumerate(panels):
        sf = panel_figs[i // 2, i % 2]
        sf.set_facecolor("#f0f0f0")
        sf.text(
            0.5, 0.98, title, ha="center", va="top", transform=sf.transSubfigure
        )
        axes = np.asarray(sf.subplots(grid, grid))
        sf.subplots_adjust(
            left=m,
            right=1 - m,
            bottom=m,
            top=1 - m - title_h,
            hspace=0.02,
            wspace=0.02,
        )
        for r in range(grid):
            for c in range(grid):
                axes[r, c].imshow(
                    _cached(data, key)[r * grid + c].squeeze(-1),
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
                axes[r, c].axis("off")

    out = FIG_DIR / "constrained_mnist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[constrained_mnist] wrote {out}")
    plt.close(fig)


# ============================================================================
# Mini MNIST inpainting (one sample per method)
# ============================================================================


def plot_mnist_mini(
    regenerate: bool = False,
    seed: int = 8,
):
    """One row: reference, then a single sample from each method.

    A compact version of :func:`plot_constrained_mnist` for the intro figure.
    All four samples start from the same initial noise (``seed``), so the
    panels differ only in how the constraint is imposed.
    """
    _ensure_dirs()
    data_file = DATA_DIR / "mnist_mini.pkl"

    if regenerate or not data_file.exists():
        print("[mnist_mini] regenerating raw data ...")
        model, normalizer = _load_model("mnist")
        from problems.mnist import _reference_and_mask

        reference, mask = _reference_and_mask()
        data = {"reference": np.asarray(reference), "mask": np.asarray(mask)}

        def _one(x):
            return np.asarray(jnp.clip(x, 0.0, 1.0))[0]

        x, _, _ = generate_unconstrained(
            model, normalizer, num_samples=1, dt=0.01, seed=seed
        )
        data["unconstrained"] = _one(x)
        x, _, _ = ldf.generate(
            model,
            normalizer,
            _constraint("mnist"),
            num_samples=1,
            dt=0.01,
            seed=seed,
            penalty_weight=10.0,
            rescale_factor=1.0,
        )
        data["ldf"] = _one(x)
        x, _, _ = pigdm.generate(
            model,
            normalizer,
            _constraint("mnist"),
            num_samples=1,
            dt=0.01,
            seed=seed,
            guidance_scale=1.0,
            eps_reg=1e-4,
        )
        data["pigdm"] = _one(x)
        x, _, _ = pcfm.generate(
            model,
            normalizer,
            _constraint("mnist"),
            num_samples=1,
            num_steps=100,
            seed=seed,
        )
        data["pcfm"] = _one(x)
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    ref = data["reference"]
    mask = data["mask"]
    # Dim the region being inpainted, as in the full MNIST figure.
    masked_ref = np.where(mask, ref, 0.5 * ref)

    # Short titles: the registry labels are too wide for a one-row figure.
    panels = [
        (masked_ref, "Reference"),
        (_cached(data, "unconstrained"), "Unconstrained"),
        (_cached(data, "ldf"), "LDF"),
        (_cached(data, "pigdm"), "PiGDM"),
        (_cached(data, "pcfm"), "PCFM"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.3))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img.squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.tight_layout()
    out = FIG_DIR / "mnist_mini.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[mnist_mini] wrote {out}")
    plt.close(fig)


# ============================================================================
# Inequality-constrained star
# ============================================================================


def plot_inequality_star(
    regenerate: bool = False,
    num_samples: int = 1000,
):
    """Scatter of star samples constrained to x[0] > 0."""
    _ensure_dirs()
    data_file = DATA_DIR / "inequality_star.pkl"

    if regenerate or not data_file.exists():
        print("[inequality_star] regenerating raw data ...")
        model, normalizer = _load_model("star")
        x_unconstrained, _, _ = generate_unconstrained(
            model, normalizer, num_samples=num_samples, dt=0.01
        )
        x, _, _ = ldf.generate(
            model,
            normalizer,
            _right_half(),
            num_samples=num_samples,
            dt=0.01,
            penalty_weight=20.0,
            rescale_factor=1.0,
        )
        with open(data_file, "wb") as f:
            pickle.dump(
                {"x": np.asarray(x), "x_unc": np.asarray(x_unconstrained)}, f
            )

    with open(data_file, "rb") as f:
        data = pickle.load(f)
    x = data["x"]
    x_unc = data.get("x_unc")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axvspan(
        -2, 0, color="lightcoral", alpha=0.15, label="Forbidden (x[0] <= 0)"
    )
    if x_unc is not None:
        ax.scatter(
            x_unc[:, 0],
            x_unc[:, 1],
            s=15,
            alpha=0.3,
            color="lightgray",
            label="Unconstrained Samples",
        )
    ax.scatter(
        x[:, 0],
        x[:, 1],
        s=15,
        alpha=0.6,
        color="C3",
        label="Constrained Samples",
    )
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / "inequality_star.png"
    fig.savefig(out, dpi=150)
    print(f"[inequality_star] wrote {out}")
    plt.close(fig)


# ============================================================================
# Violation vs number of steps (dual flow p=2 vs penalty-only)
# ============================================================================


def plot_violation_vs_steps(
    regenerate: bool = False,
    num_samples: int = 200,
):
    """Violation vs actual solver steps, dual flow (p=2) vs penalty-only.

    Each point is one penalty_weight value. Uses Dopri5 with tol=1e-5; points
    within each method are connected by a dashed line.
    """
    _ensure_dirs()
    data_file = DATA_DIR / "violation_vs_steps.json"

    penalties = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    tol = 1e-5

    if regenerate or not data_file.exists():
        print("[violation_vs_steps] regenerating raw data ...")
        model, normalizer = _load_model("star")
        results = {"penalties": penalties}

        for key, rescale_factor in (("exp2", 1.0), ("penalty", 0.0)):
            records = []  # list of (num_steps, violation) per penalty
            for pw in penalties:
                # Stiff settings (large penalty_weight, tight tol) can make
                # the adaptive solver fail outright. Record the point as nan
                # rather than losing the whole sweep; it is dropped at plot
                # time.
                try:
                    x, _, n = ldf.generate(
                        model,
                        normalizer,
                        _circle(),
                        num_samples=num_samples,
                        dt=0.01,
                        penalty_weight=pw,
                        rescale_factor=rescale_factor,
                        rescale_exponent=2.0,
                        solver=diffrax.Dopri5(),
                        stepsize_controller=diffrax.PIDController(
                            rtol=tol,
                            atol=tol,
                            dtmin=1e-5,
                        ),
                    )
                except Exception:
                    print(f"  {key}, pw={pw}, generation failed.")
                    records.append((None, float("nan")))
                    continue
                viol = float(jnp.mean(jnp.abs(_circle().fn(x))))
                records.append((int(n) if n is not None else None, viol))
                print(f"  {key}, pw={pw}, steps={n}, violation={viol:.4f}")
            results[key] = records

        with open(data_file, "w") as f:
            json.dump(results, f, indent=2)

    with open(data_file) as f:
        results = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    configs = [
        ("exp2", "C0", "o", "Dual flow (p=2)"),
        ("penalty", "C2", "^", "Penalty only"),
    ]
    for key, color, marker, label in configs:
        pairs = sorted(
            r
            for r in results[key]
            if r[0] is not None and not np.isnan(r[1])
        )
        steps = [r[0] for r in pairs]
        viols = [r[1] for r in pairs]
        ax.plot(
            steps,
            viols,
            color=color,
            marker=marker,
            linestyle="--",
            alpha=0.8,
            label=label,
        )

    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("Mean Constraint Violation (|g(x)|)")
    ax.set_yscale("log")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "violation_vs_steps.png"
    fig.savefig(out, dpi=150)
    print(f"[violation_vs_steps] wrote {out}")
    plt.close(fig)


# ============================================================================
# PCFM flow paths with 1 vs 8 projection iterations
# ============================================================================


def plot_pcfm_projection_iters(
    regenerate: bool = False,
    num_samples: int = 500,
    num_paths: int = 20,
):
    """PCFM flow paths on the star example, 1 vs 8 Gauss-Newton projections.

    A single per-step projection badly overshoots for the nonlinear unit-circle
    constraint (projecting a near-origin endpoint estimate onto the circle
    blows up the radius), so the paths swing wildly before the final sweep
    snaps them back. Eight iterations converge each step, giving smooth paths.
    """
    _ensure_dirs()
    data_file = DATA_DIR / "pcfm_projection_iters.pkl"

    iter_configs = [
        (1, "Original (N=1)"),
        (8, "Improved (N=8)"),
    ]

    if regenerate or not data_file.exists():
        print("[pcfm_projection_iters] regenerating raw data ...")
        model, normalizer = _load_model("star")
        data = {}
        for n_iters, _ in iter_configs:
            x, xs, _  = pcfm.generate(
                model,
                normalizer,
                _circle(),
                num_samples=num_samples,
                num_steps=100,
                num_projection_iters=n_iters,
            )
            viol = float(jnp.mean(jnp.abs(_circle().fn(x))))
            print(f"  num_projection_iters={n_iters}, violation={viol:.3e}")
            data[n_iters] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    theta = np.linspace(0, 2 * np.pi, 200)
    color = METHOD_COLORS["pcfm"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    for ax, (n_iters, title) in zip(axes, iter_configs):
        d = data[n_iters]
        x, xs = d["x"], d["xs"]
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4)
        n_show = min(num_paths, x.shape[0])
        for i in range(n_show):
            ax.plot(xs[:, i, 0], xs[:, i, 1], lw=1.0, alpha=0.7, color=color)
        ax.scatter(
            xs[0, :n_show, 0],
            xs[0, :n_show, 1],
            s=15,
            color="k",
            alpha=0.5,
            label="start",
        )
        ax.scatter(
            xs[-1, :n_show, 0],
            xs[-1, :n_show, 1],
            s=15,
            color=color,
            label="end",
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(linestyle=":", alpha=0.4)
    axes[0].legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    out = FIG_DIR / "pcfm_projection_iters.png"
    fig.savefig(out, dpi=150)
    print(f"[pcfm_projection_iters] wrote {out}")
    plt.close(fig)


# ============================================================================
# Obstacle avoidance: unconstrained vs constrained paths
# ============================================================================


def plot_obstacle_avoidance(
    regenerate: bool = False,
    num_samples: int = 40,
    scene_seed: int = 0,
    num_obstacles: int = 7,
):
    """Robot paths before and after imposing obstacle avoidance.

    The flow model is trained unconditionally on wiggly start-to-goal paths,
    so the unconstrained samples (left) know nothing about the scene. The
    same model, constrained at inference time (right), routes around the
    obstacles it is shown for the first time.

    Scene 2 is used rather than the CLI's default scene 0: at these gains a
    handful of paths in scene 0 diverge and shoot off the figure, which is a
    solver artifact rather than anything the figure is about.
    """
    _ensure_dirs()
    data_file = DATA_DIR / "obstacle_avoidance.pkl"

    if regenerate or not data_file.exists():
        print("[obstacle_avoidance] regenerating raw data ...")
        from problems import obstacles as obstacle_problem

        model, normalizer = _load_model("obstacles")
        problem = problems.get("obstacles")
        # make_constraint samples the scene and remembers it, so the figure
        # draws exactly the obstacles the samples were generated against.
        constraint = problem.make_constraint(
            scene_seed=scene_seed, num_obstacles=num_obstacles
        )
        centers, radii = obstacle_problem._LAST_SCENE["obstacles"]

        x_unc, _, _ = generate_unconstrained(
            model, normalizer, num_samples=num_samples, dt=0.01
        )
        x, _, _ = ldf.generate(
            model,
            normalizer,
            constraint,
            num_samples=num_samples,
            dt=0.002,
            **problem.gains_for("ldf"),
        )
        obstacle_problem.report_violations(x, centers, radii)
        data = {
            "unconstrained": np.asarray(x_unc),
            "constrained": np.asarray(x),
            "centers": np.asarray(centers),
            "radii": np.asarray(radii),
        }
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    from cfm.plotting import plot_paths
    from problems.obstacle_scene import GOAL, PLOT_SUBSAMPLE, START, path

    obstacles = (data["centers"], data["radii"])
    panels = [
        ("unconstrained", "Unconstrained", "gray", None),
        ("constrained", "Constrained", METHOD_COLORS["ldf"], obstacles),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), sharex=True, sharey=True)
    for ax, (key, title, color, scene) in zip(axes, panels):
        plot_paths(
            np.asarray(path(jnp.asarray(_cached(data, key)), PLOT_SUBSAMPLE)),
            obstacles=scene,
            start=START,
            goal=GOAL,
            ax=ax,
            title=title,
            color=color,
            alpha=0.6,
        )
    fig.tight_layout()
    out = FIG_DIR / "obstacle_avoidance.png"
    fig.savefig(out, dpi=150)
    print(f"[obstacle_avoidance] wrote {out}")
    plt.close(fig)


# ============================================================================
# Obstacle avoidance: CBF vs LDF across scene difficulty
# ============================================================================


def plot_obstacle_comparison(
    regenerate: bool = False,
    num_samples: int = 40,
    dt: float = 0.002,
    rows=((1, 1), (3, 2), (5, 3), (7, 4)),
):
    """Unconstrained / CBF / LDF paths, one row per scene difficulty.

    An extended version of :func:`plot_obstacle_avoidance`. Every row is a
    different scene -- ``rows`` gives ``(num_obstacles, seed)`` pairs, and the
    seed drives both the obstacle layout and the initial noise, so the rows
    are independent samples rather than the same paths four times over.

    Both methods integrate at the same ``dt`` so the comparison is at equal
    step count. The CBF filter uses its registry default ``qp="elastic"``
    rather than the ``qp="exact"`` the obstacles problem pins for the paper
    table: at this step size the exact QP hits mutually infeasible barrier
    conditions and fails outright on every row here, leaving nothing to plot.

    Even elastic, the CBF interior-point solve returns a non-finite
    correction on the very first step for a fair share of samples, and those
    paths are gone from the panel rather than drawn badly. Each panel is
    annotated with how many of its paths collided and how many diverged, so
    the missing ones stay visible in the figure.
    """
    _ensure_dirs()
    data_file = DATA_DIR / "obstacle_comparison.pkl"
    keys = ("unconstrained", "cbf", "ldf")

    if regenerate or not data_file.exists():
        print("[obstacle_comparison] regenerating raw data ...")
        from problems import obstacles as obstacle_problem

        model, normalizer = _load_model("obstacles")
        problem = problems.get("obstacles")
        data = {"rows": []}
        for num_obstacles, seed in rows:
            constraint = problem.make_constraint(
                scene_seed=seed, num_obstacles=num_obstacles
            )
            centers, radii = obstacle_problem._LAST_SCENE["obstacles"]
            row = {
                "num_obstacles": num_obstacles,
                "seed": seed,
                "centers": np.asarray(centers),
                "radii": np.asarray(radii),
            }
            x, _, _ = generate_unconstrained(
                model, normalizer, num_samples=num_samples, dt=dt, seed=seed
            )
            row["unconstrained"] = np.asarray(x)
            for name in ("cbf", "ldf"):
                gains = problem.gains_for(name)
                if name == "cbf":
                    gains["qp"] = "elastic"
                x, _, _ = methods.get(name).run(
                    model,
                    normalizer,
                    constraint,
                    num_samples=num_samples,
                    dt=dt,
                    seed=seed,
                    **gains,
                )
                row[name] = np.asarray(x)
            print(f"  {num_obstacles} obstacles (seed {seed}):")
            for name in keys:
                print(f"    {name}:")
                obstacle_problem.report_violations(
                    jnp.asarray(row[name]), centers, radii
                )
            data["rows"].append(row)
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    from cfm.plotting import plot_paths
    from problems.obstacle_scene import GOAL, PLOT_SUBSAMPLE, START, path

    def _stats(knots, centers, radii):
        """(collisions, divergences, total) for one panel's paths.

        Collisions are checked densely along the spline, as in
        ``problems.obstacles.report_violations``, so they reflect the swept
        path rather than the constraint residual.
        """
        knots = jnp.asarray(knots)
        diverged = int(jnp.sum(jnp.any(~jnp.isfinite(knots), axis=(1, 2))))
        dense = path(knots, 50)
        dists = jnp.linalg.norm(
            dense[:, :, None, :] - centers[None, None, :, :], axis=-1
        )
        worst = jnp.max(radii[None, None, :] - dists, axis=(1, 2))
        return int(jnp.sum(worst > 0.0)), diverged, knots.shape[0]

    titles = {"unconstrained": "Unconstrained", "cbf": "CBF", "ldf": "LDF"}
    colors = dict(METHOD_COLORS, unconstrained="gray")

    fig, axes = plt.subplots(
        len(data["rows"]), len(keys),
        figsize=(4.2 * len(keys), 3.4 * len(data["rows"])),
        sharex=True, sharey=True,
    )
    for r, row in enumerate(data["rows"]):
        # The obstacles go on all three panels, including the unconstrained
        # one: the model never sees them, and the point of that column is
        # watching the unconstrained paths run straight through them.
        obstacles = (row["centers"], row["radii"])
        for c, key in enumerate(keys):
            ax = axes[r, c]
            plot_paths(
                np.asarray(path(jnp.asarray(row[key]), PLOT_SUBSAMPLE)),
                obstacles=obstacles,
                start=START,
                goal=GOAL,
                ax=ax,
                title=titles[key] if r == 0 else "",
                color=colors[key],
                alpha=0.6,
            )
            # plot_paths adds a start/goal legend to every panel; one is
            # enough for the whole figure.
            if (r, c) != (0, 0):
                ax.get_legend().remove()
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            hits, diverged, total = _stats(row[key], *obstacles)
            note = f"{hits}/{total} collide"
            if diverged:
                note += f", {diverged}/{total} diverged"
            ax.text(
                0.03, 0.03, note, transform=ax.transAxes, fontsize=11,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8",
                          alpha=0.85),
            )
        # The paths never leave |y| < 1, so the square limits plot_paths
        # applies would leave four rows of mostly empty figure.
        axes[r, 0].set_ylim(-1.2, 1.2)
        n = row["num_obstacles"]
        axes[r, 0].set_ylabel(f"{n} obstacle{'s' if n != 1 else ''}")
    fig.tight_layout()
    out = FIG_DIR / "obstacle_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"[obstacle_comparison] wrote {out}")
    plt.close(fig)


# ============================================================================
# Locomotion phase portraits
# ============================================================================

# Framing for the robot panel: (width, height) in pixels, the world height it
# covers in metres, and the world height at its centre. The pixel aspect is
# roughly the shape of the panel the figure gives it -- an image axes is
# aspect-locked, so whatever it does not match it pads with white -- and the
# extent leaves room for a trailing leg, which reaches further back than the
# robot is tall.
POSE_PIXELS = (340, 460)
POSE_EXTENT = 1.85
POSE_CAMERA_Z = 0.83

# Markers for the two clouds. Shape as well as colour, so the panel survives
# being printed in greyscale and read by someone who cannot tell C0 from grey.
TRAINING_STYLE = dict(marker="o", c="0.55", alpha=0.45, lw=0)
GENERATED_STYLE = dict(marker="^", c="C0", alpha=0.5, lw=0)


def _locomotion_samples(env, height_limit, phi, num_samples, seed, dt):
    """Training windows and constrained samples for one environment.

    Everything here is what ``cfm.cli generate --problem <env> --method ldf``
    would do: the problem's own constraint with its default roof, its default
    sample count, the registry's LDF gains, and the CLI's default seed and
    step size. A figure generated against a hand-picked roof is a different
    experiment from the one the rest of the repo reports.
    """
    from problems.locomotion import REFERENCE_WINDOWS, make_dataset
    from problems.locomotion_spec import SPECS, resolve

    spec = SPECS[env]
    problem = problems.get(env)
    model, normalizer = _load_model(env)
    constraint = problem.make_constraint(height_limit=height_limit, phi=phi)

    x, _, _ = methods.get("ldf").run(
        model, normalizer, constraint,
        num_samples=num_samples or problem.default_num_samples,
        rng=jax.random.key(seed), dt=dt,
        **problem.gains_for("ldf"),
    )
    limit, weight = resolve(spec, height_limit, phi)
    # The same reference set the problem's own plot builds. It draws the
    # first PLOT_WINDOWS of it and measures against the whole thing; here the
    # rest of it is the pool the drawn pose is chosen from.
    reference = make_dataset(
        spec, max_windows=REFERENCE_WINDOWS
    ).windows().numpy()
    return {
        "env": env,
        "height_limit": limit,
        "phi": weight,
        "reference": np.asarray(reference),
        "constrained": np.asarray(x),
    }


# Which frames of the demonstrations are worth drawing: standing on the floor
# rather than mid-flight, and upright rather than pitched over. Most frames
# fail one of the two, and one that does reads as the robot falling rather
# than as a picture of the system.
POSE_CLEARANCE = 0.01
POSE_MAX_PITCH = 0.15


def _pose_frame(renderer, spec, reference, num_windows=64):
    """Pick a pose to draw, and render it.

    Among the frames worth drawing this takes the one of median torso height,
    so the panel shows an ordinary stance rather than the extreme the eye
    would otherwise be drawn to.
    """
    from problems.locomotion_render import window_to_qpos

    poses = np.concatenate(
        [window_to_qpos(spec, window) for window in reference[:num_windows]]
    )
    # qpos is (x, z, pitch, joints...) for both models.
    upright = np.abs(poses[:, 2]) < POSE_MAX_PITCH
    grounded = np.array([
        abs(renderer.ground_clearance(qpos)) < POSE_CLEARANCE
        for qpos in poses
    ])
    usable = upright & grounded
    candidates = poses[usable] if usable.any() else poses
    heights = candidates[:, 1]
    chosen = candidates[np.argsort(heights)[len(heights) // 2]]
    return renderer.frame(chosen), float(chosen[1])


def _draw_pose_panel(ax, renderer, frame, torso_z, label):
    """The robot, with the two axes of the phase plane marked on it.

    No constraint boundary here. It bounds ``z + phi * v_z``, which is not a
    height a line across this panel could stand for, and the panel's job is
    to say what ``z`` and ``v_z`` are -- the panel beside it is where the
    constraint lives.
    """
    ax.imshow(frame)
    ax.set_xlim(0, renderer.width)
    ax.set_ylim(renderer.height, 0)

    ground_row = renderer.row_of_z(0.0)
    torso_row = renderer.row_of_z(torso_z)
    torso_col = renderer.col_of_x(0.0, 0.0)
    ax.axhline(ground_row, color="0.35", lw=1.4)

    # z is measured to the torso centre, not to the top of the robot, and a
    # figure that does not say so invites the reader to check the wrong
    # thing.
    arrow_col = torso_col - 0.30 * renderer.width
    ax.annotate(
        "", xy=(arrow_col, torso_row), xytext=(arrow_col, ground_row),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.2),
    )
    ax.text(
        arrow_col - 6, (torso_row + ground_row) / 2, "$z$",
        fontsize=13, ha="right", va="center", color="0.25",
    )
    ax.plot([arrow_col, torso_col], [torso_row, torso_row],
            color="0.25", lw=0.8, ls=":")
    ax.plot(torso_col, torso_row, "o", ms=6, mfc="white", mec="0.25",
            mew=1.2, zorder=3)
    tip = torso_row - 0.11 * renderer.height
    ax.annotate(
        "", xy=(torso_col, tip), xytext=(torso_col, torso_row),
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.8),
    )
    ax.text(torso_col + 9, tip, "$v_z$", fontsize=13, ha="left",
            va="center", color="C0")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(label, fontsize=15)


def _draw_phase_panel(ax, spec, data, point_size):
    """The (z, v_z) cloud, training behind and constrained samples on top."""
    from problems.locomotion import PLOT_WINDOWS

    training = data["reference"][:PLOT_WINDOWS]
    generated = data["constrained"]
    limit, weight = data["height_limit"], data["phi"]
    z_ref = training[..., spec.z_index].ravel()
    vz_ref = training[..., spec.vz_index].ravel()
    z_gen = generated[..., spec.z_index].ravel()
    vz_gen = generated[..., spec.vz_index].ravel()

    # The grey goes down slightly larger, so it still shows around the blue
    # where the two clouds overlap.
    ax.scatter(z_ref, vz_ref, s=point_size * 1.6,
               label="training data (unconstrained)", **TRAINING_STYLE)
    ax.scatter(z_gen, vz_gen, s=point_size * 1.35, label="generated, LDF",
               **GENERATED_STYLE)

    ax.set_xlim(
        min(z_ref.min(), z_gen.min()) - 0.03,
        max(z_ref.max(), z_gen.max()) + 0.03,
    )
    ax.set_ylim(
        min(vz_ref.min(), vz_gen.min()) - 0.2,
        max(vz_ref.max(), vz_gen.max()) + 0.2,
    )

    # The boundary z + phi*v_z = h_r, slanted because of the lookahead: a
    # window descending fast enough is feasible above h_r, and one rising
    # fast enough is infeasible below it. Drawn across the axes rather than
    # over the data range, so the shaded side reaches the corners.
    vz_line = np.array(ax.get_ylim())
    z_line = limit - weight * vz_line
    ax.fill_betweenx(vz_line, z_line, ax.get_xlim()[1], color="C3",
                     alpha=0.08, lw=0)
    ax.plot(z_line, vz_line, color="C3", ls="--", lw=1.8,
            label="constraint")

    ax.set_xlabel("torso height $z$ (m)")
    ax.set_ylabel("vertical velocity $v_z$ (m/s)")
    ax.grid(alpha=0.3)
    # Above the axes rather than inside them: the cloud fills the frame, and
    # in the Hopper panel every interior corner the legend could take has
    # part of the hop cycle in it.
    ax.legend(
        loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.1), mode="expand",
        ncols=3, fontsize=11, frameon=False, markerscale=2.5,
        borderaxespad=0.0,
    )


def plot_locomotion_phase(
    env: str,
    regenerate: bool = False,
    num_samples=None,
    height_limit=None,
    phi=None,
    seed: int = 0,
    dt: float = 0.01,
    point_size: float = 14.0,
):
    """A rendering of the robot beside the phase plane it moves in.

    The left panel is one frame of the demonstrations with the torso height
    and its velocity marked on it, so the axes of the right panel are
    something the reader has seen on the robot rather than two names. The
    right panel is every timestep of the training windows against every
    timestep of the constrained samples, with the constraint boundary drawn.

    The arguments mirror the ``generate`` command's, and their defaults are
    its defaults: leaving ``height_limit`` and ``phi`` as None takes the
    problem's own roof, exactly as the CLI does.
    """
    _ensure_dirs()
    data_file = DATA_DIR / f"locomotion_{env}.pkl"

    if regenerate or not data_file.exists():
        print(f"[locomotion_{env}] regenerating raw data ...")
        data = _locomotion_samples(
            env, height_limit, phi, num_samples, seed, dt
        )
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    from problems.locomotion_render import LocomotionRenderer
    from problems.locomotion_spec import SPECS

    spec = SPECS[env]
    renderer = LocomotionRenderer(
        spec, width=POSE_PIXELS[0], height=POSE_PIXELS[1],
        extent=POSE_EXTENT, camera_z=POSE_CAMERA_Z,
    )
    try:
        frame, torso_z = _pose_frame(renderer, spec, data["reference"])
        fig, axes = plt.subplots(
            1, 2, figsize=(11.0, 4.6),
            gridspec_kw={"width_ratios": (1.0, 2.1)},
        )
        _draw_pose_panel(axes[0], renderer, frame, torso_z, spec.label)
        _draw_phase_panel(axes[1], spec, data, point_size)
        fig.tight_layout()
        out = FIG_DIR / f"locomotion_{env}.png"
        fig.savefig(out, dpi=200)
        print(f"[locomotion_{env}] wrote {out}")
        plt.close(fig)
    finally:
        renderer.close()


# ============================================================================
# CLI
# ============================================================================

PLOTS = {
    "violation_vs_penalty": plot_violation_vs_penalty,
    "mnist_violation_vs_penalty": plot_mnist_violation_vs_penalty,
    "constrained_star": plot_constrained_star,
    "constrained_mnist": plot_constrained_mnist,
    "mnist_mini": plot_mnist_mini,
    "inequality_star": plot_inequality_star,
    "violation_vs_steps": plot_violation_vs_steps,
    "pcfm_projection_iters": plot_pcfm_projection_iters,
    "obstacle_avoidance": plot_obstacle_avoidance,
    "obstacle_comparison": plot_obstacle_comparison,
    "locomotion_walker2d": partial(plot_locomotion_phase, "walker2d"),
    "locomotion_hopper": partial(plot_locomotion_phase, "hopper"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot",
        choices=list(PLOTS) + ["all"],
        default="all",
        help="Which plot to make (or all).",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run the underlying generation/benchmark, overwriting cache.",
    )
    args = parser.parse_args()

    names = list(PLOTS) if args.plot == "all" else [args.plot]
    for name in names:
        print(f"\n=== {name} ===")
        PLOTS[name](regenerate=args.regenerate)


if __name__ == "__main__":
    main()
