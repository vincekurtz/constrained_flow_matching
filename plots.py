"""Paper figures.

Raw data is cached in plots/data/; pass --regenerate to rebuild it.

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

METHODS = ("ldf", "pigdm", "pcfm")
METHOD_COLORS = {
    "ldf": "C0", "pigdm": "C1", "pcfm": "C2", "penalty": "C3", "cbf": "C4",
}
METHOD_NAMES = {name: methods.get(name).label for name in METHODS}


def _cached(container, method):
    """Read a method's entry from cached raw data."""
    if method in container:
        return container[method]
    raise KeyError(
        f"no cached data for method {method!r} (have: "
        f"{', '.join(sorted(container))}); re-run with --regenerate"
    )


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


def _constraint(example):
    return problems.get(example).make_constraint()


CIRCLE = None
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
    """MNIST inpainting: constraint violation vs penalty weight, per solver."""
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
    """Constrained samples and a few flow paths for each method."""
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
    """Reference image and a grid of samples per method."""
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

    fig = plt.figure(figsize=(12, 9))
    left_fig, right_fig = fig.subfigures(
        1, 2, width_ratios=[1, 2 * grid], wspace=0.06
    )

    ax_ref = left_fig.subplots(1, 1)
    ax_ref.imshow(masked_ref.squeeze(-1), cmap="gray", vmin=0, vmax=1)
    ax_ref.set_title("Reference")
    ax_ref.axis("off")

    m = 0.0  # margin around each image grid
    title_h = 0.10  # fraction of panel height reserved for the title
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
    """Reference and one sample per method, all from the same initial noise."""
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
    masked_ref = np.where(mask, ref, 0.5 * ref)

    panels = [
        (masked_ref, "Reference"),
        (_cached(data, "unconstrained"), "Unconstrained"),
        (_cached(data, "ldf"), "LDF (ours)"),
        (_cached(data, "pigdm"), "ΠGDM"),
        (_cached(data, "pcfm"), "PCFM"),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.3))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img.squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=20)
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
    """Violation vs solver steps, dual flow (p=2) vs penalty-only."""
    _ensure_dirs()
    data_file = DATA_DIR / "violation_vs_steps.json"

    penalties = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    tol = 1e-5

    if regenerate or not data_file.exists():
        print("[violation_vs_steps] regenerating raw data ...")
        model, normalizer = _load_model("star")
        results = {"penalties": penalties}

        for key, rescale_factor in (("exp2", 1.0), ("penalty", 0.0)):
            records = []  # (num_steps, violation) per penalty
            for pw in penalties:
                # Stiff settings can make the adaptive solver fail; record nan.
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
    """PCFM flow paths on the star example, 1 vs 8 Gauss-Newton projections."""
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
            x, xs, _ = pcfm.generate(
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
    """Robot paths with and without the obstacle constraint."""
    _ensure_dirs()
    data_file = DATA_DIR / "obstacle_avoidance.pkl"

    if regenerate or not data_file.exists():
        print("[obstacle_avoidance] regenerating raw data ...")
        from problems import obstacles as obstacle_problem

        model, normalizer = _load_model("obstacles")
        problem = problems.get("obstacles")
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

    ``rows`` holds ``(num_obstacles, seed)`` pairs; the seed sets both the
    scene and the initial noise.
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
                # The exact QP is infeasible at this dt on every row.
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
        """(collisions, divergences, total), checked densely along paths."""
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
        # Paths stay within |y| < 1; override plot_paths' square limits.
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

# Robot render framing: pixel size, world height covered (m), camera height.
POSE_PIXELS = (340, 460)
POSE_EXTENT = 1.85
POSE_CAMERA_Z = 0.83

# Distinct markers so the clouds survive greyscale printing.
TRAINING_STYLE = dict(marker="o", c="0.55", alpha=0.45, lw=0)
GENERATED_STYLE = dict(marker="^", c="C0", alpha=0.5, lw=0)


def _locomotion_samples(env, height_limit, phi, num_samples, seed, dt):
    """Training windows and LDF samples for one environment."""
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


# Only draw poses that are grounded and upright.
POSE_CLEARANCE = 0.01
POSE_MAX_PITCH = 0.15


def _pose_frame(renderer, spec, reference, num_windows=64):
    """Render the grounded, upright pose of median torso height."""
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


def _draw_pose_panel(ax, renderer, frame, torso_z, label, fontsize=13,
                     vz_length=0.11):
    """The robot with ``z`` and ``v_z`` marked on it.

    ``vz_length`` is the ``v_z`` arrow length as a fraction of panel height.
    """
    ax.imshow(frame)
    ax.set_xlim(0, renderer.width)
    ax.set_ylim(renderer.height, 0)

    ground_row = renderer.row_of_z(0.0)
    torso_row = renderer.row_of_z(torso_z)
    torso_col = renderer.col_of_x(0.0, 0.0)
    ax.axhline(ground_row, color="0.35", lw=1.4)

    # z is measured to the torso centre.
    arrow_col = torso_col - 0.30 * renderer.width
    ax.annotate(
        "", xy=(arrow_col, torso_row), xytext=(arrow_col, ground_row),
        arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.2),
    )
    ax.text(
        arrow_col - 6, (torso_row + ground_row) / 2, "$z$",
        fontsize=fontsize, ha="right", va="center", color="0.25",
    )
    ax.plot([arrow_col, torso_col], [torso_row, torso_row],
            color="0.25", lw=0.8, ls=":")
    ax.plot(torso_col, torso_row, "o", ms=6, mfc="white", mec="0.25",
            mew=1.2, zorder=3)
    tip = torso_row - vz_length * renderer.height
    ax.annotate(
        "", xy=(torso_col, tip), xytext=(torso_col, torso_row),
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.8),
    )
    ax.text(torso_col + 9, tip, "$v_z$", fontsize=fontsize, ha="left",
            va="center", color="C0")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if label:
        ax.set_title(label, fontsize=15)


def _draw_phase_panel(ax, spec, data, point_size, legend=True):
    """The (z, v_z) cloud, training behind and constrained samples on top."""
    from problems.locomotion import PLOT_WINDOWS

    training = data["reference"][:PLOT_WINDOWS]
    generated = data["constrained"]
    limit, weight = data["height_limit"], data["phi"]
    z_ref = training[..., spec.z_index].ravel()
    vz_ref = training[..., spec.vz_index].ravel()
    z_gen = generated[..., spec.z_index].ravel()
    vz_gen = generated[..., spec.vz_index].ravel()

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

    # Boundary z + phi * v_z = h_r, drawn across the full axes.
    vz_line = np.array(ax.get_ylim())
    z_line = limit - weight * vz_line
    ax.fill_betweenx(vz_line, z_line, ax.get_xlim()[1], color="C3",
                     alpha=0.08, lw=0)
    ax.plot(z_line, vz_line, color="C3", ls="--", lw=1.8,
            label="constraint")

    ax.set_xlabel("torso height $z$ (m)")
    ax.set_ylabel("vertical velocity $v_z$ (m/s)")
    ax.grid(alpha=0.3)
    if not legend:
        return
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
    """Robot render beside the (z, v_z) phase plane of training and LDF samples.

    Arguments and defaults mirror the ``generate`` CLI command.
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
# Overview: the star, obstacles and hopper in one full-width figure
# ============================================================================

# Drawn at print size (ICLR \textwidth, inches). STIX matches Times.
ICLR_TEXT_WIDTH = 5.5
OVERVIEW_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titlepad": 3,
    "axes.labelsize": 9,
    "axes.labelpad": 2,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "legend.fontsize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
}
OVERVIEW_CAPTION_SIZE = 10

UNCONSTRAINED_COLOR = "0.6"
CONSTRAINED_COLOR = METHOD_COLORS["ldf"]
CONSTRAINT_COLOR = "C3"


def _overview_data(name, plot_fn, regenerate):
    """Load a source figure's cached data, building it if needed."""
    data_file = DATA_DIR / f"{name}.pkl"
    if regenerate or not data_file.exists():
        plot_fn(regenerate=True)
    with open(data_file, "rb") as f:
        return pickle.load(f)


def _bare(ax):
    """Remove ticks and grey the spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("0.4")


def _overview_legend(fig, center_y):
    """Shared legend, centred at figure y-coordinate ``center_y``."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Line2D([], [], color=UNCONSTRAINED_COLOR, marker="o", ms=6,
                   lw=1.2, label="Unconstrained"),
            Line2D([], [], color=CONSTRAINED_COLOR, marker="^", ms=6,
                   lw=1.2, label="Constrained (LDF)"),
            Patch(facecolor=CONSTRAINT_COLOR, alpha=0.3,
                  edgecolor=CONSTRAINT_COLOR, label="Constraint"),
        ],
        loc="center", bbox_to_anchor=(0.5, center_y),
        ncols=3, frameon=False, handlelength=1.8, columnspacing=1.5,
        borderaxespad=0.0,
    )


def _overview_star(ax, star, point_size):
    """Star samples kept in the right half plane."""
    lim = 1.35
    ax.axvspan(-lim, 0, color=CONSTRAINT_COLOR, alpha=0.12, lw=0)
    ax.axvline(0, color=CONSTRAINT_COLOR, ls="--", lw=1.0)
    x_unc = star.get("x_unc")
    if x_unc is not None:
        ax.scatter(x_unc[:, 0], x_unc[:, 1], s=point_size * 1.6,
                   color=UNCONSTRAINED_COLOR, alpha=0.5, lw=0)
    ax.scatter(star["x"][:, 0], star["x"][:, 1], s=point_size * 1.35,
               marker="^", color=CONSTRAINED_COLOR, alpha=0.7, lw=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    _bare(ax)


def _overview_obstacles(axes, obstacles):
    """Unconstrained and constrained paths, one panel each."""
    from matplotlib.patches import Circle

    from problems.obstacle_scene import GOAL, PLOT_SUBSAMPLE, START, path

    lim = 1.15
    for ax, (key, title, color) in zip(axes, [
        ("unconstrained", "Unconstrained", UNCONSTRAINED_COLOR),
        ("constrained", "Constrained", CONSTRAINED_COLOR),
    ]):
        for center, radius in zip(obstacles["centers"], obstacles["radii"]):
            ax.add_patch(Circle(tuple(center), float(radius),
                                color=CONSTRAINT_COLOR, alpha=0.3, lw=0,
                                zorder=1))
        paths = np.asarray(
            path(jnp.asarray(_cached(obstacles, key)), PLOT_SUBSAMPLE)
        )
        for p in paths:
            ax.plot(p[:, 0], p[:, 1], color=color, alpha=0.6, lw=0.6,
                    zorder=2)
        ax.plot(*START, "o", color="k", ms=4, zorder=3)
        ax.plot(*GOAL, "*", color="k", ms=7, zorder=3)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.text(0.5, 0.04, title, transform=ax.transAxes, ha="center",
                va="bottom")
        _bare(ax)


def _overview_pose(ax, hopper, crop, vz_length):
    """Hopper render, keeping the left ``crop`` fraction of its width."""
    from problems.locomotion_render import LocomotionRenderer
    from problems.locomotion_spec import SPECS

    spec = SPECS["hopper"]
    renderer = LocomotionRenderer(
        spec, width=POSE_PIXELS[0], height=POSE_PIXELS[1],
        extent=POSE_EXTENT, camera_z=POSE_CAMERA_Z,
    )
    try:
        frame, torso_z = _pose_frame(renderer, spec, hopper["reference"])
        _draw_pose_panel(ax, renderer, frame, torso_z, None, fontsize=9,
                         vz_length=vz_length)
        ax.set_xlim(0, crop * renderer.width)
    finally:
        renderer.close()


def _overview_phase(ax, hopper, point_size):
    """Hopper phase plane with compact labels."""
    from problems.locomotion_spec import SPECS

    _draw_phase_panel(ax, SPECS["hopper"], hopper, point_size, legend=False)
    ax.set_xlabel("torso height $z$ (m)")
    ax.set_ylabel("$v_z$ (m/s)")
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))


def plot_overview(regenerate: bool = False, point_size: float = 2.0):
    """Star and obstacles on top, hopper below, at ICLR text width.

    Drawn from the inequality_star, obstacle_avoidance and locomotion_hopper
    caches.
    """
    _ensure_dirs()
    star = _overview_data("inequality_star", plot_inequality_star, regenerate)
    obstacles = _overview_data(
        "obstacle_avoidance", plot_obstacle_avoidance, regenerate
    )
    hopper = _overview_data(
        "locomotion_hopper", partial(plot_locomotion_phase, "hopper"),
        regenerate,
    )

    # Inches, from the top-left corner.
    W = ICLR_TEXT_WIDTH
    margin, legend_h, caption_h, gap = 0.03, 0.22, 0.22, 0.12
    xlabel_h = 0.33
    left = 0.08
    right = W - margin - 0.02
    square, pair_gap = 1.60, 0.05
    phase_h = 1.25
    pose_crop = 0.8
    pose_w = phase_h * pose_crop * POSE_PIXELS[0] / POSE_PIXELS[1]
    ylabel_w = 0.36

    top_row = margin + legend_h + 0.05  # top of the star and obstacles
    bottom_row = top_row + square + caption_h + gap  # top of the hopper
    caption_top = bottom_row + phase_h + xlabel_h + 0.02
    H = caption_top + caption_h + margin

    def axes(x, y, width, height):
        return fig.add_axes(
            (x / W, 1 - (y + height) / H, width / W, height / H)
        )

    def caption(text, center, y):
        fig.text(center / W, 1 - y / H, text, ha="center", va="top",
                 fontsize=OVERVIEW_CAPTION_SIZE)

    with plt.rc_context(OVERVIEW_RC):
        fig = plt.figure(figsize=(W, H))
        _overview_legend(fig, 1 - (margin + legend_h / 2) / H)

        # (a) Star, flush left.
        _overview_star(axes(left, top_row, square, square), star, point_size)
        caption("(a)", left + square / 2,
                top_row + square + 0.03)

        # (b) Obstacle avoidance, flush right with the phase plane below.
        pair_w = 2 * square + pair_gap
        x = right - pair_w
        _overview_obstacles(
            [axes(x + i * (square + pair_gap), top_row, square, square)
             for i in range(2)],
            obstacles,
        )
        caption("(b)", x + pair_w / 2,
                top_row + square + 0.03)

        # (c) Hopper: the robot, then its phase plane.
        _overview_pose(axes(left, bottom_row, pose_w, phase_h), hopper,
                       crop=pose_crop, vz_length=0.18)
        phase_left = left + pose_w + ylabel_w
        _overview_phase(
            axes(phase_left, bottom_row, right - phase_left, phase_h),
            hopper, point_size,
        )
        caption("(c)", (left + W - margin) / 2, caption_top)

        out = FIG_DIR / "overview.png"
        fig.savefig(out, dpi=300)
        print(f"[overview] wrote {out}")
        plt.close(fig)


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
    "overview": plot_overview,
}

# Drawn from other figures' caches, so --plot all need not regenerate them.
COMPOSITES = {"overview"}


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
        composite = args.plot == "all" and name in COMPOSITES
        PLOTS[name](regenerate=args.regenerate and not composite)


if __name__ == "__main__":
    main()
