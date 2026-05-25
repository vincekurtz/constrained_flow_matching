"""Plots for the constrained flow-matching paper figures.

Each ``plot_*`` function takes a ``regenerate`` flag. When ``regenerate=False``
and a cached raw-data file exists in ``plots/data/``, the function loads it
and only redraws the figure. When ``regenerate=True`` (or the cache is
missing), the function re-runs the underlying generation/benchmark, writes
the raw data, and then draws.

Usage:
    python plots.py --plot all
    python plots.py --plot generation_times --regenerate
"""

import argparse
import json
import pickle
import time
from pathlib import Path
import diffrax

import cloudpickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from datasets.mnist import MNISTDataset
from datasets.star import StarDataset
from generation import (
    generate,
    generate_constrained,
    generate_inequality_constrained,
)
from pcfm import generate_pcfm
from pi_gdm import generate_pigdm


DATA_DIR = Path("plots/data")
FIG_DIR = Path("plots/figures")
MODEL_DIR = Path("data")

METHODS = ("ours", "pigdm", "pcfm")
METHOD_COLORS = {"ours": "C0", "pigdm": "C1", "pcfm": "C2"}
METHOD_NAMES = {
    "ours": "Dual Flow (ours)",
    "pigdm": "Pseudoinverse Guidance",
    "pcfm": "Physics-Constrained",
}

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
    with open(MODEL_DIR / f"{example}_model.pkl", "rb") as f:
        d = cloudpickle.load(f)
    return d["model"], d["normalizer"]


# ---- constraints ---------------------------------------------------------


def _unit_circle_constraint(x):
    return jnp.sum(x**2, axis=-1) - 1.0


def _right_half_constraint(x):
    """Inequality h(x) = -x[0] <= 0."""
    return -x[0]


def _build_mnist_inpaint():
    """Top-half inpainting constraint matching examples/mnist.py."""
    ds = MNISTDataset(train=False, digit=5)
    reference = jnp.array(ds[0])
    mask = jnp.zeros((28, 28, 1), dtype=bool).at[:14, :, :].set(True)
    observed_indices = jnp.where(mask.ravel())[0]
    y = reference.ravel()[observed_indices]
    A = jnp.eye(28 * 28 * 1)[observed_indices]

    def inpaint(x):
        return A @ x.ravel() - y

    return inpaint, reference, mask


def _get_constraint(example):
    if example == "star":
        return _unit_circle_constraint, lambda x: float(
            jnp.abs(_unit_circle_constraint(x))
        )
    if example == "mnist":
        inpaint, _, _ = _build_mnist_inpaint()
        return inpaint, lambda x: float(jnp.max(jnp.abs(inpaint(x))))
    raise ValueError(example)


# ---- per-sample timing ---------------------------------------------------


def _make_single_sample_fn(
    method,
    example,
    *,
    dt,
    num_steps,
    penalty_weight=5.0,
    rescale_factor=1.0,
    rescale_exponent=2.0,
    guidance_scale=1.0,
    eps_reg=1e-4,
    pcfm_penalty_weight=0.0,
    num_final_projection_iters=20,
):
    model, normalizer = _load_model(example)
    constraint_fn, _ = _get_constraint(example)

    if method == "ours":

        def _gen(rng):
            x, _, _ = generate_constrained(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                dt=dt,
                rng=rng,
                penalty_weight=penalty_weight,
                rescale_factor=rescale_factor,
                rescale_exponent=rescale_exponent,
            )
            return x[0]
    elif method == "pigdm":

        def _gen(rng):
            x, _ = generate_pigdm(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                dt=dt,
                rng=rng,
                guidance_scale=guidance_scale,
                eps_reg=eps_reg,
            )
            return x[0]
    elif method == "pcfm":

        def _gen(rng):
            x, _ = generate_pcfm(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                num_steps=num_steps,
                rng=rng,
                penalty_weight=pcfm_penalty_weight,
                num_final_projection_iters=num_final_projection_iters,
            )
            return x[0]
    else:
        raise ValueError(method)

    return jax.jit(_gen)


def _time_method(method, example, num_samples, *, dt, num_steps, seed=0):
    """JIT a single-sample generator and time ``num_samples`` invocations."""
    gen = _make_single_sample_fn(method, example, dt=dt, num_steps=num_steps)
    rngs = jax.random.split(jax.random.key(seed), num_samples)
    # Warm-up (compile + first call).
    jax.block_until_ready(gen(rngs[0]))
    times = []
    for rng in rngs:
        t0 = time.perf_counter()
        sample = gen(rng)
        jax.block_until_ready(sample)
        times.append(time.perf_counter() - t0)
    return times


# ============================================================================
# Plot 1: generation times
# ============================================================================


def plot_generation_times(regenerate: bool = False, num_samples: int = 20):
    """Box plots: per-sample generation time for 2 examples x 3 methods x 2 step sizes."""
    _ensure_dirs()
    data_file = DATA_DIR / "generation_times.json"

    if regenerate or not data_file.exists():
        print("[generation_times] regenerating raw data ...")
        configs = (("fine", 0.01, 100), ("coarse", 0.1, 10))
        results = {}
        for example in ("star", "mnist"):
            results[example] = {}
            for method in METHODS:
                results[example][method] = {}
                for label, dt, n_steps in configs:
                    print(f"  {example} / {method} / {label}")
                    times = _time_method(
                        method,
                        example,
                        num_samples,
                        dt=dt,
                        num_steps=n_steps,
                    )
                    results[example][method][label] = times
        with open(data_file, "w") as f:
            json.dump(results, f, indent=2)

    with open(data_file) as f:
        results = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, example in zip(axes, ("star", "mnist")):
        data, labels, positions, colors = [], [], [], []
        x = 0.0
        for method in METHODS:
            for label in ("fine", "coarse"):
                data.append([t * 1000 for t in results[example][method][label]])
                labels.append(f"{method}\n{label}")
                positions.append(x)
                colors.append(METHOD_COLORS[method])
                x += 1.0
            x += 0.5  # gap between methods
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            medianprops={"color": "k"},
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("Time per sample (ms)")
        ax.set_yscale("log")
        ax.set_title(example)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle(f"Per-sample generation time (N={num_samples} samples each)")
    fig.tight_layout()
    out = FIG_DIR / "generation_times.png"
    fig.savefig(out, dpi=150)
    print(f"[generation_times] wrote {out}")
    plt.close(fig)


# ============================================================================
# Plot 2: constraint violation vs penalty weight
# ============================================================================


def plot_violation_vs_penalty(
    regenerate: bool = False,
    num_samples: int = 200,
    dt: float = 0.01,
):
    """Mean constraint violation vs penalty weight for ours, two rescale exponents."""
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
                x, _, _ = generate_constrained(
                    model,
                    normalizer,
                    _unit_circle_constraint,
                    num_samples=num_samples,
                    dt=dt,
                    penalty_weight=pw,
                    rescale_factor=1.0,
                    rescale_exponent=exp_val,
                    solver=diffrax.Dopri5(),
                    stepsize_controller=diffrax.PIDController(
                        rtol=1e-5, atol=1e-5, dtmin=1e-5,
                    ),
                )
                violations.append(
                    float(jnp.mean(jnp.abs(_unit_circle_constraint(x))))
                )
                print(
                    f"  exp={exp_val}, penalty={pw}, violation={violations[-1]}"
                )
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
# Plot 3: unconstrained star
# ============================================================================


def plot_unconstrained_star(
    regenerate: bool = False,
    num_samples: int = 2000,
):
    """Scatter of unconstrained star samples, with training data behind."""
    _ensure_dirs()
    data_file = DATA_DIR / "unconstrained_star.pkl"

    if regenerate or not data_file.exists():
        print("[unconstrained_star] regenerating raw data ...")
        model, normalizer = _load_model("star")
        x, _ = generate(model, normalizer, num_samples=num_samples, dt=0.01)
        with open(data_file, "wb") as f:
            pickle.dump({"x": np.asarray(x)}, f)

    with open(data_file, "rb") as f:
        x = pickle.load(f)["x"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[:, 0], x[:, 1], alpha=0.6, color="C0", label="generated")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    out = FIG_DIR / "unconstrained_star.png"
    fig.savefig(out, dpi=150)
    print(f"[unconstrained_star] wrote {out}")
    plt.close(fig)


# ============================================================================
# Plot 4: constrained star + representative trajectories
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
        x, xs, _ = generate_constrained(
            model,
            normalizer,
            _unit_circle_constraint,
            num_samples=num_samples,
            dt=0.01,
            penalty_weight=5.0,
            rescale_factor=1.0,
        )
        data["ours"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        x, xs = generate_pigdm(
            model,
            normalizer,
            _unit_circle_constraint,
            num_samples=num_samples,
            dt=0.01,
            guidance_scale=1.0,
            eps_reg=1e-4,
        )
        data["pigdm"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        x, xs = generate_pcfm(
            model,
            normalizer,
            _unit_circle_constraint,
            num_samples=num_samples,
            num_steps=100,
        )
        data["pcfm"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    theta = np.linspace(0, 2 * np.pi, 200)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    for col, method in enumerate(METHODS):
        d = data[method]
        x, xs = d["x"], d["xs"]

        ax = axes[0, col]
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4)
        ax.scatter(
            x[:, 0], x[:, 1], alpha=0.5, s=8, color=METHOD_COLORS[method]
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect("equal")
        ax.set_title(METHOD_NAMES[method])
        ax.grid(linestyle=":", alpha=0.4)

        ax = axes[1, col]
        ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.4)
        n_show = min(num_paths, x.shape[0])
        for i in range(n_show):
            ax.plot(
                xs[:, i, 0],
                xs[:, i, 1],
                lw=1.0,
                alpha=0.7,
                color=METHOD_COLORS[method],
            )
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
            color=METHOD_COLORS[method],
            label="end",
        )
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # Leave grid lines, but remove ticks and labels for the trajectory subplots.
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
# Plot 5: constrained MNIST inpainting
# ============================================================================


def plot_constrained_mnist(
    regenerate: bool = False,
    num_samples: int = 5,
):
    """One row per method, columns are inpainted samples; first column is reference."""
    _ensure_dirs()
    data_file = DATA_DIR / "constrained_mnist.pkl"

    if regenerate or not data_file.exists():
        print("[constrained_mnist] regenerating raw data ...")
        model, normalizer = _load_model("mnist")
        inpaint, reference, mask = _build_mnist_inpaint()
        data = {"reference": np.asarray(reference), "mask": np.asarray(mask)}
        x, _, _ = generate_constrained(
            model,
            normalizer,
            inpaint,
            num_samples=num_samples,
            dt=0.01,
            penalty_weight=10.0,
            rescale_factor=1.0,
        )
        data["ours"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        x, _ = generate_pigdm(
            model,
            normalizer,
            inpaint,
            num_samples=num_samples,
            dt=0.01,
            guidance_scale=1.0,
            eps_reg=1e-4,
        )
        data["pigdm"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        x, _ = generate_pcfm(
            model,
            normalizer,
            inpaint,
            num_samples=num_samples,
            num_steps=100,
        )
        data["pcfm"] = np.asarray(jnp.clip(x, 0.0, 1.0))
        with open(data_file, "wb") as f:
            pickle.dump(data, f)

    with open(data_file, "rb") as f:
        data = pickle.load(f)

    n = data["ours"].shape[0]
    fig, axes = plt.subplots(
        len(METHODS), n + 1, figsize=(n + 1, len(METHODS) + 0.4)
    )
    ref = data["reference"]
    mask = data["mask"]
    masked_ref = np.where(mask, ref, 0.5 * ref)
    for row, method in enumerate(METHODS):
        ax = axes[row, 0]
        ax.imshow(masked_ref.squeeze(-1), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(method, fontsize=10)
        if row == 0:
            ax.set_title("ref", fontsize=9)
        for col in range(n):
            ax = axes[row, col + 1]
            ax.imshow(
                data[method][col].squeeze(-1), cmap="gray", vmin=0, vmax=1
            )
            ax.axis("off")
    fig.suptitle("Inpainted MNIST (top half fixed)", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "constrained_mnist.png"
    fig.savefig(out, dpi=150)
    print(f"[constrained_mnist] wrote {out}")
    plt.close(fig)


# ============================================================================
# Plot 6: inequality-constrained star
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
        x, _ = generate_inequality_constrained(
            model,
            normalizer,
            _right_half_constraint,
            num_samples=num_samples,
            dt=0.01,
            penalty_weight=10.0,
            rescale_factor=1.0,
        )
        with open(data_file, "wb") as f:
            pickle.dump({"x": np.asarray(x)}, f)

    with open(data_file, "rb") as f:
        x = pickle.load(f)["x"]

    dataset = StarDataset(num_samples=1024)
    train = np.asarray(dataset.data)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axvspan(
        -2, 0, color="lightcoral", alpha=0.15, label="forbidden (x[0] <= 0)"
    )
    ax.scatter(
        train[:, 0],
        train[:, 1],
        s=4,
        alpha=0.25,
        color="lightgray",
        label="training data",
    )
    ax.scatter(x[:, 0], x[:, 1], s=4, alpha=0.6, color="C3", label="generated")
    ax.axvline(0, color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Star: inequality-constrained samples (x[0] > 0)")
    fig.tight_layout()
    out = FIG_DIR / "inequality_star.png"
    fig.savefig(out, dpi=150)
    print(f"[inequality_star] wrote {out}")
    plt.close(fig)


# ============================================================================
# Plot 7: violation vs number of steps (dual flow p=2 vs penalty-only)
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
                x, _, n = generate_constrained(
                    model,
                    normalizer,
                    _unit_circle_constraint,
                    num_samples=num_samples,
                    dt=0.01,
                    penalty_weight=pw,
                    rescale_factor=rescale_factor,
                    rescale_exponent=2.0,
                    solver=diffrax.Dopri5(),
                    stepsize_controller=diffrax.PIDController(
                        rtol=tol, atol=tol, dtmin=1e-5,
                    ),
                )
                viol = float(jnp.mean(jnp.abs(_unit_circle_constraint(x))))
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
        pairs = sorted(r for r in results[key] if r[0] is not None)
        steps = [r[0] for r in pairs]
        viols = [r[1] for r in pairs]
        ax.plot(
            viols, steps, color=color, marker=marker,
            linestyle="--", alpha=0.8, label=label,
        )

    ax.invert_xaxis()
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.set_ylabel("Denoising Steps")
    ax.set_xlabel("Mean Constraint Violation (|g(x)|)")
    ax.grid(which="both", linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    out = FIG_DIR / "violation_vs_steps.png"
    fig.savefig(out, dpi=150)
    print(f"[violation_vs_steps] wrote {out}")
    plt.close(fig)


# ============================================================================
# CLI
# ============================================================================

PLOTS = {
    "generation_times": plot_generation_times,
    "violation_vs_penalty": plot_violation_vs_penalty,
    "unconstrained_star": plot_unconstrained_star,
    "constrained_star": plot_constrained_star,
    "constrained_mnist": plot_constrained_mnist,
    "inequality_star": plot_inequality_star,
    "violation_vs_steps": plot_violation_vs_steps,
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
