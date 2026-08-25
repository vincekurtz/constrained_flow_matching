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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import problems
from cfm import methods
from cfm.core import checkpoint
from cfm.methods import ldf
from cfm.methods.ldf import generate_unconstrained
from cfm.methods.pcfm import generate_pcfm
from cfm.methods.pigdm import generate_pigdm


DATA_DIR = Path("plots/data")
FIG_DIR = Path("plots/figures")

# Which methods appear in the comparison figures, in plotting order. Labels
# come from the registry so a rename lands here too.
METHODS = ("ldf", "pigdm", "pcfm")
METHOD_COLORS = {"ldf": "C0", "pigdm": "C1", "pcfm": "C2", "penalty": "C3"}
METHOD_NAMES = {name: methods.get(name).label for name in METHODS}

# Cached raw data written before the ours -> ldf rename still uses the old
# key. Read-side alias only; drop it once the figures have been regenerated.
LEGACY_METHOD_KEYS = {"ours": "ldf"}


def _canonical_method(key):
    return LEGACY_METHOD_KEYS.get(key, key)


def _cached(container, method):
    """Read a method's entry from cached raw data.

    Data written before the ours -> ldf rename is keyed on the old name;
    fall back to it so existing caches still render. Regenerating writes the
    new key, after which the fallback is dead.
    """
    if method in container:
        return container[method]
    for old, new in LEGACY_METHOD_KEYS.items():
        if new == method and old in container:
            return container[old]
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


def _get_constraint(example):
    """(constraint, scalar violation fn) for one problem."""
    c = _constraint(example)
    return c, lambda x: float(c.violation(x))


# ---- per-sample timing ---------------------------------------------------


def _make_single_sample_fn(method, example, *, dt, num_steps, **overrides):
    """JIT a ``rng -> single sample`` function for one method and problem.

    This was a near-copy of ``benchmark.py:build_generator``; both now go
    through the method registry, so the figures and the table cannot drift
    apart in how they invoke a method.
    """
    model, normalizer = _load_model(example)
    problem = problems.get(example)
    constraint = _constraint(example)
    spec = methods.get(_canonical_method(method))

    cfg = problem.gains_for(spec.name)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if spec.name in methods.USES_DT:
        cfg["dt"] = dt
    else:
        cfg["num_steps"] = num_steps

    def _gen(rng):
        return spec.run(
            model, normalizer, constraint, num_samples=1, rng=rng, **cfg
        ).x[0]

    return jax.jit(_gen)


def _time_method(method, example, num_samples, *, dt, num_steps, seed=0):
    """JIT a single-sample generator and time ``num_samples`` invocations.

    Returns ``(times, violations)``: per-sample wall-clock times (s) and the
    corresponding scalar constraint violations of each generated sample.
    """
    gen = _make_single_sample_fn(method, example, dt=dt, num_steps=num_steps)
    _, violation_fn = _get_constraint(example)
    rngs = jax.random.split(jax.random.key(seed), num_samples)
    # Warm-up (compile + first call).
    jax.block_until_ready(gen(rngs[0]))
    times, violations = [], []
    for rng in rngs:
        t0 = time.perf_counter()
        sample = gen(rng)
        jax.block_until_ready(sample)
        times.append(time.perf_counter() - t0)
        violations.append(violation_fn(sample))
    return times, violations


# ============================================================================
# Plot 1: generation times
# ============================================================================


def plot_generation_times(regenerate: bool = False, num_samples: int = 20):
    """Box plots: per-sample generation times."""
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
                    times, violations = _time_method(
                        method,
                        example,
                        num_samples,
                        dt=dt,
                        num_steps=n_steps,
                    )
                    results[example][method][label] = {
                        "times": times,
                        "violations": violations,
                    }
        with open(data_file, "w") as f:
            json.dump(results, f, indent=2)

    with open(data_file) as f:
        results = json.load(f)

    print("[generation_times] summary per sample:")
    header = f"    {'method':>12} / {'config':<6}: {'time (sec)':>10} | "
    header += f"{'viol mean':>10} {'viol min':>10} {'viol max':>10}"
    for example in ("star", "mnist"):
        print(f"  {example}:")
        print(header)
        for method in METHODS:
            for label in ("fine", "coarse"):
                entry = _cached(results[example], method)[label]
                times = entry["times"]
                violations = entry["violations"]
                mean_sec = sum(times) / len(times)
                v_mean = np.nanmean(violations)
                v_min = min(violations)
                v_max = max(violations)
                print(
                    f"    {method:>12} / {label:<6}: {mean_sec:10.3f} | "
                    f"{v_mean:10.3e} {v_min:10.3e} {v_max:10.3e}"
                )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, example in zip(axes, ("star", "mnist")):
        data, labels, positions, colors = [], [], [], []
        x = 0.0
        for method in METHODS:
            for label in ("fine", "coarse"):
                data.append(
                    [t * 1000
                     for t in _cached(results[example], method)[label]["times"]]
                )
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
                        rtol=1e-5, atol=1e-5, dtmin=1e-5,
                    ),
                )
                violations.append(
                    float(jnp.mean(jnp.abs(_circle().fn(x))))
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
# Plot 2b: MNIST constraint violation vs penalty weight (solver configs)
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
                print(
                    f"  {key}, penalty={pw}, violation={violations[-1]}"
                )
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
# Plot 3: constrained star + representative trajectories
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
            "x": np.asarray(x_unc), "xs": np.asarray(xs_unc)
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
        x, xs = generate_pigdm(
            model,
            normalizer,
            _circle().fn,
            num_samples=num_samples,
            dt=0.01,
            guidance_scale=1.0,
            eps_reg=1e-4,
        )
        data["pigdm"] = {"x": np.asarray(x), "xs": np.asarray(xs)}
        x, xs = generate_pcfm(
            model,
            normalizer,
            _circle().fn,
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
            xs[0, :n_show, 0], xs[0, :n_show, 1],
            s=15, color="k", alpha=0.5, label="start",
        )
        ax.scatter(
            xs[-1, :n_show, 0], xs[-1, :n_show, 1],
            s=15, color=color, label="end",
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
# Plot 5: constrained MNIST inpainting
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

        inpaint = _constraint("mnist")
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
    m = 0.0   # equal margin fraction on all four sides of the image grid
    title_h = 0.10  # fraction of panel height reserved for the title above
    panel_figs = right_fig.subfigures(2, 2, wspace=0.08, hspace=0.08)
    for i, (key, title) in enumerate(panels):
        sf = panel_figs[i // 2, i % 2]
        sf.set_facecolor("#f0f0f0")
        sf.text(0.5, 0.98, title, ha="center", va="top",
                transform=sf.transSubfigure)
        axes = np.asarray(sf.subplots(grid, grid))
        sf.subplots_adjust(
            left=m, right=1 - m, bottom=m, top=1 - m - title_h,
            hspace=0.02, wspace=0.02,
        )
        for r in range(grid):
            for c in range(grid):
                axes[r, c].imshow(
                    _cached(data, key)[r * grid + c].squeeze(-1),
                    cmap="gray", vmin=0, vmax=1,
                )
                axes[r, c].axis("off")

    out = FIG_DIR / "constrained_mnist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
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
                        rtol=tol, atol=tol, dtmin=1e-5,
                    ),
                )
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
        pairs = sorted(r for r in results[key] if r[0] is not None)
        steps = [r[0] for r in pairs]
        viols = [r[1] for r in pairs]
        ax.plot(
            steps, viols, color=color, marker=marker,
            linestyle="--", alpha=0.8, label=label,
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
# Plot 8: PCFM flow paths with 1 vs 8 projection iterations
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
            x, xs = generate_pcfm(
                model,
                normalizer,
                _circle().fn,
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
            xs[0, :n_show, 0], xs[0, :n_show, 1],
            s=15, color="k", alpha=0.5, label="start",
        )
        ax.scatter(
            xs[-1, :n_show, 0], xs[-1, :n_show, 1],
            s=15, color=color, label="end",
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
# CLI
# ============================================================================

PLOTS = {
    "generation_times": plot_generation_times,
    "violation_vs_penalty": plot_violation_vs_penalty,
    "mnist_violation_vs_penalty": plot_mnist_violation_vs_penalty,
    "constrained_star": plot_constrained_star,
    "constrained_mnist": plot_constrained_mnist,
    "inequality_star": plot_inequality_star,
    "violation_vs_steps": plot_violation_vs_steps,
    "pcfm_projection_iters": plot_pcfm_projection_iters,
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
