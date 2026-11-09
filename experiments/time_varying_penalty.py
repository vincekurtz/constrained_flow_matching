"""Star/unit-circle: constraint violation vs penalty weight, no projection.

All methods use fixed-step Midpoint with the same ``dt`` and initial noise.
Run with ``python -m experiments.time_varying_penalty --regenerate``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import problems
from cfm import methods
from cfm.core import checkpoint

CACHE = Path("plots/data/time_varying_penalty.json")
FIGURE = Path("plots/figures/time_varying_penalty_violation_vs_penalty")
METHODS = ("ldf", "penalty", "time_varying_penalty")
PENALTIES = [0.1, 0.2, 0.5, 1., 2., 5., 10., 20., 50.]


def run_study(num_samples=64, dt=0.01):
    """Mean |g(x)| for each method and penalty weight (NaN on divergence)."""
    problem = problems.get("star")
    model, normalizer = checkpoint.load(problem.checkpoint_path)
    constraint = problem.make_constraint()
    results = {"dt": dt, "num_samples": num_samples, "penalties": PENALTIES}
    for name in METHODS:
        method = methods.get(name)
        results[name] = []
        for c in PENALTIES:
            x = method.generate(
                model, normalizer, constraint, num_samples=num_samples,
                dt=dt, seed=0, **dict(method.defaults, penalty_weight=c),
            )[0]
            violation = float(np.mean(constraint.violations(x)))
            results[name].append(violation)
            print(f"{name}: c={c:g}, violation={violation:.3g}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(results, indent=2) + "\n")
    return results


def plot_study(regenerate=False):
    results = (run_study() if regenerate or not CACHE.exists()
               else json.loads(CACHE.read_text()))
    with plt.rc_context({"font.size": 14, "font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for name, marker in zip(METHODS, "o^s"):
            ax.plot(results["penalties"], results[name], marker=marker,
                    label=methods.get(name).label, lw=1.6, ms=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Penalty weight $c$")
        ax.set_ylabel(r"Mean constraint violation $|g(x)|$")
        ax.set_title(f"Fixed-step Midpoint, $\\Delta t = {results['dt']:g}$",
                     fontsize=11)
        ax.grid(which="major", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=11)
        fig.tight_layout()
        FIGURE.parent.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(FIGURE.with_suffix(f".{ext}"), dpi=200)
            print(f"Wrote {FIGURE}.{ext}", flush=True)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate", action="store_true")
    plot_study(regenerate=parser.parse_args().regenerate)
