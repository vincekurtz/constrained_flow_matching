"""Star/unit-circle: ODE stiffness along adaptive LDF and time-varying penalty.

Stiffness is the spectral radius of the per-sample Jacobian df/dy, evaluated
at every accepted step of an adaptive Dopri5 solve. LDF's state includes the
multiplier, so its Jacobian is over (x, lambda).
Run with ``python -m experiments.stiffness --regenerate``.
"""

import argparse
import json
from pathlib import Path

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import problems
from cfm.core import checkpoint
from cfm.core.solve import initial_noise, wrap_constraint
from cfm.methods.ldf import _equality_flow
from cfm.methods.time_varying_penalty import penalty_flow

CACHE = Path("plots/data/stiffness.json")
FIGURE = Path("plots/figures/stiffness")
METHODS = {"ldf": "LDF (ours)", "time_varying_penalty": "Time-varying penalty"}


def run_study(num_samples=16, penalty_weight=5.0, t1=0.99, tol=1e-5,
              max_steps=100_000):
    """Spectral radius per sample at each adaptive step, plus step sizes."""
    problem = problems.get("star")
    model, normalizer = checkpoint.load(problem.checkpoint_path)
    constraint = problem.make_constraint()
    residual = wrap_constraint(
        constraint.fn, normalizer, model.data_shape, penalty_weight,
    )
    x0 = initial_noise(jax.random.key(0), num_samples, model.data_shape)
    n = x0[0].size
    results = {"penalty_weight": penalty_weight, "t1": t1, "rtol": tol,
               "atol": tol, "num_samples": num_samples}
    for name in METHODS:
        if name == "ldf":
            flow = _equality_flow(model, residual, 1., 2.)
            y0 = (x0, jnp.zeros((num_samples, 1)))
            # Single sample as a flat vector z = (x, lambda).
            f = lambda t, z: jnp.concatenate(jax.tree.leaves(flow(
                t, (z[None, :n].reshape((1,) + x0.shape[1:]), z[None, n:]),
                None)), axis=None)
            pack = lambda y: jnp.concatenate(
                [y[0].reshape(y[0].shape[:2] + (-1,)), y[1]], axis=-1)
        else:
            flow = penalty_flow(model, residual)
            y0 = x0
            f = lambda t, z: flow(
                t, z.reshape((1,) + x0.shape[1:]), None).ravel()
            pack = lambda y: y.reshape(y.shape[:2] + (-1,))
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(flow), diffrax.Dopri5(), t0=0., t1=t1, dt0=1e-3,
            y0=y0, saveat=diffrax.SaveAt(steps=True),
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
            max_steps=max_steps, throw=False,
        )
        k = int(sol.stats["num_accepted_steps"])
        ts, zs = sol.ts[:k], pack(sol.ys)[:k]
        jac = jax.jit(jax.vmap(jax.vmap(jax.jacfwd(f, argnums=1),
                                        (None, 0)), (0, 0)))
        jacs = np.asarray(jac(ts, zs))  # (steps, samples, d, d)
        rho = np.abs(np.linalg.eigvals(jacs)).max(-1)
        results[name] = {
            "success": bool(sol.result == diffrax.RESULTS.successful),
            "num_steps": int(sol.stats["num_steps"]),
            "t": np.asarray(ts).tolist(),
            "step": np.diff(np.concatenate([[0.], np.asarray(ts)])).tolist(),
            "rho": rho.tolist(),
        }
        print(f"{name}: steps={results[name]['num_steps']}, "
              f"success={results[name]['success']}, "
              f"max rho={np.nanmax(rho):.3g}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(results) + "\n")
    return results


def plot_study(regenerate=False):
    results = (run_study() if regenerate or not CACHE.exists()
               else json.loads(CACHE.read_text()))
    with plt.rc_context({"font.size": 14, "font.family": "serif"}):
        fig, (ax_rho, ax_h) = plt.subplots(1, 2, figsize=(11, 4.4))
        for (name, label), color in zip(METHODS.items(), ("C0", "C2")):
            r = results[name]
            s = 1 - np.array(r["t"])
            rho = np.array(r["rho"])
            ax_rho.plot(s, np.median(rho, 1), color=color, label=label)
            ax_rho.fill_between(s, rho.min(1), rho.max(1), color=color,
                                alpha=0.2, lw=0)
            ax_h.plot(s, r["step"], color=color,
                      label=f"{label} ({r['num_steps']} steps)")
        for ax in (ax_rho, ax_h):
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.invert_xaxis()
            ax.set_xlabel(r"Time to go $1 - t$")
            ax.grid(which="major", linestyle=":", alpha=0.4)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=11)
        ax_rho.set_ylabel(r"Stiffness $\rho(\partial f / \partial y)$")
        ax_h.set_ylabel("Adaptive step size $h$")
        fig.suptitle(f"Adaptive Dopri5, rtol = atol = {results['rtol']:g}, "
                     f"$c = {results['penalty_weight']:g}$", fontsize=11)
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
