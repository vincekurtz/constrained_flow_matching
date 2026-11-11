"""Obstacles (2): the small-variation condition (A4) near t = 1.

For each LDF sample, S is the set of constraints active at the end of
denoising, G(t) = J_S(x_t) J_S(x_t)^T, and A_* = G(1). The plot shows
K ||G(t) - A_*||_2 with K = ||A_*^{-1} + I/2||_2 + 1, which (A4) needs below 1.

LDF (closed-form slack, exact 1 / (1 - t) gain) is integrated in float64 in
sigma = -log(1 - t), so the time to go never loses precision. G at
1 - t = 1e-17 stands in for G(1). The plot stops at 1 - t = 1e-13: near 1e-14
the state stops changing in float64 and G(t) equals A_* exactly.
Run with ``python -m experiments.small_variation`` (about 20 s).
"""

import jax

jax.config.update("jax_enable_x64", True)  # lambda_min(A_*) reaches ~1e-13

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import problems  # noqa: E402
from cfm.core import checkpoint  # noqa: E402
from cfm.core.solve import (  # noqa: E402
    batched_velocity,
    initial_noise,
    wrap_constraint,
)

FIGURE = "plots/figures/small_variation"
ACTIVE_TOL = 1e-3  # h(x_1) >= -tol counts as active
REF_TIME_TO_GO, PLOT_TIME_TO_GO = 1e-17, 1e-13
D_SIGMA, SAVE_EVERY = 0.002, 5


def log_time_flow(model, c, rescale_factor):
    """LDF with the closed-form slack, in sigma = -log(1 - t)."""

    def ode_fn(sigma, y, args):
        x, lmbda = y
        s = jnp.exp(-sigma)  # time to go; dt = s dsigma
        x_flat, v_flat = batched_velocity(model, 1.0 - s, x)

        def _single(x_i, v_i, lmbda_i):
            h, vjp_fn = jax.vjp(c, x_i)
            g = jnp.maximum(h, -lmbda_i)
            active = (h > -lmbda_i).astype(g.dtype)
            x_dot = v_i - vjp_fn(active * (lmbda_i + g))[0]
            return s * x_dot, rescale_factor * g

        x_dot, lmbda_dot = jax.vmap(_single)(x_flat, v_flat, lmbda)
        return x_dot.reshape(x.shape), lmbda_dot

    return ode_fn


def a4_curves(num_samples=40):
    """Time to go and K ||G(t) - A_*||_2 per sample with active constraints."""
    problem = problems.get("obstacles")
    model, normalizer = checkpoint.load(problem.checkpoint_path)
    constraint = problem.make_constraint(scene_seed=0, num_obstacles=2)
    gains = problem.gains_for("ldf")
    c = wrap_constraint(constraint.fn, normalizer, model.data_shape,
                        gains["penalty_weight"])
    h = wrap_constraint(constraint.fn, normalizer, model.data_shape, 1.0)

    x0 = initial_noise(jax.random.key(0), num_samples, model.data_shape)
    x0 = x0.astype(jnp.float64)
    lam0 = jnp.zeros_like(jax.vmap(lambda xi: c(xi.ravel()))(x0))
    sigma1 = -np.log(REF_TIME_TO_GO)
    num_steps = int(np.ceil(sigma1 / D_SIGMA))
    sigmas = np.linspace(0.0, sigma1, num_steps + 1)[::SAVE_EVERY]
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(log_time_flow(model, c, gains["rescale_factor"])),
        diffrax.Midpoint(), t0=0.0, t1=sigma1, dt0=sigma1 / num_steps,
        y0=(x0, lam0), saveat=diffrax.SaveAt(ts=sigmas),
        max_steps=num_steps + 1,
    )
    xs = sol.ys[0].reshape(sol.ys[0].shape[:2] + (-1,))  # (T, N, n)

    jac = jax.jit(jax.vmap(jax.jacfwd(h)))  # over time, one sample
    h_end = np.asarray(jax.vmap(h)(xs[-1]))
    curves = []
    for i in range(num_samples):
        S = np.flatnonzero(h_end[i] >= -ACTIVE_TOL)
        if len(S):
            J = np.asarray(jac(xs[:, i]))[:, S]
            G = J @ J.transpose(0, 2, 1)
            A = G[-1]
            K = np.linalg.norm(np.linalg.inv(A) + 0.5 * np.eye(len(S)), 2) + 1
            curves.append(K * np.linalg.norm(G - A, ord=2, axis=(1, 2)))
    return np.exp(-sigmas), np.array(curves)


def plot():
    s, curves = a4_curves()
    keep = s >= PLOT_TIME_TO_GO
    s, curves = s[keep], curves[:, keep]
    with plt.rc_context({"font.size": 14, "font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.plot(s, curves.T, color="C0", alpha=0.25, lw=0.8)
        ax.plot(s, np.median(curves, 0), color="C0", lw=2.2, label="Median")
        ax.axhline(1.0, color="k", ls="--", lw=1.2, label="(A4) bound")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, PLOT_TIME_TO_GO)
        ax.set_xticks(10.0 ** -np.arange(0, 13, 3))
        ax.set_xlabel(r"Time to go $1 - t$")
        ax.set_ylabel(r"$K_{A_*}\,\|G(t) - A_*\|_2$")
        ax.grid(which="major", linestyle=":", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=11)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(f"{FIGURE}.{ext}", dpi=200)
            print(f"Wrote {FIGURE}.{ext}")


if __name__ == "__main__":
    plot()
