"""DPS (diffusion posterior sampling) baseline.

Chung et al. 2023, https://arxiv.org/abs/2209.14687. Adapted to the
deterministic flow ODE and the noise-free constraint ``g(x) = 0``.

DPS (Algorithm 1) follows each sampler step with

    x <- x - zeta_i grad_x ||y - A(x0_hat(x))||^2,
    zeta_i = zeta' / ||y - A(x0_hat(x))||        (footnote 5),

where the gradient is backpropagated through the denoiser. With residual
``r = g(mu)`` at the clean-sample estimate ``mu = x + (1 - t) v``, the
per-step displacement is ``2 zeta' (dmu/dx)^T (dg/dmu)^T r / ||r||``.
Dividing by ``dt`` gives the ODE drift used here, so ``step_size`` keeps the
paper's per-step meaning with fixed-step solvers.
"""

import diffrax
import jax
import jax.numpy as jnp

from cfm.core.constraints import EQUALITY
from cfm.core.solve import (
    Samples, initial_noise, resolve_rng, save_times, wrap_constraint,
)


def dps_flow(model, residual, data_shape, *, step_size, dt):
    """Build the DPS-guided vector field on normalized samples."""
    scale = 2.0 * step_size / dt

    def ode_fn(t, x, args):
        del args

        def mu(x_flat):
            """Clean-sample estimate, with the velocity as aux."""
            v = model(x_flat.reshape(data_shape)[None], jnp.array([t]))[0]
            v_flat = v.ravel()
            return x_flat + (1.0 - t) * v_flat, v_flat

        def single(x_flat):
            mu_val, pullback_mu, v_flat = jax.vjp(mu, x_flat, has_aux=True)
            r, pullback_g = jax.vjp(residual, mu_val)
            # grad of ||r||; zero when the constraint already holds.
            norm = jnp.linalg.norm(r)
            unit = jnp.where(norm > 0, r / jnp.where(norm > 0, norm, 1.), 0.)
            (grad_mu,) = pullback_g(unit)
            (grad_x,) = pullback_mu(grad_mu)
            return v_flat - scale * grad_x

        x_flat = x.reshape((x.shape[0], -1))
        return jax.vmap(single)(x_flat).reshape(x.shape)

    return ode_fn


def generate(
    model, normalizer, constraint, num_samples=1000, dt=0.01, seed=0,
    rng=None, step_size=0.1, solver=None, max_steps=4096,
    save_trajectory=True,
):
    """Generate samples with DPS guidance on ``g(x) = 0``.

    Args:
        step_size: the paper's zeta'; it recommends 0.1 to 1. Large values
            diverge when the residual gradient grows with ``x``.
        dt: fixed step; also sets the per-step guidance scale.
        solver: ODE integrator; defaults to Midpoint. Adaptive step-size
            control is not offered because the guidance depends on ``dt``.
    """
    if constraint.kind != EQUALITY:
        raise ValueError(
            f"DPS supports equality constraints only, "
            f"got {constraint.kind!r}"
        )
    if not 0 < dt <= 1:
        raise ValueError("dt must lie in (0, 1]")
    if step_size < 0:
        raise ValueError("step_size must be nonnegative")
    data_shape = model.data_shape
    residual = wrap_constraint(constraint.fn, normalizer, data_shape)
    x0 = initial_noise(resolve_rng(rng, seed), num_samples, data_shape)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(dps_flow(
            model, residual, data_shape, step_size=step_size, dt=dt,
        )),
        diffrax.Midpoint() if solver is None else solver,
        t0=0.0, t1=1.0, dt0=dt, y0=x0,
        saveat=diffrax.SaveAt(
            ts=save_times(dt, 1.0) if save_trajectory else None, t1=True,
        ),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=max_steps,
    )
    xs = normalizer.unnormalize(sol.ys)
    return Samples(xs[-1], xs, sol.stats["num_steps"])
