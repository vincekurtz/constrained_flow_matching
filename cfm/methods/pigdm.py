"""PiGDM (pseudo-inverse guidance) baseline.

Pokle et al. 2023, https://arxiv.org/abs/2310.04432. Specialized to the
noise-free constraint ``g(x) = 0``, with the Jacobian of ``g`` from autodiff.
"""

from typing import Callable, Tuple

import diffrax
from flax import nnx
import jax
import jax.numpy as jnp

from cfm.core.constraints import EQUALITY, Constraint
from cfm.core.solve import (
    Samples,
    initial_noise,
    resolve_rng,
    save_times,
    wrap_constraint,
)
from cfm.models.normalizer import Normalizer


def generate_pigdm(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    rng: jax.Array = None,
    guidance_scale: float = 1.0,
    eps_reg: float = 1e-4,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples satisfying ``g(x) = 0`` with PiGDM guidance.

    With ``mu = x + (1 - t) v`` and ``A = dg/dmu``, the drift is (Eq. 15)

        v + guidance_scale * ((1 - t) / t)
          * (dmu/dx)^T A^T (r_t^2 A A^T + eps_reg I)^{-1} (-g(mu)),

    where ``r_t^2 = (1 - t)^2 / (t^2 + (1 - t)^2)``.

    Args:
        constraint_fn: ``g(x)`` on a single unnormalized sample.
        guidance_scale: weight on the correction; 1 matches the paper.
        eps_reg: regularizer standing in for ``sigma_y^2``.
    """
    rng = resolve_rng(rng, seed)
    data_shape = model.data_shape
    _g = wrap_constraint(constraint_fn, normalizer, data_shape)

    def _ode_fn(t, y, args):
        del args
        x = y
        x_flat = x.reshape((x.shape[0], -1))

        # Eq. 16.
        r_t_sq = (1.0 - t) ** 2 / (t ** 2 + (1.0 - t) ** 2)
        # Floored to keep the first step finite.
        vf_scale = (1.0 - t) / jnp.maximum(t, 0.1)

        def _mu(x_t_flat: jax.Array):
            """Clean-sample estimate, with the velocity as aux."""
            x_t = x_t_flat.reshape(data_shape)
            v_flat = model(x_t[None], jnp.array([t]))[0].ravel()
            return x_t_flat + (1.0 - t) * v_flat, v_flat

        def _single(x_t_flat: jax.Array):
            mu, vjp_mu, v_flat = jax.vjp(_mu, x_t_flat, has_aux=True)
            g_val = _g(mu)
            A = jax.jacobian(_g)(mu)
            m = g_val.shape[0]
            AAT = A @ A.T
            z = jnp.linalg.solve(r_t_sq * AAT + eps_reg * jnp.eye(m), g_val)
            grad_at_mu = -A.T @ z
            (grad_at_xt,) = vjp_mu(grad_at_mu)
            return v_flat + guidance_scale * vf_scale * grad_at_xt

        x_dot_flat = jax.vmap(_single)(x_flat)
        return x_dot_flat.reshape(x.shape)

    x_init = initial_noise(rng, num_samples, data_shape)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        diffrax.Midpoint(),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=x_init,
        saveat=diffrax.SaveAt(ts=save_times(dt, 1.0), t1=True),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    print(solution.stats["num_steps"], "steps taken")
    xs = solution.ys
    x = xs[-1]

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)


def generate(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint: Constraint,
    **kwargs,
) -> Samples:
    """Registry entry point (equality constraints only)."""
    if constraint.kind != EQUALITY:
        raise ValueError(
            f"PiGDM supports equality constraints only, "
            f"got {constraint.kind!r}"
        )
    x, xs = generate_pigdm(model, normalizer, constraint.fn, **kwargs)
    return Samples(x, xs)
