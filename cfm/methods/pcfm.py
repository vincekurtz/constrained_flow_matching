"""Physics-Constrained Flow Matching (PCFM) baseline.

Utkarsh et al., "Physics-Constrained Flow Matching: Sampling Generative Models
with Hard Constraints", https://arxiv.org/abs/2506.04171, Algorithm 1.

Each step: shoot to t = 1, project onto ``h(x) = 0``, interpolate back along
the OT path, then optionally apply a relaxed penalty correction.
"""

import math
from typing import Callable, Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from cfm.core.constraints import EQUALITY, Constraint
from cfm.core.projection import gauss_newton_step
from cfm.core.solve import (
    Samples,
    initial_noise,
    resolve_rng,
    wrap_constraint,
)
from cfm.models.normalizer import Normalizer


def generate_pcfm(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    num_steps: int = 100,
    seed: int = 0,
    rng: jax.Array = None,
    correction_weight: float = 0.1,
    num_correction_iters: int = 10,
    correction_lr: float = 0.1,
    num_projection_iters: int = 8,
    num_final_projection_iters: int = 20,
    eps_reg: float = 1e-10,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples satisfying ``constraint_fn(x) = 0`` via PCFM.

    Args:
        constraint_fn: ``h(x)`` on a single unnormalized sample.
        num_steps: number of outer steps.
        correction_weight: penalty weight in the relaxed correction; 0
            disables it.
        num_correction_iters: gradient steps for the relaxed correction.
        correction_lr: step size for the relaxed correction.
        num_projection_iters: Gauss-Newton iterations per step. One step
            overshoots badly on nonlinear constraints.
        num_final_projection_iters: Gauss-Newton iterations after the loop.
        eps_reg: regularizer on ``J J^T``.
    """
    rng = resolve_rng(rng, seed)
    data_shape = model.data_shape
    flat_dim = math.prod(data_shape)
    dtau = 1.0 / num_steps
    _h = wrap_constraint(constraint_fn, normalizer, data_shape)

    def _v(u_flat: jax.Array, tau: jax.Array) -> jax.Array:
        u = u_flat.reshape(data_shape)
        return model(u[None], jnp.atleast_1d(tau))[0].ravel()

    def _project_step(u_flat: jax.Array) -> jax.Array:
        return gauss_newton_step(_h, u_flat, eps_reg=eps_reg)

    def _project(u_flat: jax.Array) -> jax.Array:
        for _ in range(num_projection_iters):
            u_flat = _project_step(u_flat)
        return u_flat

    def _relaxed_correction(
        u_hat: jax.Array, tau_next: jax.Array
    ) -> jax.Array:
        if correction_weight == 0.0:
            return u_hat
        gamma = 1.0 - tau_next

        def _loss(u_flat: jax.Array) -> jax.Array:
            proxy = u_flat + gamma * _v(u_flat, tau_next)
            h_val = _h(proxy)
            return (jnp.sum((u_flat - u_hat) ** 2)
                    + correction_weight * jnp.sum(h_val ** 2))

        grad_fn = jax.grad(_loss)
        u = u_hat
        for _ in range(num_correction_iters):
            u = u - correction_lr * grad_fn(u)
        return u

    def _per_sample(u0_flat: jax.Array):
        def _scan_step(u_flat, k):
            tau = k * dtau
            tau_next = (k + 1) * dtau
            # Shoot to t = 1, project, then interpolate back to tau_next.
            u1 = u_flat + (1.0 - tau) * _v(u_flat, tau)
            u_proj = _project(u1)
            u_hat = (1.0 - tau_next) * u0_flat + tau_next * u_proj
            u_next = _relaxed_correction(u_hat, tau_next)
            return u_next, u_next

        ks = jnp.arange(num_steps, dtype=jnp.float32)
        u_final, traj = jax.lax.scan(_scan_step, u0_flat, ks)

        u_final_proj = u_final
        for _ in range(num_final_projection_iters):
            u_final_proj = _project_step(u_final_proj)

        traj_full = jnp.concatenate(
            [u0_flat[None], traj[:-1], u_final_proj[None]], axis=0
        )
        return u_final_proj, traj_full

    x_init = initial_noise(rng, num_samples, data_shape)
    x_init_flat = x_init.reshape((num_samples, flat_dim))

    x_final_flat, xs_flat = jax.vmap(_per_sample)(x_init_flat)

    x_final = x_final_flat.reshape((num_samples,) + data_shape)
    xs = xs_flat.transpose((1, 0, 2)).reshape(
        (num_steps + 1, num_samples) + data_shape
    )

    return normalizer.unnormalize(x_final), normalizer.unnormalize(xs)


def generate(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint: Constraint,
    **kwargs,
) -> Samples:
    """Registry entry point (equality constraints only)."""
    if constraint.kind != EQUALITY:
        raise ValueError(
            f"PCFM supports equality constraints only, got {constraint.kind!r}"
        )
    x, xs = generate_pcfm(model, normalizer, constraint.fn, **kwargs)
    return Samples(x, xs)
