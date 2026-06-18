"""Implements the PCFM (Physics-Constrained Flow Matching) baseline.

Reference: Utkarsh et al., "Physics-Constrained Flow Matching: Sampling
Generative Models with Hard Constraints" (https://arxiv.org/pdf/2506.04171),
Algorithm 1.

The method enforces a (possibly nonlinear) constraint ``h(x) = 0`` at the final
sample by interleaving four operations at every step:

    1. Forward shoot from the current time to ``t = 1`` to predict the clean
       sample ``u_1`` (we use a single Euler / Tweedie step).
    2. Gauss-Newton projection of ``u_1`` onto the linearised constraint
       manifold: ``u_proj = u_1 - J^T (J J^T)^{-1} h(u_1)``.
    3. Reverse OT solve: a constant-velocity reverse integration along the OT
       displacement, which for the linear OT interpolant collapses to
       ``u_hat = (1 - t') u_0 + t' u_proj``.
    4. (Optional) relaxed penalty correction that gradient-descends
       ``||u - u_hat||^2 + lambda ||h(u + gamma v(u, t'))||^2`` for a few
       iterations.

A final Gauss-Newton loop drives the residual to numerical zero.
"""

import math
from typing import Callable, Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from architectures.normalizer import Normalizer


def generate_pcfm(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    num_steps: int = 100,
    rng: jax.Array = None,
    penalty_weight: float = 0.1,
    num_correction_iters: int = 10,
    correction_lr: float = 0.1,
    num_projection_iters: int = 8,
    num_final_projection_iters: int = 20,
    eps_reg: float = 1e-10,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples satisfying ``constraint_fn(x) = 0`` via PCFM.

    Implements Algorithm 1 of Utkarsh et al., 2025.

    Args:
        model: Trained flow model ``xdot = v(x, t)``. Must have ``data_shape``.
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable ``h(x)`` on a single *unnormalized*
            sample; the constraint is ``h(x) = 0``. May return a scalar or a
            1-D array.
        num_samples: Number of samples to generate.
        num_steps: Number of outer integration steps ``N`` (paper default
            100-200 for PDE benchmarks).
        rng: PRNG key for the initial noise. Defaults to ``jax.random.key(0)``
            when not provided.
        penalty_weight: Weight ``lambda`` on the relaxed-correction penalty.
            The paper notes ``lambda = 0`` is appropriate for linear
            constraints (the projection alone suffices).
        num_correction_iters: Gradient-descent iterations for the relaxed
            correction (ignored when ``penalty_weight == 0``).
        correction_lr: Step size for the relaxed-correction gradient descent.
        num_projection_iters: Number of Gauss-Newton iterations used for the
            per-step projection of the endpoint estimate onto the constraint
            manifold. A single step is unreliable for strongly nonlinear
            constraints (e.g. projecting a near-origin point onto a circle
            overshoots to a huge radius), so we iterate to convergence.
        num_final_projection_iters: Number of Gauss-Newton iterations applied
            after the main loop to drive ``||h(x)||`` to numerical zero.
        eps_reg: Tikhonov regulariser on ``J J^T`` for the projection solve.

    Returns:
        x: Final samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps + 1, num_samples, *data_shape)``,
            spanning ``t = 0`` (initial noise) to ``t = 1`` (post-projection
            final sample).
    """
    if rng is None:
        rng = jax.random.key(0)
    data_shape = model.data_shape
    flat_dim = math.prod(data_shape)
    dtau = 1.0 / num_steps

    def _h(u_flat: jax.Array) -> jax.Array:
        """Constraint on a flat *normalized* sample."""
        x = u_flat.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(constraint_fn(x))

    def _v(u_flat: jax.Array, tau: jax.Array) -> jax.Array:
        """Flow velocity at a single flat normalized sample."""
        u = u_flat.reshape(data_shape)
        return model(u[None], jnp.atleast_1d(tau))[0].ravel()

    def _project_step(u_flat: jax.Array) -> jax.Array:
        """One Gauss-Newton step: u <- u - J^T (J J^T)^{-1} h(u)."""
        r = _h(u_flat)
        J = jax.jacobian(_h)(u_flat)  # (m, n)
        m = r.shape[0]
        z = jnp.linalg.solve(J @ J.T + eps_reg * jnp.eye(m), r)
        return u_flat - J.T @ z

    def _project(u_flat: jax.Array) -> jax.Array:
        """Iterate Gauss-Newton onto the manifold ``h(u) = 0``.

        A single linearised step badly overshoots for nonlinear constraints
        (projecting a point near the origin onto a circle blows up to radius
        ``~1/(2|u|)``), so we iterate to convergence.
        """
        for _ in range(num_projection_iters):
            u_flat = _project_step(u_flat)
        return u_flat

    def _relaxed_correction(
        u_hat: jax.Array, tau_next: jax.Array
    ) -> jax.Array:
        """Soft refinement on the relaxed objective (no-op when lambda = 0)."""
        if penalty_weight == 0.0:
            return u_hat
        gamma = 1.0 - tau_next

        def _loss(u_flat: jax.Array) -> jax.Array:
            proxy = u_flat + gamma * _v(u_flat, tau_next)
            h_val = _h(proxy)
            return (jnp.sum((u_flat - u_hat) ** 2)
                    + penalty_weight * jnp.sum(h_val ** 2))

        grad_fn = jax.grad(_loss)
        u = u_hat
        for _ in range(num_correction_iters):
            u = u - correction_lr * grad_fn(u)
        return u

    def _per_sample(u0_flat: jax.Array):
        def _scan_step(u_flat, k):
            tau = k * dtau
            tau_next = (k + 1) * dtau
            # 1. Forward shoot to t = 1 with a single Euler / Tweedie step.
            u1 = u_flat + (1.0 - tau) * _v(u_flat, tau)
            # 2. Gauss-Newton projection onto the linearised manifold.
            u_proj = _project(u1)
            # 3. Reverse OT solve: linear interpolation u0 -> u_proj at tau'.
            u_hat = (1.0 - tau_next) * u0_flat + tau_next * u_proj
            # 4. Relaxed penalty correction.
            u_next = _relaxed_correction(u_hat, tau_next)
            return u_next, u_next

        ks = jnp.arange(num_steps, dtype=jnp.float32)
        u_final, traj = jax.lax.scan(_scan_step, u0_flat, ks)

        # Final Gauss-Newton sweep to drive ||h(u)|| to numerical zero.
        u_final_proj = u_final
        for _ in range(num_final_projection_iters):
            u_final_proj = _project_step(u_final_proj)

        # Prepend the initial noise and overwrite the last step with the
        # post-projection sample so the trajectory ends on the manifold.
        traj_full = jnp.concatenate(
            [u0_flat[None], traj[:-1], u_final_proj[None]], axis=0
        )
        return u_final_proj, traj_full

    # Initial noise (normalised space) sampled once per generation call.
    x_init = jax.random.normal(rng, (num_samples,) + data_shape)
    x_init_flat = x_init.reshape((num_samples, flat_dim))

    x_final_flat, xs_flat = jax.vmap(_per_sample)(x_init_flat)

    # Reshape back to (num_samples, *data_shape) and put time first on xs.
    x_final = x_final_flat.reshape((num_samples,) + data_shape)
    xs = xs_flat.transpose((1, 0, 2)).reshape(
        (num_steps + 1, num_samples) + data_shape
    )

    return normalizer.unnormalize(x_final), normalizer.unnormalize(xs)
