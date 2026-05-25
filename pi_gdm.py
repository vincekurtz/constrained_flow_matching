"""Implements the PiGDM (pseudo-inverse guidance) baseline of Pokle et al. 2023.

Reference: https://arxiv.org/pdf/2310.04432

The original method is written for noisy linear inverse problems y = A x + eps.
We specialise to the noise-free equality constraint g(x) = A x - y = 0, but
the implementation works for any differentiable ``constraint_fn`` by evaluating
the residual ``g(x)`` and its Jacobian ``A = dg/dx`` via autodiff.
"""

from typing import Callable, Tuple

import diffrax
from flax import nnx
import jax
import jax.numpy as jnp

from architectures.normalizer import Normalizer


def generate_pigdm(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    rng: jax.Array = None,
    guidance_scale: float = 1.0,
    eps_reg: float = 1e-4,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples from a flow-matching model with PiGDM guidance.

    Implements the PiGDM baseline (Pokle et al., 2023, Algorithm 1) for the
    equality constraint ``g(x) = 0``. At each integration step we compute the
    Tweedie-style clean estimate ``mu = x_t + (1 - t) v(x_t, t)``, evaluate
    the constraint residual ``g(mu)`` and its Jacobian ``A = dg/dmu``, and add
    the pseudo-inverse correction of Eq. 15:

        v_y = v + guidance * ((1 - t) / t)
                * J^T A^T (r_t^2 A A^T + eps_reg I)^{-1} (y - A mu)

    where ``J = d mu / d x_t``, ``r_t^2 = (1 - t)^2 / (t^2 + (1 - t)^2)`` (Eq.
    16), and ``(1 - t) / t`` is the score-to-vector-field scale from line 8 of
    Algorithm 1. Both the chain-rule push back through ``J`` and the
    multiplication by ``A^T`` are evaluated with vector-Jacobian products, so
    no Jacobians are formed explicitly except the small ``m x n`` constraint
    Jacobian needed to build ``A A^T``.

    Args:
        model: Trained flow model xdot = v(x, t). Must have a ``data_shape``
            attribute.
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable function ``g(x)`` (operating on a single
            *unnormalized* sample) such that ``g(x) = 0`` is the desired
            constraint. May return a scalar or a 1-D array. For a linear
            inverse problem this is ``g(x) = A x - y``.
        num_samples: Number of samples to generate.
        dt: Step size hint for the adaptive integrator.
        rng: PRNG key for the initial noise. Defaults to ``jax.random.key(0)``
            when not provided.
        guidance_scale: Extra multiplicative weight on the PiGDM correction;
            ``1.0`` matches Algorithm 1 (``gamma_t = 1`` for OT-ODE).
        eps_reg: Tikhonov regulariser on ``r_t^2 A A^T`` (substitutes for
            ``sigma_y^2`` in our noise-free setting).

    Returns:
        x: Final generated samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``.
    """
    if rng is None:
        rng = jax.random.key(0)
    data_shape = model.data_shape

    def _g(x_flat: jax.Array) -> jax.Array:
        """Constraint residual on a single flattened *normalized* sample."""
        x = x_flat.reshape(data_shape)
        x = normalizer.unnormalize(x)
        return jnp.atleast_1d(constraint_fn(x))

    def _ode_fn(t, y, args):
        del args
        x = y
        x_flat = x.reshape((x.shape[0], -1))

        # Posterior variance scale (Eq. 16); r_t -> 0 as t -> 1.
        r_t_sq = (1.0 - t) ** 2 / (t ** 2 + (1.0 - t) ** 2)
        # Score-to-vector-field scale from Algorithm 1, line 8. Floored to keep
        # the first integration step finite.
        vf_scale = (1.0 - t) / jnp.maximum(t, 0.1)

        def _mu(x_t_flat: jax.Array):
            """Tweedie estimate of the clean sample for a single x_t.

            Returns ``mu`` as the primary output and the flat velocity ``v``
            as an auxiliary so we can reuse the same forward pass for both
            the unconditional drift and the chain-rule push back.
            """
            x_t = x_t_flat.reshape(data_shape)
            v_flat = model(x_t[None], jnp.array([t]))[0].ravel()
            return x_t_flat + (1.0 - t) * v_flat, v_flat

        def _single(x_t_flat: jax.Array):
            mu, vjp_mu, v_flat = jax.vjp(_mu, x_t_flat, has_aux=True)
            # Constraint residual and Jacobian A = dg/dmu at the clean estimate.
            g_val = _g(mu)
            A = jax.jacobian(_g)(mu)
            m = g_val.shape[0]
            # Solve (r_t^2 A A^T + eps I) z = g, then form A^T z. With our
            # convention g_val = A mu - y, so -A^T z = A^T (...)^{-1} (y - A mu)
            # which is the paper's gradient direction at mu (Eq. 15).
            AAT = A @ A.T
            z = jnp.linalg.solve(r_t_sq * AAT + eps_reg * jnp.eye(m), g_val)
            grad_at_mu = -A.T @ z
            # Push the gradient back to x_t via J^T = (d mu / d x_t)^T.
            (grad_at_xt,) = vjp_mu(grad_at_mu)
            return v_flat + guidance_scale * vf_scale * grad_at_xt

        x_dot_flat = jax.vmap(_single)(x_flat)
        return x_dot_flat.reshape(x.shape)

    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        diffrax.Midpoint(),
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=x_init,
        saveat=diffrax.SaveAt(ts=jnp.arange(dt, 1.0, dt), t1=True),
        stepsize_controller=diffrax.ConstantStepSize(),
    )
    print(solution.stats["num_steps"], "steps taken")
    xs = solution.ys
    x = xs[-1]

    return normalizer.unnormalize(x), normalizer.unnormalize(xs)
