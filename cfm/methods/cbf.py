"""SafeFlow control barrier function baseline.

Xiao et al., "SafeFlow: Safe Robot Motion Planning with Flow Matching via
Control Barrier Functions", https://arxiv.org/abs/2504.08661, Algorithm 1.

With barrier ``b = -h`` and ``dx/dt = v + u``, ``u`` solves

    min_u ||u||^2   s.t.   grad b_i . (v + u) + phi(t, b_i) b_i >= 0,
    phi(t, b) = phi0 if b >= 0 else omega / (1 - t)^2,

followed by a terminal projection onto ``h(x) <= 0``.
"""

import math
from typing import Callable, Tuple

import diffrax
import equinox as eqx
from flax import nnx
import jax
import jax.numpy as jnp
import qpax

from cfm.core.constraints import INEQUALITY, Constraint
from cfm.core.projection import gauss_newton_project
from cfm.core.solve import Samples, initial_noise, resolve_rng, save_times
from cfm.models.normalizer import Normalizer


def generate_cbf(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    rng: jax.Array = None,
    phi0: float = 1.0,
    omega: float = 4.0,
    qp: str = "exact",
    qp_penalty: float = 1e4,
    qp_tol: float = 1e-3,
    qp_max_iter: int = 30,
    num_terminal_iters: int = 20,
    eps_reg: float = 1e-6,
    solver: diffrax.AbstractSolver = diffrax.Midpoint(),
    stepsize_controller: diffrax.AbstractStepSizeController = (
        diffrax.ConstantStepSize()
    ),
    max_steps: int = 100_000,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples satisfying ``h(x) <= 0`` with a CBF safety filter.

    The barrier condition and QP are posed in unnormalized state space.

    Args:
        constraint_fn: ``h(x)`` on a single unnormalized sample.
        phi0: class-K gain where the sample is feasible.
        omega: gain of the ``omega / (1 - t)^2`` schedule where infeasible.
            The paper requires ``omega > 2``.
        qp: ``"exact"`` (errors if infeasible) or ``"elastic"`` (penalized
            relaxation, always solvable).
        qp_penalty: price per unit of constraint violation for ``"elastic"``.
        num_terminal_iters: Gauss-Newton iterations for the terminal filter.
        stepsize_controller: the schedule is stiff near t = 1; pair
            ``Tsit5()`` with a ``PIDController`` if fixed steps blow up.
    """
    if qp not in ("exact", "elastic"):
        raise ValueError(f'qp must be "exact" or "elastic", got {qp!r}')

    rng = resolve_rng(rng, seed)
    data_shape = model.data_shape
    num_vars = math.prod(data_shape)  # plain int: QP shapes need it static
    std_flat = jnp.broadcast_to(normalizer.std, data_shape).ravel()

    def _h(x_state_flat: jax.Array) -> jax.Array:
        return jnp.atleast_1d(constraint_fn(x_state_flat.reshape(data_shape)))

    def _phi(t: jax.Array, b: jax.Array) -> jax.Array:
        return jnp.where(b >= 0.0, phi0, omega / (1.0 - t + 1e-8) ** 2)

    def _safety_filter(J: jax.Array, a: jax.Array) -> jax.Array:
        """Solve min ||u||^2 / 2 s.t. J u <= a."""
        Q, q = jnp.eye(num_vars), jnp.zeros(num_vars)
        if qp == "elastic":
            return qpax.solve_qp_elastic_primal(
                Q, q, J, a, qp_penalty,
                solver_tol=qp_tol, max_iter=qp_max_iter,
            )

        u, _, _, _, converged, _ = qpax.solve_qp(
            Q, q, jnp.zeros((0, num_vars)), jnp.zeros((0,)), J, a,
            solver_tol=qp_tol, max_iter=qp_max_iter,
        )
        return eqx.error_if(
            u,
            converged != 1,
            "The CBF quadratic program did not solve: the barrier conditions "
            "are mutually infeasible at this state, or qp_max_iter is too "
            'small. Pass qp="elastic" to relax them instead.',
        )

    def _ode_fn(t, y, args):
        del args
        x = y
        x_flat = x.reshape((x.shape[0], -1))
        v_flat = model(x, jnp.full((x.shape[0],), t)).reshape(x_flat.shape)

        def _single(x_i: jax.Array, v_i: jax.Array) -> jax.Array:
            x_state = normalizer.unnormalize(x_i.reshape(data_shape)).ravel()
            v_state = v_i * std_flat

            h = _h(x_state)
            J = jax.jacfwd(_h)(x_state)  # (m, n)

            a = -(J @ v_state) - _phi(t, -h) * h

            u_state = _safety_filter(J, a)
            return v_i + u_state / std_flat

        return jax.vmap(_single)(x_flat, v_flat).reshape(x.shape)

    def _terminal_filter(x_state_flat: jax.Array) -> jax.Array:
        """Project onto h(x) <= 0 (Algorithm 1, line 17)."""
        return gauss_newton_project(
            _h, x_state_flat, num_iters=num_terminal_iters,
            active_only=True, eps_reg=eps_reg,
        )

    x_init = initial_noise(rng, num_samples, data_shape)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=x_init,
        saveat=diffrax.SaveAt(ts=save_times(dt, 1.0), t0=True),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    print(solution.stats["num_steps"], "steps taken")

    xs = normalizer.unnormalize(solution.ys)
    x = xs[-1]

    if num_terminal_iters > 0:
        x_flat = jax.vmap(_terminal_filter)(x.reshape((num_samples, -1)))
        x = x_flat.reshape((num_samples,) + data_shape)
        xs = xs.at[-1].set(x)

    return x, xs


def generate(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint: Constraint,
    **kwargs,
) -> Samples:
    """Registry entry point (inequality constraints only)."""
    if constraint.kind != INEQUALITY:
        raise ValueError(
            f"The CBF safety filter supports inequality constraints only, "
            f"got {constraint.kind!r}"
        )
    x, xs = generate_cbf(model, normalizer, constraint.fn, **kwargs)
    return Samples(x, xs)
