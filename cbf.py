"""Implements the SafeFlow (control barrier function) baseline.

Reference: Xiao et al., "SafeFlow: Safe Robot Motion Planning with Flow
Matching via Control Barrier Functions" (https://arxiv.org/abs/2504.08661),
Algorithm 1.

The method leaves the learned flow untouched wherever the sample is safely
inside the feasible set, and adds the minimum-norm correction ``u_t`` needed to
satisfy a control-barrier-function condition everywhere else:

    dx/dt = v(x, t) + u_t.

Writing the barrier as ``b(x) = -h(x) >= 0`` for our inequality convention
``h(x) <= 0``, the flow-matching barrier function (FMBF) condition of the paper
asks that

    grad b_i . (v + u) + phi(t, b_i) b_i >= 0    for every constraint i,

with the scheduling function

    phi(t, b) = phi0                 if b >= 0  (feasible),
                omega / (1 - t)^2    otherwise  (infeasible).

The blow-up of ``phi`` as ``t -> 1`` is what drives an infeasible sample back
into the feasible set before the flow terminates: for ``b < 0`` the condition
forces ``|b|`` to decay like ``exp(-omega / (1 - t))``. Unbounded, it also
makes the ODE arbitrarily stiff at the very end of the horizon, so we cap the
schedule at ``phi_max``; the residual violation this leaves behind is what the
terminal filter is for.

``u_t`` is the smallest correction meeting every condition at once, i.e. the
solution of the quadratic program of Algorithm 1, line 8,

    min_u ||u||^2   s.t.   grad b_i . u >= -a_i,
    a_i = grad b_i . v + phi(t, b_i) b_i.

In the paper each constraint acts on a single, disjoint waypoint of the
trajectory, so the QP separates and Algorithm 1 quotes its closed-form
solution ``u_i = -grad b_i a_i / ||grad b_i||^2`` for the violated
constraints. Our constraints share decision variables (every knot of a spline
moves every collision-check point), so the QP is solved jointly; it collapses
back to that closed form whenever only one constraint is active.

A terminal safety filter runs after the integration to mop up whatever
violation the numerics leave behind at ``t = 1``.
"""

from typing import Callable, Tuple

import diffrax
from flax import nnx
import jax
import jax.numpy as jnp

from architectures.normalizer import Normalizer


def generate_cbf(
    model: nnx.Module,
    normalizer: Normalizer,
    constraint_fn: Callable[[jax.Array], jax.Array],
    num_samples: int = 1000,
    dt: float = 0.01,
    seed: int = 0,
    phi0: float = 1.0,
    omega: float = 4.0,
    phi_max: float = 1e3,
    num_qp_iters: int = 50,
    num_terminal_iters: int = 20,
    eps_reg: float = 1e-6,
    solver: diffrax.AbstractSolver = diffrax.Midpoint(),
    stepsize_controller: diffrax.AbstractStepSizeController = (
        diffrax.ConstantStepSize()
    ),
    max_steps: int = 100_000,
) -> Tuple[jax.Array, jax.Array]:
    """Generate samples satisfying ``h(x) <= 0`` with a CBF safety filter.

    Implements Algorithm 1 of Xiao et al., 2025 (SafeFlow). Following the
    paper, the barrier condition and the resulting quadratic program are posed
    in the *unnormalized* state space: the learned velocity is denormalized,
    filtered, and the correction is normalized again before it is handed to
    the integrator. Adaptive step-size control (the paper uses embedded RK45)
    is available through ``stepsize_controller``, but the default matches the
    other solvers in this repository.

    Args:
        model: Trained flow model ``xdot = v(x, t)``. Must have a
            ``data_shape`` attribute.
        normalizer: Normalizer used during training, applied in reverse to
            produce samples in the original data space.
        constraint_fn: Differentiable function ``h(x)`` (operating on a single
            *unnormalized* sample) defining the inequality ``h(x) <= 0``. May
            return a scalar or a 1-D array.
        num_samples: Number of samples to generate.
        dt: Step size (or initial step size hint for adaptive controllers).
        seed: Random seed for the initial noise.
        phi0: Class-K gain used where the sample is feasible. Larger values
            let the flow approach the constraint boundary more freely.
        omega: Gain of the blow-up schedule ``omega / (1 - t)^2`` used where
            the sample is infeasible. The paper requires ``omega > 2``.
        phi_max: Cap on the scheduling function. The uncapped schedule
            diverges at ``t = 1`` and makes the flow unintegrably stiff there;
            the cap only kicks in over the last ``sqrt(omega / phi_max)`` of
            the horizon, by which point an infeasible sample has already been
            driven to within ``exp(-phi_max * sqrt(omega / phi_max))`` of the
            feasible set.
        num_qp_iters: Dual FISTA iterations used to solve the safety-filter
            QP at each right-hand side evaluation.
        num_terminal_iters: Gauss-Newton iterations for the terminal safety
            filter applied at ``t = 1``. Set to 0 to disable it.
        eps_reg: Ridge on the QP dual (a soft-constraint relaxation that keeps
            the filter well posed when the barrier conditions cannot all be
            met at once), and Tikhonov regulariser for the terminal filter.
        solver: diffrax solver to use. Defaults to ``Midpoint()``, matching
            the rest of the repository.
        stepsize_controller: diffrax step-size controller. Defaults to
            ``ConstantStepSize()``. The ``omega / (1 - t)^2`` schedule is
            stiff near ``t = 1`` and a fixed-step low-order rule can throw
            a sample off to infinity there, so pair a higher-order solver
            (``Tsit5()``) with a ``PIDController`` when that shows up; this
            is what Algorithm 1's embedded RK45 error control is for.
        max_steps: Maximum number of solver steps.

    Returns:
        x: Final samples of shape ``(num_samples, *data_shape)``.
        xs: Trajectories of shape ``(num_steps, num_samples, *data_shape)``,
            with the terminal filter applied to the last entry.
    """
    rng = jax.random.key(seed)
    data_shape = model.data_shape
    std_flat = jnp.broadcast_to(normalizer.std, data_shape).ravel()

    def _h(x_state_flat: jax.Array) -> jax.Array:
        """Constraint on a single flat *unnormalized* sample."""
        return jnp.atleast_1d(constraint_fn(x_state_flat.reshape(data_shape)))

    def _phi(t: jax.Array, b: jax.Array) -> jax.Array:
        """Scheduling function phi(t, b), blowing up where b < 0."""
        phi1 = jnp.minimum(omega / (1.0 - t + 1e-8) ** 2, phi_max)
        return jnp.where(b >= 0.0, phi0, phi1)

    def _safety_filter(J: jax.Array, a: jax.Array) -> jax.Array:
        """Minimum-norm ``u`` with ``grad b_i . u >= -a_i`` for all ``i``.

        With ``grad b = -J``, the dual of ``min ||u||^2 / 2`` subject to
        ``-J u >= -a`` is the nonnegative least-squares problem

            min_{lam >= 0}  lam^T (J J^T + eps I) lam / 2  +  a^T lam,
            u = -J^T lam,

        which we solve with projected gradient descent, Nesterov-accelerated.
        The Hessian is applied in factored form, and its exact Lipschitz
        constant is the largest eigenvalue of the small ``n x n`` matrix
        ``J^T J``, so each iteration costs two thin matrix-vector products.
        """
        L = jnp.linalg.eigvalsh(J.T @ J)[-1] + eps_reg
        lmbda = jnp.zeros_like(a)

        def _step(carry, _):
            lmbda, y, theta = carry
            grad = J @ (J.T @ y) + eps_reg * y + a
            lmbda_next = jnp.maximum(y - grad / L, 0.0)
            theta_next = 0.5 * (1.0 + jnp.sqrt(1.0 + 4.0 * theta**2))
            y_next = lmbda_next + ((theta - 1.0) / theta_next) * (
                lmbda_next - lmbda
            )
            return (lmbda_next, y_next, theta_next), None

        (lmbda, _, _), _ = jax.lax.scan(
            _step, (lmbda, lmbda, 1.0), None, length=num_qp_iters
        )
        return -J.T @ lmbda

    def _ode_fn(t, y, args):
        del args
        x = y
        x_flat = x.reshape((x.shape[0], -1))
        v_flat = model(x, jnp.full((x.shape[0],), t)).reshape(x_flat.shape)

        def _single(x_i: jax.Array, v_i: jax.Array) -> jax.Array:
            # Lines 6-7: work in state space, where the constraint lives.
            x_state = normalizer.unnormalize(x_i.reshape(data_shape)).ravel()
            v_state = v_i * std_flat

            # Barrier b = -h, so grad b = -J with J = dh/dx.
            h = _h(x_state)
            J = jax.jacfwd(_h)(x_state)  # (m, n)

            # a_i = grad b_i . v + phi(t, b_i) b_i, with b_i = -h_i.
            a = -(J @ v_state) - _phi(t, -h) * h

            # Line 8: the CBF quadratic program, then line 9, back to
            # normalized space for the integrator.
            u_state = _safety_filter(J, a)
            return v_i + u_state / std_flat

        return jax.vmap(_single)(x_flat, v_flat).reshape(x.shape)

    def _terminal_filter(x_state_flat: jax.Array) -> jax.Array:
        """Line 17: argmin ||x - x_1-|| subject to h(x) <= 0.

        Solved with a Gauss-Newton active-set iteration: at each pass the
        violated constraints are linearized and the min-norm step that zeros
        them is taken, which lands on the boundary of the feasible set.
        """
        def _step(x, _):
            h = _h(x)
            J = jax.jacfwd(_h)(x)
            active = h > 0.0
            J_act = jnp.where(active[:, None], J, 0.0)
            r = jnp.where(active, h, 0.0)
            m = h.shape[0]
            z = jnp.linalg.solve(
                J_act @ J_act.T + eps_reg * jnp.eye(m), r
            )
            return x - J_act.T @ z, None

        x_state_flat, _ = jax.lax.scan(
            _step, x_state_flat, None, length=num_terminal_iters
        )
        return x_state_flat

    x_init = jax.random.normal(rng, (num_samples,) + data_shape)

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(_ode_fn),
        solver,
        t0=0.0,
        t1=1.0,
        dt0=dt,
        y0=x_init,
        saveat=diffrax.SaveAt(ts=jnp.arange(dt, 1.0, dt), t0=True, t1=True),
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    print(solution.stats["num_steps"], "steps taken")

    # Trajectories are normalized; the constraint (and the filter) are not.
    xs = normalizer.unnormalize(solution.ys)
    x = xs[-1]

    if num_terminal_iters > 0:
        x_flat = jax.vmap(_terminal_filter)(x.reshape((num_samples, -1)))
        x = x_flat.reshape((num_samples,) + data_shape)
        xs = xs.at[-1].set(x)

    return x, xs
