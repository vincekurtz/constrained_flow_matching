"""Analytic checks for the singular penalty and benchmark failures."""

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfm import methods, sweep
from cfm.core.constraints import equality, inequality
from cfm.core.solve import initial_noise, wrap_constraint
from cfm.methods.time_varying_penalty import generate, penalty_flow
from cfm.models.normalizer import Normalizer


class ZeroVelocity:
    data_shape = (2,)

    def __call__(self, x, t):
        return jnp.zeros_like(x)


def test_force_scales_by_inverse_time_squared_and_residual_weight():
    # Physical g(x)=x[0], with x[0]=2*z[0]+1 and c=3:
    # grad_z (c*g)^2/2 = 18*(2*z[0]+1).
    normalizer = Normalizer(mean=jnp.array([1., 0.]),
                            std=jnp.array([2., 1.]))
    residual = wrap_constraint(lambda x: x[0], normalizer, (2,), 3.)
    flow = penalty_flow(ZeroVelocity(), residual)
    x = jnp.array([[1., 2.]])
    np.testing.assert_allclose(flow(0., x, None), [[-54., 0.]])
    np.testing.assert_allclose(flow(0.5, x, None), [[-216., 0.]])
    np.testing.assert_allclose(flow(0.9, x, None), [[-5400., 0.]],
                               rtol=1e-6)


def test_inequality_penalizes_only_positive_residual(identity_normalizer):
    residual = wrap_constraint(lambda x: x[0], identity_normalizer, (2,))
    flow = penalty_flow(ZeroVelocity(), residual, equality=False)
    x = jnp.array([[-2., 1.], [2., 1.]])
    np.testing.assert_allclose(flow(0.5, x, None), [[0., 0.], [-8., 0.]])


@pytest.mark.parametrize("kind", [equality, inequality])
def test_adaptive_solution_matches_integrated_linear_penalty(
    identity_normalizer, kind,
):
    # dx/dt=-c²*x/(1-t)² has x(T)=x(0)*exp(-c²*T/(1-T)).
    c, t1 = 0.7, 0.8
    x0 = initial_noise(jax.random.key(0), 8, (2,))
    out = generate(
        ZeroVelocity(), identity_normalizer, kind(lambda x: x[0]),
        num_samples=8, penalty_weight=c, t1=t1,
        solver=diffrax.Dopri5(),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-7),
    )
    expected = x0.at[:, 0].multiply(jnp.exp(-c**2 * t1 / (1 - t1)))
    if kind is inequality:
        expected = expected.at[:, 0].set(
            jnp.where(x0[:, 0] <= 0, x0[:, 0], expected[:, 0]))
    np.testing.assert_allclose(out.x, expected, rtol=1e-5, atol=2e-7)


def test_jitted_default_endpoint_and_failed_solve(identity_normalizer):
    run = jax.jit(lambda rng: methods.TIME_VARYING_PENALTY.run(
        ZeroVelocity(), identity_normalizer, equality(lambda x: x[0]),
        num_samples=2, rng=rng, penalty_weight=0., dt=0.1,
        save_trajectory=False,
    ))
    out = run(jax.random.key(0))
    assert out.xs.shape == (2, 2, 2)
    np.testing.assert_array_equal(out.x, out.xs[0])
    failed = generate(
        ZeroVelocity(), identity_normalizer, equality(lambda x: x[0]),
        num_samples=2, max_steps=1, throw=False,
    )
    assert bool(jnp.all(jnp.isnan(failed.x)))
    assert sweep._latex_violation(float("nan"), 2) == r"\textit{failed}"
    assert sweep._latex_violation(float("inf"), 2) == r"\textit{failed}"


@pytest.mark.parametrize("t1", [0., 1., 1.1])
def test_singular_or_invalid_endpoint_is_rejected(identity_normalizer, t1):
    with pytest.raises(ValueError, match="t1"):
        generate(ZeroVelocity(), identity_normalizer,
                 equality(lambda x: x[0]), t1=t1)


def test_table1_compares_weights_and_keeps_failures():
    cases = sweep.build_cases(sweep.load_config("experiments/table1.toml"))
    for variant, base, projected in [
        ("time_varying_penalty", "penalty", False),
        ("time_varying_penalty_projected", "penalty_projected", True),
    ]:
        new = [c for c in cases if c.variant == variant]
        assert len(new) == 7
        for case in new:
            old = next(c for c in cases if c.variant == base
                       and c.problem_label == case.problem_label)
            assert (case.gains["penalty_weight"]
                    == old.gains["penalty_weight"])
            assert case.gains["throw"] is False
            if projected:
                assert case.gains["num_projection_iters"] == 2
            else:
                assert "num_projection_iters" not in case.gains
