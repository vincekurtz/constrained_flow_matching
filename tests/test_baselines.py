"""Behavior specific to individual baselines."""

import jax.numpy as jnp
import numpy as np
import pytest

from cfm.core.constraints import inequality
from cfm.core import posterior_variance
from cfm.core.solve import wrap_constraint
from cfm.methods import cbf, dps, ldf, pcfm, pigdm
from tests.conftest import DT, NUM_SAMPLES, SEED

STEPS = int(1 / DT)
COMMON = dict(num_samples=NUM_SAMPLES, seed=SEED)


# --------------------------------------------------------------------------
# PCFM
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pcfm_default(run_once, model, normalizer, circle):
    return run_once("pcfm:default", lambda: pcfm.generate(
        model, normalizer, circle, num_steps=STEPS, **COMMON
    ))


def test_pcfm_final_projection_drives_residual_to_zero(pcfm_default, circle):
    """The constraint holds to numerical precision."""
    assert float(jnp.max(circle.violations(pcfm_default.x))) < 1e-5


def test_pcfm_without_final_projection_is_looser(
    run_once, model, normalizer, circle, pcfm_default
):
    """The final Gauss-Newton sweep is what buys the tight residual."""
    loose = run_once("pcfm:no-final-projection", lambda: pcfm.generate(
        model, normalizer, circle, num_steps=STEPS,
        num_final_projection_iters=0, **COMMON
    ))
    assert (float(jnp.max(circle.violations(pcfm_default.x)))
            <= float(jnp.max(circle.violations(loose.x))))


def test_pcfm_correction_weight_zero_skips_the_relaxed_step(
    run_once, model, normalizer, circle
):
    """lambda = 0 is documented as a no-op for linear constraints."""
    out = run_once("pcfm:no-correction", lambda: pcfm.generate(
        model, normalizer, circle, num_steps=STEPS, correction_weight=0.0,
        **COMMON
    ))
    assert jnp.all(jnp.isfinite(out.x))


def test_pcfm_trajectory_spans_noise_to_sample(pcfm_default):
    assert pcfm_default.xs.shape[0] == STEPS + 1
    assert jnp.allclose(pcfm_default.xs[-1], pcfm_default.x, atol=1e-6)


# --------------------------------------------------------------------------
# PiGDM
# --------------------------------------------------------------------------


@pytest.fixture(scope="session")
def pigdm_unguided(run_once, model, normalizer, circle):
    return run_once("pigdm:unguided", lambda: pigdm.generate(
        model, normalizer, circle, dt=DT, guidance_scale=0.0, **COMMON
    ))


def test_pigdm_guidance_scale_zero_ignores_the_constraint(
    pigdm_unguided, baseline
):
    """With no guidance PiGDM must reduce to the unconstrained flow."""
    # The unconstrained flow stops one step short of t = 1. Not exact because
    # PiGDM still computes the (zero-scaled) correction.
    assert jnp.allclose(pigdm_unguided.xs[-2], baseline.xs[-1], atol=1e-3)


def test_pigdm_stronger_guidance_tightens_the_constraint(
    run_once, model, normalizer, circle, pigdm_unguided
):
    strong = run_once("pigdm:guided", lambda: pigdm.generate(
        model, normalizer, circle, dt=DT, guidance_scale=1.0, **COMMON
    ))
    assert (float(jnp.mean(circle.violations(strong.x)))
            < float(jnp.mean(circle.violations(pigdm_unguided.x))))


# --------------------------------------------------------------------------
# OPC (PiGDM with an estimated posterior variance)
# --------------------------------------------------------------------------


class ZeroVelocity:
    data_shape = (2,)

    def __call__(self, x, t):
        return jnp.zeros_like(x)


def test_posterior_variance_estimate_matches_closed_form(identity_normalizer):
    # v = 0 gives mu = x_t, so x1 - mu = (1 - t)(x1 - x0) and, for x1 = 1,
    # r_t^2 = (1 - t)^2 (1 + 1).
    data = jnp.ones((4096, 2))
    ts, r2 = posterior_variance.estimate(
        ZeroVelocity(), identity_normalizer, data, num_times=5, num_noise=4)
    np.testing.assert_allclose(r2, 2 * (1 - ts) ** 2, rtol=0.03)


def test_posterior_variance_round_trips_and_reports_missing_tables(tmp_path):
    path = tmp_path / "table.json"
    posterior_variance.save(path, [0., 1.], [1., 0.], problem="toy")
    ts, r2 = posterior_variance.load(path)
    np.testing.assert_array_equal(ts, [0., 1.])
    np.testing.assert_array_equal(r2, [1., 0.])
    with pytest.raises(FileNotFoundError, match="posterior_variance"):
        posterior_variance.load(tmp_path / "missing.json")


def test_opc_with_the_pigdm_prior_reproduces_pigdm(
    run_once, model, normalizer, circle,
):
    ts = jnp.linspace(0., 1., 2001)
    prior = (1 - ts) ** 2 / (ts ** 2 + (1 - ts) ** 2)
    opc = pigdm.generate_opc(model, normalizer, circle, dt=DT,
                             posterior_variance=(ts, prior), **COMMON)
    ref = run_once("pigdm:guided", lambda: pigdm.generate(
        model, normalizer, circle, dt=DT, guidance_scale=1.0, **COMMON
    ))
    np.testing.assert_allclose(opc.x, ref.x, atol=1e-3)


def test_opc_requires_a_variance_table(model, normalizer, circle):
    with pytest.raises(ValueError, match="posterior_variance"):
        pigdm.generate_opc(model, normalizer, circle, dt=DT, **COMMON)


# --------------------------------------------------------------------------
# DPS
# --------------------------------------------------------------------------


class LinearVelocity:
    """v(x, t) = a x, so the clean estimate is (1 + (1 - t) a) x."""

    data_shape = (2,)

    def __init__(self, a):
        self.a = a

    def __call__(self, x, t):
        return self.a * x


def test_dps_drift_is_normalized_and_backpropagates_through_the_model(
    identity_normalizer,
):
    # g(x) = 3 x[0]: grad_x ||g(mu)|| = 3 sign(mu_0) (1 + (1 - t) a) e_0.
    a, t, zeta, dt = 0.5, 0.2, 0.1, 0.05
    residual = wrap_constraint(lambda x: 3 * x[0], identity_normalizer, (2,))
    flow = dps.dps_flow(LinearVelocity(a), residual, (2,),
                        step_size=zeta, dt=dt)
    x = jnp.array([[2., 1.], [-4., 1.], [0., 1.]])
    dmu = 1 + (1 - t) * a
    guide = 2 * zeta / dt * 3 * dmu
    np.testing.assert_allclose(flow(t, x, None), [
        [2 * a - guide, a], [-4 * a + guide, a], [0., a],
    ], rtol=1e-6)


@pytest.fixture(scope="session")
def dps_unguided(run_once, model, normalizer, circle):
    return run_once("dps:unguided", lambda: dps.generate(
        model, normalizer, circle, dt=DT, step_size=0.0, **COMMON
    ))


def test_dps_step_size_zero_ignores_the_constraint(dps_unguided, baseline):
    # Per-sample model calls round differently from the batched baseline.
    assert jnp.allclose(dps_unguided.xs[-2], baseline.xs[-1], atol=1e-3)


def test_dps_guidance_tightens_the_constraint(
    run_once, model, normalizer, circle, dps_unguided
):
    guided = run_once("dps:guided", lambda: dps.generate(
        model, normalizer, circle, dt=DT, step_size=0.02, **COMMON
    ))
    assert (float(jnp.mean(circle.violations(guided.x)))
            < float(jnp.mean(circle.violations(dps_unguided.x))))


def test_dps_rejects_inequalities(model, normalizer):
    with pytest.raises(ValueError, match="equality constraints only"):
        dps.generate(model, normalizer, inequality(lambda x: x[0]))


# --------------------------------------------------------------------------
# CBF safety filter
# --------------------------------------------------------------------------


def test_cbf_exact_qp_fails_loudly_with_an_actionable_message(
    model, normalizer, feasible
):
    """``qp="exact"`` fails to converge here and should say to use elastic."""
    with pytest.raises(Exception, match='qp="elastic"'):
        cbf.generate(
            model, normalizer, feasible, dt=DT, qp="exact", **COMMON
        )


def test_cbf_leaves_a_satisfied_constraint_alone(
    run_once, model, normalizer, feasible, baseline
):
    """Where the sample is safely feasible the learned flow is untouched."""
    filtered = run_once("cbf:feasible", lambda: cbf.generate(
        model, normalizer, feasible, dt=DT, qp="elastic",
        num_terminal_iters=0, **COMMON
    ))
    assert jnp.allclose(filtered.xs[-1], baseline.xs[-1], atol=1e-2)


def test_cbf_elastic_survives_mutually_infeasible_conditions(
    run_once, model, normalizer
):
    """Contradictory barriers must relax rather than crash."""
    from cfm.core.constraints import inequality

    impossible = inequality(
        lambda x: jnp.array([-x[0] - 5.0, x[0] - 4.0]),
        name="x[0] >= 5 and x[0] <= 4",
    )
    out = run_once("cbf:infeasible", lambda: cbf.generate(
        model, normalizer, impossible, dt=DT, qp="elastic",
        num_terminal_iters=0, **COMMON
    ))
    assert out.x.shape == (NUM_SAMPLES, 2)


def test_cbf_rejects_an_unknown_qp_mode(model, normalizer, right_half):
    with pytest.raises(ValueError, match='qp must be'):
        cbf.generate(
            model, normalizer, right_half, dt=DT, qp="approximate", **COMMON
        )


def test_cbf_terminal_filter_tightens_the_result(
    run_once, model, normalizer, right_half
):
    """The terminal filter reduces residual violation."""
    filtered = run_once("cbf:filtered", lambda: cbf.generate(
        model, normalizer, right_half, dt=DT, qp="elastic", **COMMON
    ))
    unfiltered = run_once("cbf:unfiltered", lambda: cbf.generate(
        model, normalizer, right_half, dt=DT, qp="elastic",
        num_terminal_iters=0, **COMMON
    ))
    assert (float(jnp.max(right_half.violations(filtered.x)))
            <= float(jnp.max(right_half.violations(unfiltered.x))))


# --------------------------------------------------------------------------
# LDF slack modes and projection
# --------------------------------------------------------------------------


@pytest.mark.parametrize("slack", ldf.SLACK_MODES)
def test_ldf_slack_modes_both_reduce_violation(
    slack, run_once, model, normalizer, right_half, baseline
):
    out = run_once(f"ldf:slack-{slack}", lambda: ldf.generate(
        model, normalizer, right_half, dt=DT, penalty_weight=1.5,
        rescale_factor=10.0, slack=slack, **COMMON
    ))
    assert (float(jnp.mean(right_half.violations(out.x)))
            < float(jnp.mean(right_half.violations(baseline.x))))


def test_ldf_rejects_unknown_slack_mode(model, normalizer, right_half):
    with pytest.raises(ValueError, match="slack must be one of"):
        ldf.generate(
            model, normalizer, right_half, dt=DT, slack="magic", **COMMON
        )


def test_ldf_projection_tightens_the_result(
    run_once, model, normalizer, circle
):
    kwargs = dict(dt=DT, penalty_weight=1.5, **COMMON)
    plain = run_once("ldf:circle", lambda: ldf.generate(
        model, normalizer, circle, **kwargs
    ))
    polished = run_once("ldf:circle-projected", lambda: ldf.generate(
        model, normalizer, circle, num_projection_iters=3, **kwargs
    ))
    assert (float(jnp.max(circle.violations(polished.x)))
            < float(jnp.max(circle.violations(plain.x))))


def test_ldf_inequality_projection_leaves_feasible_samples_alone(
    run_once, model, normalizer, feasible
):
    """Projecting an inequality must not disturb already-feasible samples."""
    kwargs = dict(dt=DT, penalty_weight=0.0, **COMMON)
    plain = run_once("ldf:feasible", lambda: ldf.generate(
        model, normalizer, feasible, **kwargs
    ))
    polished = run_once("ldf:feasible-projected", lambda: ldf.generate(
        model, normalizer, feasible, num_projection_iters=5, **kwargs
    ))
    assert jnp.allclose(plain.x, polished.x, atol=1e-6)
