"""Behavior specific to individual baselines.

The registry contract in ``test_methods.py`` covers what every method must
do; these cover the claims each baseline makes for itself.

Results go through the session-wide ``run_once`` cache, so a configuration
several tests need -- each baseline's default settings, most of all -- is
generated once and shared. Tests stay one-claim-each; only the compilation
is pooled.
"""

import jax.numpy as jnp
import pytest

from cfm.methods import cbf, ldf, pcfm, pigdm
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
    """PCFM's whole point: the constraint holds to numerical precision."""
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
    # The unconstrained flow stops one step short of t = 1 by design, so
    # compare at the last time both of them record. PiGDM still evaluates the
    # Tweedie estimate and its VJP before scaling the correction by zero, so
    # the two paths differ by float32 accumulation rather than exactly.
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
# CBF safety filter
# --------------------------------------------------------------------------


def test_cbf_exact_qp_fails_loudly_with_an_actionable_message(
    model, normalizer, feasible
):
    """``qp="exact"`` raises rather than returning a bad correction.

    The exact interior-point solve does not reliably converge on this flow --
    even for a constraint that is satisfied everywhere -- which is why
    ``benchmark`` and the sweep default to the elastic relaxation while
    ``problems/obstacles.py`` keeps ``exact`` to reproduce the paper. The
    contract being pinned here is that the failure is a clear error naming
    the workaround, not a silently wrong answer.
    """
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
    """The terminal filter mops up whatever violation the numerics leave."""
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
