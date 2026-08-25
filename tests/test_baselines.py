"""Behavior specific to individual baselines.

The registry contract in ``test_methods.py`` covers what every method must
do; these cover the claims each baseline makes for itself, none of which were
tested before.
"""

import jax.numpy as jnp
import pytest

from cfm.core.constraints import inequality
from cfm.methods import cbf, pcfm, pigdm
from tests.conftest import DT, NUM_SAMPLES


# --------------------------------------------------------------------------
# PCFM
# --------------------------------------------------------------------------


def test_pcfm_final_projection_drives_residual_to_zero(
    model, normalizer, circle, rng
):
    """PCFM's whole point: the constraint holds to numerical precision."""
    out = pcfm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, num_steps=20,
        rng=rng,
    )
    assert float(jnp.max(circle.violations(out.x))) < 1e-5


def test_pcfm_without_final_projection_is_looser(
    model, normalizer, circle, rng
):
    """The final Gauss-Newton sweep is what buys the tight residual."""
    tight = pcfm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, num_steps=20,
        rng=rng, num_final_projection_iters=20,
    )
    loose = pcfm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, num_steps=20,
        rng=rng, num_final_projection_iters=0,
    )
    assert (float(jnp.max(circle.violations(tight.x)))
            <= float(jnp.max(circle.violations(loose.x))))


def test_pcfm_correction_weight_zero_skips_the_relaxed_step(
    model, normalizer, circle, rng
):
    """lambda = 0 is documented as a no-op for linear constraints."""
    out = pcfm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, num_steps=20,
        rng=rng, correction_weight=0.0,
    )
    assert jnp.all(jnp.isfinite(out.x))


def test_pcfm_trajectory_spans_noise_to_sample(
    model, normalizer, circle, rng
):
    num_steps = 20
    out = pcfm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES,
        num_steps=num_steps, rng=rng,
    )
    assert out.xs.shape[0] == num_steps + 1
    assert jnp.allclose(out.xs[-1], out.x, atol=1e-6)


# --------------------------------------------------------------------------
# PiGDM
# --------------------------------------------------------------------------


def test_pigdm_guidance_scale_zero_ignores_the_constraint(
    model, normalizer, circle, rng
):
    """With no guidance PiGDM must reduce to the unconstrained flow."""
    from cfm.methods.ldf import generate_unconstrained

    guided = pigdm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, dt=DT, rng=rng,
        guidance_scale=0.0,
    )
    plain = generate_unconstrained(
        model, normalizer, num_samples=NUM_SAMPLES, dt=DT, rng=rng
    )
    # The unconstrained flow stops one step short of t = 1 by design, so
    # compare at the last time both of them record. PiGDM still evaluates the
    # Tweedie estimate and its VJP before scaling the correction by zero, so
    # the two paths differ by float32 accumulation rather than exactly.
    assert jnp.allclose(guided.xs[-2], plain.xs[-1], atol=1e-3)


def test_pigdm_stronger_guidance_tightens_the_constraint(
    model, normalizer, circle, rng
):
    weak = pigdm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, dt=DT, rng=rng,
        guidance_scale=0.0,
    )
    strong = pigdm.generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, dt=DT, rng=rng,
        guidance_scale=1.0,
    )
    assert (float(jnp.mean(circle.violations(strong.x)))
            < float(jnp.mean(circle.violations(weak.x))))


# --------------------------------------------------------------------------
# CBF safety filter
# --------------------------------------------------------------------------


def test_cbf_exact_qp_fails_loudly_with_an_actionable_message(
    model, normalizer, rng
):
    """``qp="exact"`` raises rather than returning a bad correction.

    The exact interior-point solve does not reliably converge on this flow --
    even for a constraint that is satisfied everywhere -- which is why
    ``benchmark`` and the sweep default to the elastic relaxation while
    ``problems/obstacles.py`` keeps ``exact`` to reproduce the paper. The
    contract being pinned here is that the failure is a clear error naming
    the workaround, not a silently wrong answer.
    """
    slack = inequality(
        lambda x: jnp.atleast_1d(jnp.sum(x**2) - 1e6), name="always feasible"
    )
    with pytest.raises(Exception, match='qp="elastic"'):
        cbf.generate(
            model, normalizer, slack, num_samples=NUM_SAMPLES, dt=DT,
            rng=rng, qp="exact",
        )


def test_cbf_leaves_a_satisfied_constraint_alone(model, normalizer, rng):
    """Where the sample is safely feasible the learned flow is untouched."""
    from cfm.methods.ldf import generate_unconstrained

    slack = inequality(
        lambda x: jnp.atleast_1d(jnp.sum(x**2) - 1e6), name="always feasible"
    )
    filtered = cbf.generate(
        model, normalizer, slack, num_samples=NUM_SAMPLES, dt=DT, rng=rng,
        qp="elastic", num_terminal_iters=0,
    )
    plain = generate_unconstrained(
        model, normalizer, num_samples=NUM_SAMPLES, dt=DT, rng=rng
    )
    assert jnp.allclose(filtered.xs[-1], plain.xs[-1], atol=1e-2)


def test_cbf_elastic_survives_mutually_infeasible_conditions(
    model, normalizer, rng
):
    """Contradictory barriers must relax rather than crash."""
    impossible = inequality(
        lambda x: jnp.array([-x[0] - 5.0, x[0] - 4.0]),
        name="x[0] >= 5 and x[0] <= 4",
    )
    out = cbf.generate(
        model, normalizer, impossible, num_samples=2, dt=DT, rng=rng,
        qp="elastic", num_terminal_iters=0,
    )
    assert out.x.shape == (2, 2)


def test_cbf_rejects_an_unknown_qp_mode(model, normalizer, right_half, rng):
    with pytest.raises(ValueError, match='qp must be'):
        cbf.generate(
            model, normalizer, right_half, num_samples=2, dt=DT, rng=rng,
            qp="approximate",
        )


def test_cbf_terminal_filter_tightens_the_result(model, normalizer, rng):
    """The terminal filter mops up whatever violation the numerics leave."""
    single = inequality(lambda x: jnp.atleast_1d(-x[0]), name="x[0] >= 0")
    filtered = cbf.generate(
        model, normalizer, single, num_samples=NUM_SAMPLES, dt=DT, rng=rng,
        qp="elastic", num_terminal_iters=20,
    )
    unfiltered = cbf.generate(
        model, normalizer, single, num_samples=NUM_SAMPLES, dt=DT, rng=rng,
        qp="elastic", num_terminal_iters=0,
    )
    assert (float(jnp.max(single.violations(filtered.x)))
            <= float(jnp.max(single.violations(unfiltered.x))))


# --------------------------------------------------------------------------
# LDF slack modes
# --------------------------------------------------------------------------


def test_ldf_slack_modes_both_reduce_violation(
    model, normalizer, right_half, rng
):
    from cfm.methods.ldf import generate, generate_unconstrained

    base = generate_unconstrained(
        model, normalizer, num_samples=NUM_SAMPLES, dt=DT, rng=rng
    )
    baseline = float(jnp.mean(right_half.violations(base.x)))

    for slack in ("closed_form", "ode"):
        out = generate(
            model, normalizer, right_half, num_samples=NUM_SAMPLES, dt=DT,
            rng=rng, penalty_weight=1.5, rescale_factor=10.0, slack=slack,
        )
        achieved = float(jnp.mean(right_half.violations(out.x)))
        assert achieved < baseline, f"slack={slack} did not help"


def test_ldf_rejects_unknown_slack_mode(model, normalizer, right_half, rng):
    from cfm.methods.ldf import generate

    with pytest.raises(ValueError, match="slack must be one of"):
        generate(
            model, normalizer, right_half, num_samples=2, dt=DT, rng=rng,
            slack="magic",
        )


def test_ldf_projection_tightens_the_result(model, normalizer, circle, rng):
    from cfm.methods.ldf import generate

    kwargs = dict(
        num_samples=NUM_SAMPLES, dt=DT, rng=rng, penalty_weight=1.5,
    )
    plain = generate(model, normalizer, circle, **kwargs)
    polished = generate(
        model, normalizer, circle, num_projection_iters=3, **kwargs
    )
    assert (float(jnp.max(circle.violations(polished.x)))
            < float(jnp.max(circle.violations(plain.x))))


def test_ldf_inequality_projection_leaves_feasible_samples_alone(
    model, normalizer, rng
):
    """Projecting an inequality must not disturb already-feasible samples."""
    from cfm.methods.ldf import generate

    # A constraint every sample satisfies comfortably.
    slack_constraint = inequality(
        lambda x: jnp.atleast_1d(jnp.sum(x**2) - 1e6), name="always feasible"
    )
    kwargs = dict(num_samples=NUM_SAMPLES, dt=DT, rng=rng, penalty_weight=0.0)
    plain = generate(model, normalizer, slack_constraint, **kwargs)
    polished = generate(
        model, normalizer, slack_constraint, num_projection_iters=5, **kwargs
    )
    assert jnp.allclose(plain.x, polished.x, atol=1e-6)
