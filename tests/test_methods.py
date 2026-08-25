"""The contract every registered method must satisfy.

One test body, parameterized over the method registry, so a newly registered
baseline is covered the moment it is added rather than needing its own file.

Each method is generated from exactly once (see ``run_once`` in conftest);
the assertions below all inspect that single result. Properties that belong
to the shared plumbing rather than to any individual method -- seeding, in
particular -- are tested directly against ``cfm.core.solve`` and end-to-end
through LDF only, instead of once per method.
"""

import jax
import jax.numpy as jnp
import pytest

from cfm import methods

from cfm.core.constraints import EQUALITY, INEQUALITY
from cfm.core.solve import Samples, initial_noise, resolve_rng
from cfm.methods.ldf import generate, generate_unconstrained
from tests.conftest import (
    DT, NUM_SAMPLES, PENALTY_WEIGHT, SEED, constraint_for,
)

ALL = sorted(methods.METHODS.values(), key=lambda m: m.name)
IDS = [m.name for m in ALL]


def run(method, model, normalizer, constraint, **overrides):
    """Invoke a method with test-scale settings it can actually accept."""
    cfg = {"num_samples": NUM_SAMPLES, "seed": SEED}
    cfg.update(overrides)
    if method.name in methods.USES_DT:
        cfg.setdefault("dt", DT)
    else:
        cfg.setdefault("num_steps", int(1 / DT))
    # Only the LDF family takes penalty_weight as a constraint scale; the
    # other methods' gains are already tuned in the registry.
    if method.name in ("ldf", "penalty"):
        cfg.setdefault("penalty_weight", PENALTY_WEIGHT)
    return method.run(model, normalizer, constraint, **cfg)


@pytest.fixture(scope="session")
def result_for(run_once, model, normalizer, circle, right_half):
    """The one result per method that the contract tests below share."""
    def get(method):
        kind = EQUALITY if EQUALITY in method.supports else INEQUALITY
        constraint = constraint_for(kind, circle, right_half)
        out = run_once(
            f"contract:{method.name}",
            lambda: run(method, model, normalizer, constraint),
        )
        return constraint, out

    return get


@pytest.fixture(scope="session")
def unconstrained(run_once, model, normalizer):
    return run_once(
        "contract:baseline",
        lambda: generate_unconstrained(
            model, normalizer, num_samples=NUM_SAMPLES, dt=DT, seed=SEED
        ),
    )


@pytest.mark.parametrize("method", ALL, ids=IDS)
def test_returns_samples_with_correct_shapes(method, result_for):
    _, out = result_for(method)

    assert isinstance(out, Samples)
    assert out.x.shape == (NUM_SAMPLES, 2)
    assert out.xs.ndim == 3
    assert out.xs.shape[1:] == (NUM_SAMPLES, 2)


@pytest.mark.parametrize("method", ALL, ids=IDS)
def test_output_is_finite(method, result_for):
    _, out = result_for(method)
    assert jnp.all(jnp.isfinite(out.x))


@pytest.mark.parametrize("method", ALL, ids=IDS)
def test_trajectory_ends_at_final_sample(method, result_for):
    """``xs[-1]`` is the sample that was returned, post-projection."""
    _, out = result_for(method)
    assert jnp.allclose(out.xs[-1], out.x, atol=1e-6)


@pytest.mark.parametrize("method", ALL, ids=IDS)
def test_rejects_unsupported_constraint_kind(
    method, model, normalizer, circle, right_half
):
    """A method must refuse a constraint kind it cannot handle."""
    unsupported = {EQUALITY, INEQUALITY} - set(method.supports)
    if not unsupported:
        pytest.skip(f"{method.name} supports every constraint kind")
    kind = unsupported.pop()
    with pytest.raises(ValueError, match="does not support"):
        method.run(
            model, normalizer, constraint_for(kind, circle, right_half),
            num_samples=2,
        )


# ---------------------------------------------------------------------------
# The scientific claim: constraining actually reduces violation.
# ---------------------------------------------------------------------------

CONSTRAINED = [m for m in ALL if m.name != "unconstrained"]


@pytest.mark.parametrize(
    "method", CONSTRAINED, ids=[m.name for m in CONSTRAINED]
)
def test_reduces_violation_vs_unconstrained(
    method, result_for, unconstrained
):
    """Every method must beat the unconstrained flow on its own constraint."""
    constraint, out = result_for(method)

    baseline = float(jnp.mean(constraint.violations(unconstrained.x)))
    achieved = float(jnp.mean(constraint.violations(out.x)))
    assert achieved < baseline, (
        f"{method.name}: violation {achieved:.3e} did not improve on the "
        f"unconstrained baseline {baseline:.3e}"
    )


def test_ldf_beats_penalty_only(result_for):
    """The dual dynamics must earn their keep.

    This is the guard against a regression that silently zeroes lambda: with
    rescale_factor = 0 the flow is a pure penalty, and LDF should be clearly
    tighter. Measured at 15.8x on the trained star model.
    """
    circle, ldf = result_for(methods.LDF)
    _, penalty = result_for(methods.PENALTY)

    v_ldf = float(jnp.mean(circle.violations(ldf.x)))
    v_pen = float(jnp.mean(circle.violations(penalty.x)))
    assert v_ldf < 0.5 * v_pen, (
        f"LDF {v_ldf:.3e} vs penalty-only {v_pen:.3e}: the dual dynamics are "
        f"not helping, which suggests the multiplier flow is inactive"
    )


def test_penalty_only_freezes_multipliers(
    model, normalizer, circle, result_for
):
    """rescale_factor = 0 must be exactly what the penalty ablation runs."""
    explicit = generate(
        model, normalizer, circle, num_samples=NUM_SAMPLES, dt=DT, seed=SEED,
        penalty_weight=PENALTY_WEIGHT, rescale_factor=0.0,
    )
    _, ablation = result_for(methods.PENALTY)
    assert jnp.array_equal(explicit.x, ablation.x)


# ---------------------------------------------------------------------------
# Seeding. Shared by every method through cfm.core.solve, so it is tested
# there directly and end-to-end through one method rather than all six.
# ---------------------------------------------------------------------------


def test_resolve_rng_prefers_an_explicit_key():
    key = jax.random.key(7)
    assert jnp.array_equal(resolve_rng(key, 0), key)
    assert jnp.array_equal(resolve_rng(None, 7), key)


def test_initial_noise_depends_on_the_seed():
    a = initial_noise(resolve_rng(None, 0), 4, (2,))
    b = initial_noise(resolve_rng(None, 1), 4, (2,))
    assert a.shape == (4, 2)
    assert not jnp.allclose(a, b)


def test_generation_is_reproducible_from_a_seed(model, normalizer, circle):
    """Same seed, same samples; different seed, different samples."""
    kwargs = dict(
        num_samples=NUM_SAMPLES, dt=DT, penalty_weight=PENALTY_WEIGHT
    )
    a = generate(model, normalizer, circle, seed=SEED, **kwargs)
    b = generate(model, normalizer, circle, seed=SEED, **kwargs)
    c = generate(model, normalizer, circle, seed=SEED + 1, **kwargs)

    assert jnp.array_equal(a.x, b.x)
    assert not jnp.allclose(a.x, c.x)
