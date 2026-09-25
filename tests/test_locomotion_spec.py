"""The Walker2D and Hopper roof constraint and transition layout."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from problems.locomotion_spec import (
    HOPPER,
    HORIZON,
    SPECS,
    WALKER2D,
    barrier,
    heights,
    make_constraint,
    make_constraint_fn,
    vertical_velocities,
)

SPEC_LIST = [WALKER2D, HOPPER]
IDS = [spec.name for spec in SPEC_LIST]


def window(spec, z=0.0, vz=0.0, fill=0.0):
    """A window whose height and velocity columns hold known values."""
    x = jnp.full((HORIZON, spec.transition_dim), fill)
    x = x.at[:, spec.z_index].set(z)
    return x.at[:, spec.vz_index].set(vz)


def test_specs_are_keyed_by_name():
    assert SPECS == {"walker2d": WALKER2D, "hopper": HOPPER}


def test_spec_dims_match_the_environments():
    """D4RL walker2d is 6 + 17 and hopper 3 + 11, x-position excluded."""
    assert (WALKER2D.action_dim, WALKER2D.obs_dim) == (6, 17)
    assert (HOPPER.action_dim, HOPPER.obs_dim) == (3, 11)
    assert WALKER2D.transition_dim == 23
    assert HOPPER.transition_dim == 14


def test_indices_are_actions_first():
    """The constrained columns are action_dim + the observation index."""
    assert (WALKER2D.z_index, WALKER2D.vz_index) == (6, 15)
    assert (HOPPER.z_index, HOPPER.vz_index) == (3, 9)


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_indices_are_inside_a_transition(spec):
    assert 0 <= spec.z_index < spec.transition_dim
    assert 0 <= spec.vz_index < spec.transition_dim
    assert spec.z_index != spec.vz_index


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_residual_has_one_row_per_timestep(spec):
    h = make_constraint_fn(spec)(jnp.zeros((HORIZON, spec.transition_dim)))
    assert h.shape == (HORIZON,)


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_residual_is_the_height_rule(spec):
    """h = z + phi * vz - h_r."""
    z, vz = 1.1, 0.7
    h = make_constraint_fn(spec)(window(spec, z=z, vz=vz))
    expected = z + spec.phi * vz - spec.height_limit
    np.testing.assert_allclose(np.asarray(h), expected, atol=1e-6)


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_residual_reads_only_z_and_vz(spec):
    fn = make_constraint_fn(spec)
    clean = window(spec, z=1.2, vz=0.3)
    noisy = window(spec, z=1.2, vz=0.3, fill=7.5)
    np.testing.assert_array_equal(np.asarray(fn(clean)), np.asarray(fn(noisy)))


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_gradient_touches_only_two_columns(spec):
    """d h / d x is 1 at the height and phi at the velocity."""
    fn = make_constraint_fn(spec)
    grad = jax.grad(lambda x: jnp.sum(fn(x)))(
        jnp.zeros((HORIZON, spec.transition_dim))
    )
    expected = np.zeros((HORIZON, spec.transition_dim), dtype=np.float32)
    expected[:, spec.z_index] = 1.0
    expected[:, spec.vz_index] = spec.phi
    np.testing.assert_allclose(np.asarray(grad), expected, atol=1e-6)


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_a_low_window_is_feasible_and_a_high_one_is_not(spec):
    fn = make_constraint_fn(spec)
    assert np.all(np.asarray(fn(window(spec, z=0.5))) < 0)
    assert np.all(np.asarray(fn(window(spec, z=3.0))) > 0)
    np.testing.assert_allclose(
        np.asarray(fn(window(spec, z=spec.height_limit))), 0.0, atol=1e-6
    )


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_the_velocity_term_flips_feasibility(spec):
    """Just under the roof, rising fast violates and falling does not."""
    fn = make_constraint_fn(spec)
    just_under = spec.height_limit - 0.05
    rising = fn(window(spec, z=just_under, vz=2.0))
    falling = fn(window(spec, z=just_under, vz=-2.0))
    assert np.all(np.asarray(rising) > 0)
    assert np.all(np.asarray(falling) < 0)


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_constraint_is_an_inequality_with_a_readable_name(spec):
    constraint = make_constraint(spec)
    assert constraint.kind == "inequality"
    assert f"{spec.height_limit:g}" in constraint.name


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_violation_is_zero_when_feasible_and_positive_when_not(spec):
    constraint = make_constraint(spec)
    assert float(constraint.violation(window(spec, z=0.5))) == 0.0
    over = spec.height_limit + 0.25
    np.testing.assert_allclose(
        float(constraint.violation(window(spec, z=over))), 0.25, atol=1e-6
    )


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_violations_are_batched(spec):
    constraint = make_constraint(spec)
    batch = jnp.stack([window(spec, z=0.5), window(spec, z=3.0)])
    v = np.asarray(constraint.violations(batch))
    assert v.shape == (2,)
    assert v[0] == 0.0 and v[1] > 0.0


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_options_override_the_defaults(spec):
    lower = spec.height_limit - 0.3
    strict = make_constraint(spec, height_limit=lower)
    default = make_constraint(spec)
    x = window(spec, z=1.0, vz=0.1)
    assert np.all(np.asarray(strict.fn(x)) > np.asarray(default.fn(x)))
    assert f"{lower:g}" in strict.name

    no_velocity = make_constraint(spec, phi=0.0)
    np.testing.assert_allclose(
        np.asarray(no_velocity.fn(window(spec, z=1.0, vz=5.0))),
        1.0 - spec.height_limit,
        atol=1e-6,
    )


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_barrier_is_the_negated_residual(spec):
    x = window(spec, z=1.2, vz=0.4)
    np.testing.assert_allclose(
        np.asarray(barrier(spec, x)),
        -np.asarray(make_constraint_fn(spec)(x)),
        atol=1e-6,
    )


@pytest.mark.parametrize("spec", SPEC_LIST, ids=IDS)
def test_trace_accessors_pull_the_right_columns(spec):
    x = window(spec, z=1.25, vz=-0.5)
    np.testing.assert_allclose(np.asarray(heights(spec, x)), 1.25, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(vertical_velocities(spec, x)), -0.5, atol=1e-6
    )
