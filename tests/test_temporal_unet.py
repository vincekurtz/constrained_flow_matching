"""Forward-pass behavior of the temporal U-Net."""

import jax.numpy as jnp
import pytest
from flax import nnx

from cfm.models.temporal_unet import FlowTemporalUNet, TemporalResBlock

BATCH = 4
HORIZON = 16
DIM = 6


@pytest.fixture(scope="session")
def net():
    return FlowTemporalUNet(
        data_shape=(HORIZON, DIM),
        time_embedding_size=8,
        channels=(8, 16, 32),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture(scope="session")
def trained(net):
    """A net with its zero-initialized output layer randomized, so
    sensitivity tests aren't vacuous."""
    model = FlowTemporalUNet(
        data_shape=(HORIZON, DIM),
        time_embedding_size=8,
        channels=(8, 16, 32),
        rngs=nnx.Rngs(0),
    )
    model.output_conv.kernel[...] = nnx.initializers.lecun_normal()(
        nnx.Rngs(1)(), model.output_conv.kernel.shape
    )
    return model


def test_output_matches_input_shape(net):
    x = jnp.ones((BATCH, HORIZON, DIM))
    y = net(x, jnp.linspace(0, 1, BATCH))
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_output_starts_at_zero(net):
    y = net(jnp.ones((BATCH, HORIZON, DIM)), jnp.linspace(0, 1, BATCH))
    assert jnp.all(y == 0.0)


def test_is_sensitive_to_time(trained):
    y = trained(jnp.ones((BATCH, HORIZON, DIM)), jnp.linspace(0, 1, BATCH))
    assert not jnp.allclose(y[0], y[-1])


def test_is_sensitive_to_input(trained):
    x = jnp.stack([jnp.full((HORIZON, DIM), float(i)) for i in range(BATCH)])
    y = trained(x, jnp.full((BATCH,), 0.5))
    assert not jnp.allclose(y[0], y[1])


def test_rejects_horizon_the_levels_cannot_halve():
    """Three levels halve the horizon twice, so it must be a multiple of 4."""
    with pytest.raises(AssertionError, match="divisible"):
        FlowTemporalUNet(
            data_shape=(6, DIM),
            time_embedding_size=8,
            channels=(8, 16, 32),
            rngs=nnx.Rngs(0),
        )


def test_neighbouring_timesteps_are_coupled(trained):
    """Perturbing one timestep moves the velocity at its neighbours."""
    x = jnp.zeros((2, HORIZON, DIM))
    poked = x.at[1, HORIZON // 2, :].set(1.0)
    y = trained(jnp.concatenate([x[:1], poked[1:]]), jnp.full((2,), 0.5))
    delta = jnp.abs(y[1] - y[0]).sum(axis=-1)
    assert delta[HORIZON // 2 + 1] > 0.0
    assert delta[HORIZON // 2 - 1] > 0.0


def test_response_to_one_timestep_is_local(trained):
    """The response is strongest at the perturbed timestep."""
    x = jnp.zeros((2, HORIZON, DIM))
    poked = x.at[1, HORIZON // 2, :].set(1.0)
    y = trained(jnp.concatenate([x[:1], poked[1:]]), jnp.full((2,), 0.5))
    delta = jnp.abs(y[1] - y[0]).sum(axis=-1)
    assert delta[HORIZON // 2] > delta[0]
    assert delta[HORIZON // 2] > delta[-1]


def test_resblock_preserves_length_and_changes_channels():
    block = TemporalResBlock(4, 8, time_dim=12, rngs=nnx.Rngs(0))
    y = block(jnp.ones((BATCH, HORIZON, 4)), jnp.ones((BATCH, 12)))
    assert y.shape == (BATCH, HORIZON, 8)
    assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# Locomotion models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["walker2d", "hopper"])
def test_locomotion_models_accept_a_real_window(name):
    from problems.locomotion import make_model
    from problems.locomotion_spec import HORIZON, SPECS

    spec = SPECS[name]
    model = make_model(spec)
    assert model.data_shape == (HORIZON, spec.transition_dim)

    x = jnp.zeros((2, HORIZON, spec.transition_dim))
    y = model(x, jnp.array([0.2, 0.8]))
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))
