"""Forward-pass behavior of the temporal U-Net.

Like ``test_flow_architectures``, the model is built once per session and
every assertion reuses it: each new input shape costs an XLA compilation.

The last two tests are the point of the architecture. A flattened MLP passes
everything above them and still generates jagged windows, so they check the
property that actually differs: the network mixes neighbouring timesteps, and
cannot respond to one timestep alone.
"""

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
    """The same net with its zero-initialized output layer perturbed.

    Straight out of ``__init__`` the output projection is zero, so the field
    is identically zero and every sensitivity test below would pass
    vacuously. One gradient-free nudge stands in for training.
    """
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
    """Down- and up-sampling must land back on the input horizon."""
    x = jnp.ones((BATCH, HORIZON, DIM))
    y = net(x, jnp.linspace(0, 1, BATCH))
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_output_starts_at_zero(net):
    """The output projection is zero-initialized, so v(x, t) = 0 at init."""
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
    """Perturbing one timestep must move the velocity at nearby ones.

    This is the inductive bias a flattened MLP has to learn instead: a
    window is a time series, so a change at step 8 says something about step
    9. Without it the model is free to emit independent per-step noise.
    """
    x = jnp.zeros((2, HORIZON, DIM))
    poked = x.at[1, HORIZON // 2, :].set(1.0)
    y = trained(jnp.concatenate([x[:1], poked[1:]]), jnp.full((2,), 0.5))
    delta = jnp.abs(y[1] - y[0]).sum(axis=-1)
    assert delta[HORIZON // 2 + 1] > 0.0
    assert delta[HORIZON // 2 - 1] > 0.0


def test_response_to_one_timestep_is_local(trained):
    """The response decays with distance from the perturbed timestep.

    Convolutions of width 5 over two downsampling levels give a wide but
    finite receptive field, so a poke is felt most strongly where it lands.
    """
    x = jnp.zeros((2, HORIZON, DIM))
    poked = x.at[1, HORIZON // 2, :].set(1.0)
    y = trained(jnp.concatenate([x[:1], poked[1:]]), jnp.full((2,), 0.5))
    delta = jnp.abs(y[1] - y[0]).sum(axis=-1)
    assert delta[HORIZON // 2] > delta[0]
    assert delta[HORIZON // 2] > delta[-1]


def test_resblock_preserves_length_and_changes_channels():
    """A block maps (batch, length, in) to (batch, length, out)."""
    block = TemporalResBlock(4, 8, time_dim=12, rngs=nnx.Rngs(0))
    y = block(jnp.ones((BATCH, HORIZON, 4)), jnp.ones((BATCH, 12)))
    assert y.shape == (BATCH, HORIZON, 8)
    assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# Wiring into the locomotion problems
#
# ``make_model`` is pure -- no disk, no download -- so the real Walker2D and
# Hopper models can be built here. Their channel count is the transition
# width, which differs per environment, and their horizon has to survive the
# downsampling; both are easy to break from the spec side and invisible until
# a training run starts.
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
