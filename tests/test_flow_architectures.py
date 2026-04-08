import jax.numpy as jnp
from flax import nnx

from architectures.flow import FlowMLP, SinusoidalPosEmb
from architectures.unet import FlowUNet


# ---------------------------------------------------------------------------
# SinusoidalPosEmb
# ---------------------------------------------------------------------------


def test_sinusoidal_output_shape():
    """Output shape is (batch, dim) for a batch of scalar time values."""
    emb = SinusoidalPosEmb(dim=16)
    t = jnp.linspace(0, 1, 8)
    out = emb(t)
    assert out.shape == (8, 16)


def test_sinusoidal_output_shape_single():
    """Works for a batch of size 1."""
    emb = SinusoidalPosEmb(dim=32)
    t = jnp.array([0.5])
    out = emb(t)
    assert out.shape == (1, 32)


def test_sinusoidal_values_bounded():
    """All output values lie in [-1, 1]."""
    emb = SinusoidalPosEmb(dim=16)
    t = jnp.linspace(0, 1, 100)
    out = emb(t)
    assert jnp.all(out >= -1.0) and jnp.all(out <= 1.0)


def test_sinusoidal_output_is_finite():
    """Forward pass produces finite values."""
    emb = SinusoidalPosEmb(dim=16)
    t = jnp.linspace(0, 1, 20)
    out = emb(t)
    assert jnp.all(jnp.isfinite(out))


def test_sinusoidal_different_times_give_different_embeddings():
    """Different time values produce different embeddings."""
    emb = SinusoidalPosEmb(dim=16)
    out0 = emb(jnp.array([0.0]))
    out1 = emb(jnp.array([1.0]))
    assert not jnp.allclose(out0, out1)


# ---------------------------------------------------------------------------
# FlowMLP
# ---------------------------------------------------------------------------


def test_flowmlp_output_shape():
    """Output shape matches (batch, *data_shape)."""
    model = FlowMLP(
        data_shape=(4,),
        time_embedding_size=16,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((8, 4))
    t = jnp.linspace(0, 1, 8)
    y = model(x, t)
    assert y.shape == (8, 4)


def test_flowmlp_output_is_finite():
    """Forward pass produces finite values."""
    model = FlowMLP(
        data_shape=(4,),
        time_embedding_size=16,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((8, 4))
    t = jnp.linspace(0, 1, 8)
    y = model(x, t)
    assert jnp.all(jnp.isfinite(y))


def test_flowmlp_sensitive_to_time():
    """Different time inputs produce different outputs."""
    model = FlowMLP(
        data_shape=(4,),
        time_embedding_size=16,
        hidden_sizes=(32,),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((1, 4))
    y0 = model(x, jnp.array([0.0]))
    y1 = model(x, jnp.array([1.0]))
    assert not jnp.allclose(y0, y1)


def test_flowmlp_sensitive_to_input():
    """Different x inputs produce different outputs."""
    model = FlowMLP(
        data_shape=(4,),
        time_embedding_size=16,
        hidden_sizes=(32,),
        rngs=nnx.Rngs(0),
    )
    t = jnp.array([0.5])
    x0 = jnp.zeros((1, 4))
    x1 = jnp.ones((1, 4))
    y0 = model(x0, t)
    y1 = model(x1, t)
    assert not jnp.allclose(y0, y1)


def test_flowmlp_image_shape():
    """Works for image-shaped inputs; output has the same shape as input."""
    model = FlowMLP(
        data_shape=(4, 4, 2),
        time_embedding_size=16,
        hidden_sizes=(32, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((8, 4, 4, 2))
    t = jnp.linspace(0, 1, 8)
    y = model(x, t)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# FlowUNet
# ---------------------------------------------------------------------------


def test_flowunet_output_shape():
    """Output shape matches (batch, H, W, C) for single-channel images."""
    model = FlowUNet(
        data_shape=(16, 16, 1),
        time_embedding_size=16,
        channels=(16, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((4, 16, 16, 1))
    t = jnp.linspace(0, 1, 4)
    y = model(x, t)
    assert y.shape == (4, 16, 16, 1)


def test_flowunet_multichannel():
    """Works for images with multiple channels."""
    model = FlowUNet(
        data_shape=(8, 8, 3),
        time_embedding_size=16,
        channels=(16, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((4, 8, 8, 3))
    t = jnp.linspace(0, 1, 4)
    y = model(x, t)
    assert y.shape == (4, 8, 8, 3)


def test_flowunet_output_is_finite():
    """Forward pass produces finite values."""
    model = FlowUNet(
        data_shape=(8, 8, 1),
        time_embedding_size=16,
        channels=(16, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((4, 8, 8, 1))
    t = jnp.linspace(0, 1, 4)
    y = model(x, t)
    assert jnp.all(jnp.isfinite(y))


def test_flowunet_sensitive_to_time():
    """Different time inputs produce different outputs."""
    model = FlowUNet(
        data_shape=(8, 8, 1),
        time_embedding_size=16,
        channels=(16, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((1, 8, 8, 1))
    y0 = model(x, jnp.array([0.0]))
    y1 = model(x, jnp.array([1.0]))
    assert not jnp.allclose(y0, y1)


def test_flowunet_sensitive_to_input():
    """Different x inputs produce different outputs."""
    model = FlowUNet(
        data_shape=(8, 8, 1),
        time_embedding_size=16,
        channels=(16, 32),
        rngs=nnx.Rngs(0),
    )
    t = jnp.array([0.5])
    x0 = jnp.zeros((1, 8, 8, 1))
    x1 = jnp.ones((1, 8, 8, 1))
    y0 = model(x0, t)
    y1 = model(x1, t)
    assert not jnp.allclose(y0, y1)


def test_flowunet_deeper_network():
    """Works with more resolution levels."""
    model = FlowUNet(
        data_shape=(16, 16, 1),
        time_embedding_size=16,
        channels=(8, 16, 32),
        rngs=nnx.Rngs(0),
    )
    x = jnp.ones((2, 16, 16, 1))
    t = jnp.linspace(0, 1, 2)
    y = model(x, t)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))
