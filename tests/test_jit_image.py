"""Offline checks against the authors' PyTorch JiT forward and input VJP."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfm.models.jit_image import JiT, JiTImageFlow
from cfm.models.normalizer import Normalizer


@pytest.fixture(scope="module")
def tiny_jit():
    with np.load(Path(__file__).parent / "goldens/jit_image.npz") as data:
        fixture = dict(data)
    weights = {
        k: v
        for k, v in fixture.items()
        if k
        not in {
            "x",
            "t",
            "labels",
            "y",
            "cotangent",
            "gradient",
        }
    }
    return JiT(
        weights, size=16, patch=4, heads=4, depth=2, context_start=1
    ), fixture


def test_clean_prediction_and_input_vjp_match_upstream(tiny_jit):
    model, f = tiny_jit
    with jax.default_matmul_precision("highest"):
        y, pullback = jax.vjp(
            lambda x: model(x, jnp.array(f["t"]), jnp.array(f["labels"])),
            jnp.array(f["x"]),
        )
        grad = pullback(jnp.array(f["cotangent"]))[0]
    # Nonzero trained-style gates exercise attention and all residual blocks.
    assert np.linalg.norm(f["gradient"]) > 1
    np.testing.assert_allclose(y, f["y"], atol=2e-6, rtol=2e-5)
    np.testing.assert_allclose(grad, f["gradient"], atol=2e-6, rtol=2e-5)


def test_incompatible_checkpoint_is_rejected(tiny_jit):
    model, _ = tiny_jit
    with pytest.raises(ValueError, match="position embedding"):
        JiT(model.p, size=32, patch=4, heads=4, depth=2, context_start=1)
    with pytest.raises(ValueError, match="depth"):
        JiT(model.p, size=16, patch=4, heads=4, depth=3, context_start=1)


def test_guidance_one_velocity_and_vjp_match_native_clean_predictor(tiny_jit):
    net, f = tiny_jit
    model = JiTImageFlow(net, cfg=1.0)
    x, t = jnp.array(f["x"][:1]), jnp.array(f["t"][:1])
    cot = jnp.array(f["cotangent"][:1])
    velocity, pullback = jax.vjp(lambda z: model(z, t), x)
    denominator = 1 - f["t"][0]
    np.testing.assert_allclose(
        velocity, (f["y"][:1] - f["x"][:1]) / denominator,
        atol=3e-6, rtol=2e-5,
    )
    np.testing.assert_allclose(
        pullback(cot)[0], (f["gradient"][:1] - cot) / denominator,
        atol=3e-6, rtol=2e-5,
    )


def test_adapter_preserves_noise_scale_cfg_interval_and_endpoint():
    class CleanPredictor:
        data_shape = (2, 2, 3)

        def __call__(self, x, t, labels):
            return (
                0.1 * x
                + jnp.where(labels == 207, 0.4, 0.1)[:, None, None, None]
            )

    model = JiTImageFlow(CleanPredictor(), cfg=2.0, noise_scale=2.0, t_eps=.05)
    t = jnp.array([0.0, 0.1, 0.2, 0.8, 0.99, 1.0])
    x = jnp.ones((6, 2, 2, 3)) * 0.3
    native_z = 2 * x
    scale = jnp.array([1.0, 1.0, 2.0, 2.0, 2.0, 1.0])[:, None, None, None]
    clean = 0.1 * native_z + 0.1 + scale * 0.3
    expected = (clean - native_z) / (
        2 * jnp.maximum(1 - t, .05)[:, None, None, None]
    )
    np.testing.assert_allclose(model(x, t), expected, atol=1e-6)
    mu = x + (1 - t[:, None, None, None]) * model(x, t)
    norm = Normalizer(0.5, 1.0)
    np.testing.assert_allclose(
        norm.unnormalize(mu[:4]), 0.5 + 0.5 * clean[:4], atol=1e-6
    )
    # Near t=1, the native floor blends toward the current state rather
    # than repeatedly projecting through the learned denoiser.
    np.testing.assert_allclose(mu[4], .8 * x[4] + .2 * clean[4] / 2,
                               atol=1e-6)
    np.testing.assert_allclose(mu[5], x[5], atol=1e-6)


def test_adapter_rejects_invalid_configuration(tiny_jit):
    net, _ = tiny_jit
    with pytest.raises(ValueError, match="label"):
        JiTImageFlow(net, label=1000)
    with pytest.raises(ValueError, match="noise scale"):
        JiTImageFlow(net, noise_scale=0)


def test_prepare_checks_download_integrity_before_loading(
    tmp_path, monkeypatch
):
    from experiments import prepare_imagenet

    source = tmp_path / "partial.pth"
    source.write_bytes(b"incomplete checkpoint")
    monkeypatch.setattr(
        prepare_imagenet,
        "load_pretrained",
        lambda *a, **kw: pytest.fail("Must verify first"),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        prepare_imagenet.prepare(source, tmp_path / "model.pkl")


def test_loader_selects_ema1_and_rgb_normalization(tmp_path, monkeypatch):
    import torch
    from cfm.models import jit_image

    source = tmp_path / "model.pth"
    torch.save(
        {
            "model": {"weight": torch.tensor(1.0)},
            "model_ema1": {"weight": torch.tensor(2.0)},
            "model_ema2": {"weight": torch.tensor(3.0)},
        },
        source,
    )
    selected = {}

    def backbone(weights):
        selected.update(weights)
        return type("Backbone", (), {"data_shape": (512, 512, 3)})()

    monkeypatch.setattr(jit_image, "JiT", backbone)
    model, norm = jit_image.load_pretrained(source, label=207)
    assert selected["weight"] == 2
    assert model.model.label == 207 and model.model.noise_scale == 2
    assert model.model.t_eps == .005 and model.model.cfg == 1.
    np.testing.assert_allclose(
        norm.unnormalize(jnp.array([-0.25, 0.0, 0.25])), [0.0, 0.5, 1.0]
    )


def test_data_coordinates_preserve_physical_endpoint_and_clean_estimate():
    from cfm.models.jit_image import DataScaleFlow

    class ExpandingFlow:
        data_shape = (2,)

        def __call__(self, x, t):
            return x / (1 + t[:, None])

    a = .25
    model = DataScaleFlow(ExpandingFlow(), data_scale=a)
    initial = jnp.array([[.3, -.7], [.8, .1]])
    t = jnp.array([.2, .8])
    y = (1 + (2 * a - 1) * t[:, None]) * initial
    v = model(y, t)
    np.testing.assert_allclose(v, (2 * a - 1) * initial, atol=1e-7)
    clean = y + (1 - t[:, None]) * v
    np.testing.assert_allclose(clean * (.5 / a), initial, atol=1e-7)
