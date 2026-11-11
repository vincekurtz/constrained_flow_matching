"""Offline fixed-pixel equality, CLI, and figure tests for image inpainting."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from PIL import Image

from cfm import cli, methods
from cfm.core.constraints import equality
from cfm.methods import ldf
from cfm.models.normalizer import Normalizer
from problems import imagenet


def test_reference_and_masks():
    reference, mask = imagenet.reference_and_mask()
    assert reference.shape == imagenet.IMAGE_SHAPE
    assert mask.sum() == 512 * 512 - 192 * 192
    assert mask[:288].all() and mask[480:].all()
    assert mask[:, :96].all() and mask[:, 288:].all()
    assert not mask[288:480, 96:288].any()
    assert reference.min() >= 0 and reference.max() <= 1
    _, lower = imagenet.reference_and_mask(mask="lower-half")
    assert lower[:256].all() and not lower[256:].any()
    _, center = imagenet.reference_and_mask(mask="center")
    assert center.sum() == 512 * 512 - 192 * 192
    constraint = imagenet.make_constraint()
    assert float(constraint.violation(jnp.array(reference))) == 0


def test_custom_mask_and_reference(tmp_path):
    ref = tmp_path / "reference.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (512, 400), (30, 60, 90)).save(ref)
    pixels = np.zeros((512, 512), dtype=np.uint8)
    pixels[:64] = 255
    Image.fromarray(pixels).save(mask)
    x, observed = imagenet.reference_and_mask(reference=ref, mask_path=mask)
    np.testing.assert_allclose(x[0, 0], np.array([30, 60, 90]) / 255)
    assert observed[:64].all() and not observed[64:].any()
    Image.new("L", (32, 32), 255).save(mask)
    with pytest.raises(ValueError, match="512 x 512"):
        imagenet.reference_and_mask(mask_path=mask)
    Image.new("L", (512, 512), 255).save(mask)
    with pytest.raises(ValueError, match="both observed and missing"):
        imagenet.reference_and_mask(mask_path=mask)


def test_fixed_pixel_equality_and_zero_subgradient():
    ref = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 12
    observed = np.array([[True, False], [True, False]])
    c = imagenet.fixed_pixel_constraint(ref, observed, reduction="norm")
    assert float(c.violation(jnp.array(ref))) == 0
    assert jnp.all(jax.jacobian(c.fn)(jnp.array(ref)) == 0)
    changed = ref.copy()
    changed[~observed] = -100
    assert float(c.fn(jnp.array(changed))[0]) == 0
    # Opposite signed errors must not cancel out.
    changed[0, 0, 0] += .5
    changed[1, 0, 0] -= .5
    assert float(c.fn(jnp.array(changed))[0]) > .7


@pytest.mark.parametrize("std", [.5, 1., 2.])
@pytest.mark.parametrize("eps_reg", [1e-8, .03])
def test_norm_and_pixelwise_equalities_give_identical_pigdm(std, eps_reg):
    class Model:
        data_shape = (2, 2, 1)

        def __call__(self, x, t):
            return -.3 * x + .1 * jnp.sin(jnp.mean(
                x, axis=(1, 2, 3), keepdims=True,
            ))

    ref = np.array([.2, 0., .6, 0.]).reshape(2, 2, 1)
    c = imagenet.fixed_pixel_constraint(
        ref, np.array([[1, 0], [1, 0]]), reduction="norm",
    )
    vector = equality(
        lambda x: x.ravel()[jnp.array([0, 2])] - jnp.array([.2, .6])
    )
    norm = Normalizer(jnp.array(.5), jnp.array(std))
    cfg = dict(num_samples=2, dt=.01, seed=0, eps_reg=eps_reg,
               guidance_scale=.6)
    a = methods.PIGDM.run(Model(), norm, c, **cfg)
    b = methods.PIGDM.run(Model(), norm, vector, **cfg)
    np.testing.assert_allclose(a.x, b.x, atol=1e-6)


def test_ldf_keeps_each_pixel_equality_and_pigdm_reduces_exactly():
    reference, observed = imagenet.reference_and_mask()
    x = jnp.array(reference)
    linear = imagenet.PROBLEM.constraint_for("ldf")
    reduced = imagenet.PROBLEM.constraint_for("pigdm")
    assert linear.fn(x).shape == (int(observed.sum()) * 3,)
    assert reduced.fn(x).shape == (1,)
    changed = x.at[0, 0, 0].add(.2).at[0, 0, 1].add(-.2)
    np.testing.assert_allclose(reduced.fn(changed)[0],
                               jnp.linalg.norm(linear.fn(changed)))


def test_sweep_only_offers_supported_image_methods():
    from cfm import sweep

    cases = sweep.build_cases({
        "problems": ["imagenet"],
        "methods": ["ldf", "pigdm", "pcfm", "cbf"], "steps": [100],
    })
    assert [case.method for case in cases] == ["ldf", "pigdm", "pcfm"]
    assert len({case.key for case in cases}) == 3


def test_pretrained_training_and_unsupported_methods_fail_early():
    with pytest.raises(SystemExit, match="prepare_imagenet"):
        cli.main(["train", "--problem", "imagenet"])
    with pytest.raises(ValueError, match="supports: ldf, pigdm, pcfm"):
        cli._resolve(imagenet.PROBLEM, methods.CBF, SimpleNamespace())


def test_cli_exports_raw_samples_and_figure(tmp_path, monkeypatch):
    reference, _ = imagenet.reference_and_mask()
    # Out-of-range values must survive export; clipping is for PNG only.
    raw = jnp.full((1, *reference.shape), 1.2)
    monkeypatch.setattr(cli.checkpoint, "load", lambda _: (
        SimpleNamespace(data_shape=imagenet.IMAGE_SHAPE), Normalizer(.5, 2.),
    ))
    monkeypatch.setattr(
        methods.Method, "run",
        lambda *a, **kw: ldf.Samples(raw, raw[None]),
    )
    cli.main([
        "generate", "--problem", "imagenet", "--method", "ldf",
        "--no-plot", "--output-dir", str(tmp_path),
    ])
    np.testing.assert_array_equal(np.load(tmp_path / "samples.npy"), raw)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "grid.png").exists()


def test_cli_uses_pretrained_unconstrained_step_size(monkeypatch):
    calls = []
    monkeypatch.setattr(cli.checkpoint, "load", lambda _: (None, None))

    def generate(*args, **kwargs):
        calls.append(kwargs)
        return ldf.Samples(jnp.zeros((1, 2)), jnp.zeros((1, 1, 2)))

    monkeypatch.setattr(cli, "generate_unconstrained", generate)
    args = ["generate", "--problem", "imagenet", "--no-plot"]
    cli.main(args)
    cli.main([*args, "--dt", "0.003"])
    assert calls[0]["dt"] == imagenet.PROBLEM.unconstrained_gains["dt"]
    assert calls[1]["dt"] == .003


def test_missing_weights_explain_preparation(monkeypatch):
    def missing(_):
        raise FileNotFoundError("missing weights")

    monkeypatch.setattr(cli.checkpoint, "load", missing)
    with pytest.raises(SystemExit, match="prepare_imagenet --download"):
        cli.main(["generate", "--problem", "imagenet", "--no-plot"])


@pytest.mark.parametrize("std", [.5, 1., 2.])
@pytest.mark.parametrize("eps_reg", [1e-10, .03])
def test_pcfm_norm_preserves_projection_and_relaxed_correction(std, eps_reg):
    class Model:
        data_shape = (2, 2, 1)

        def __call__(self, x, t):
            return -.3 * x + .1 * jnp.sin(x.mean(
                axis=(1, 2, 3), keepdims=True,
            ))

    reference = np.array([.2, 0., .6, 0.]).reshape(2, 2, 1)
    mask = np.array([[True, False], [True, False]])
    kwargs = dict(num_samples=2, num_steps=5, num_projection_iters=2,
                  num_final_projection_iters=0, num_correction_iters=2,
                  eps_reg=eps_reg)
    outputs = [methods.PCFM.run(
        Model(), Normalizer(.5, std),
        imagenet.fixed_pixel_constraint(reference, mask, reduction=reduction),
        **kwargs,
    ) for reduction in ("pixels", "norm")]
    np.testing.assert_allclose(outputs[0].x, outputs[1].x, atol=2e-6)
    np.testing.assert_allclose(outputs[0].xs, outputs[1].xs, atol=2e-6)
