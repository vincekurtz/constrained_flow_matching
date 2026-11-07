"""The LDF model wrapper preserves the native flow and fixed RGB equalities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfm.models.image_coordinates import (
    ImageCoordinateFlow, ImageCoordinateNormalizer,
)
from cfm.models.normalizer import Normalizer
from problems import imagenet


class ToyImageFlow:
    data_shape = (2, 2, 1)

    def __call__(self, x, t):
        return -.2 * x + .1 * jnp.sin(x.mean(axis=(1, 2, 3), keepdims=True))


def test_native_prior_chain_rule_and_endpoint():
    base = ToyImageFlow()
    model = ImageCoordinateFlow(base)
    norm = ImageCoordinateNormalizer(Normalizer(.5, 2.))
    a, b = model.initial_image_scale, model.noise_scale
    state = jax.random.normal(jax.random.key(4), (3, *model.data_shape))
    t = jnp.array([0., .4, 1.])
    u, w = state[:, 0], state[:, 1]
    x, e = a * u - b * w, b * u + a * w
    tt = t[:, None, None, None]
    h = a + (1 - a) * tt
    z = h * x + b * (1 - tt) * e
    velocity = model(state, t)
    vx = a * velocity[:, 0] - b * velocity[:, 1]
    ve = b * velocity[:, 0] + a * velocity[:, 1]
    np.testing.assert_allclose(ve, 0, atol=3e-7)
    np.testing.assert_allclose(h * vx + (1 - a) * x - b * e,
                               base(z, t), atol=5e-7)
    np.testing.assert_allclose(z[0], u[0], atol=3e-7)
    np.testing.assert_allclose(norm.unnormalize(state)[-1],
                               norm.normalizer.unnormalize(z[-1]), atol=3e-7)
    rotation = np.array([[a, -b], [b, a]])
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(2), atol=1e-15)
    # The comparison generates one seed at a time. Its native initial
    # Gaussian image matches the unaugmented sampler, not a new random draw.
    key = jax.random.key(2)
    np.testing.assert_array_equal(
        jax.random.normal(key, (1, *base.data_shape)),
        jax.random.normal(key, (1, *model.data_shape))[:, 0],
    )


def test_fixed_pixel_gradients_leave_auxiliary_noise_unchanged():
    model = ImageCoordinateFlow(ToyImageFlow())
    norm = ImageCoordinateNormalizer(Normalizer(.5, 2.))
    reference = np.array([.2, 0., .6, 0.], dtype=np.float32).reshape(2, 2, 1)
    observed = np.array([[True, False], [True, False]])
    constraint = imagenet.fixed_pixel_constraint(reference, observed)
    state = jax.random.normal(jax.random.key(0), model.data_shape)
    jac = jax.jacobian(lambda s: constraint.fn(norm.unnormalize(s)))(state)
    a, b = model.initial_image_scale, model.noise_scale
    # Every multiplier and the penalty term act orthogonally to e.
    np.testing.assert_allclose(b * jac[:, 0] + a * jac[:, 1], 0, atol=2e-7)
    flat = jac.reshape(2, -1)
    np.testing.assert_allclose(flat @ flat.T, 4 * np.eye(2), atol=1e-6)
    assert constraint.fn(norm.unnormalize(state)).shape == (2,)


def test_normalizer_handles_images_batches_and_trajectories():
    norm = ImageCoordinateNormalizer(Normalizer(.5, 2.))
    for shape in ((2, 2, 1), (3, 2, 2, 1), (4, 3, 2, 2, 1)):
        image = jax.random.normal(jax.random.key(1), shape)
        np.testing.assert_allclose(norm.unnormalize(norm.normalize(image)),
                                   image, atol=5e-7)


@pytest.mark.parametrize("scale", [0., -1., 1.1, float("nan")])
def test_invalid_coordinates_are_rejected(scale):
    with pytest.raises(ValueError, match="initial_image_scale"):
        ImageCoordinateFlow(ToyImageFlow(), scale)
    with pytest.raises(ValueError, match="initial_image_scale"):
        ImageCoordinateNormalizer(Normalizer(.5, 2.), scale)


def test_example_uses_wrapper_only_for_ldf():
    base, norm = ToyImageFlow(), Normalizer(.5, 2.)
    for method in (None, "unconstrained", "pigdm"):
        assert imagenet.PROBLEM.model_for(method, base, norm) == (base, norm)
    model, image_norm = imagenet.PROBLEM.model_for("ldf", base, norm)
    assert isinstance(model, ImageCoordinateFlow) and model.model is base
    assert isinstance(image_norm, ImageCoordinateNormalizer)
    assert image_norm.normalizer is norm
