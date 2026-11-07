"""Reference-independent image coordinates for an augmented Gaussian flow.

These coordinates are useful for LDF's constraints on the current image.
PiGDM should use the original model: its clean-estimate formula is not
invariant under this transformation.
"""

import math

import jax.numpy as jnp


class ImageCoordinateFlow:
    """Express a pretrained flow in image and retained-noise coordinates.

    Let a=initial_image_scale, b=sqrt(1-a²), h=a+(1-a)t. Rotate two
    independent standard Gaussian images (u,w) into x=a*u-b*w, e=b*u+a*w.
    The backbone sees z=h*x+b*(1-t)*e, with

        xdot = (v(z,t) - (1-a)*x + b*e) / h,  edot = 0.

    Thus zdot=v(z,t), z(0)=u, and z(1)=x(1): the native Gaussian prior,
    flow, and image endpoint are preserved. The positive a avoids a
    singular coordinate transform at t=0. Neither references nor masks
    enter this model. Fixed-pixel constraints act on x; their gradients
    are orthogonal to e, so LDF retains noise in the backbone's inputs.
    """

    def __init__(self, model, initial_image_scale=0.02):
        if (not math.isfinite(initial_image_scale)
                or not 0 < initial_image_scale <= 1):
            raise ValueError("initial_image_scale must be in (0, 1]")
        if len(model.data_shape) != 3:
            raise ValueError("Expected an HWC image flow")
        self.model = model
        self.initial_image_scale = initial_image_scale
        self.noise_scale = math.sqrt(1 - initial_image_scale**2)
        self.image_shape = model.data_shape
        self.data_shape = (2, *self.image_shape)

    def __call__(self, state, t):
        a, b = self.initial_image_scale, self.noise_scale
        u, w = state[:, 0], state[:, 1]
        x, e = a * u - b * w, b * u + a * w
        t_image = t[:, None, None, None]
        h = a + (1 - a) * t_image
        z = h * x + b * (1 - t_image) * e
        vx = (self.model(z, t) - (1 - a) * x + b * e) / h
        return jnp.stack([a * vx, -b * vx], axis=1)


class ImageCoordinateNormalizer:
    """Decode the image coordinate to RGB, including batched trajectories.

    ``normalize`` is a right inverse with zero auxiliary noise, not an
    inverse on the augmented state. This adapter is inference-only.
    """

    def __init__(self, normalizer, initial_image_scale=0.02):
        if (not math.isfinite(initial_image_scale)
                or not 0 < initial_image_scale <= 1):
            raise ValueError("initial_image_scale must be in (0, 1]")
        self.normalizer = normalizer
        self.initial_image_scale = initial_image_scale
        self.noise_scale = math.sqrt(1 - initial_image_scale**2)

    def unnormalize(self, state):
        a, b = self.initial_image_scale, self.noise_scale
        x = a * state[..., 0, :, :, :] - b * state[..., 1, :, :, :]
        return self.normalizer.unnormalize(x)

    def normalize(self, image):
        x = self.normalizer.normalize(image)
        return jnp.stack(
            [self.initial_image_scale * x, -self.noise_scale * x], axis=-4
        )
