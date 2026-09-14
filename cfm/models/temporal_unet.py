"""A U-Net vector field for trajectory windows, convolving along time.

The locomotion examples generate a whole window at once: one sample is
``(horizon, transition_dim)``. Flattening that into an MLP, as ``FlowMLP``
does, throws away the fact that the horizon axis is *time* -- neighbouring
rows of a demonstration are nearly equal, and the model has to relearn that
correlation from scratch for every pair of columns. It never quite does, so
the samples come out jagged: on the trained Hopper MLP the torso-height trace
is roughly fifty times rougher, step to step, than the training data.

This model is the Diffuser architecture (https://arxiv.org/abs/2205.09991),
1-D convolutions over the horizon with the transition entries as channels. A
kernel of width 5 sees a whole neighbourhood in time, the two downsampling
levels give the deeper blocks a horizon-wide receptive field, and nothing in
the network can move a single timestep independently of its neighbours. The
smoothness of the data therefore comes for free rather than being learned.

Flax convolutions are channels-last, and a window is already laid out as
``(batch, horizon, transition_dim)``, so no transposition is needed anywhere.
"""

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from cfm.models.flow import SinusoidalPosEmb
from cfm.models.unet import group_count


class TemporalResBlock(nnx.Module):
    """Residual 1-D convolution block with time conditioning.

    The same shape as the image ``ResBlock``: two convolutions with adaptive
    group normalization (AdaGN, https://arxiv.org/pdf/2105.05233) carrying
    the denoising time, and a skip that projects channels when they change.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        kernel_size: int = 5,
        *,
        rngs: nnx.Rngs,
    ):
        self.time_proj = nnx.Linear(time_dim, 2 * out_channels, rngs=rngs)

        self.norm1 = nnx.GroupNorm(
            in_channels,
            num_groups=group_count(in_channels),
            rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_channels,
            out_channels,
            kernel_size=(kernel_size,),
            padding="SAME",
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            out_channels,
            num_groups=group_count(out_channels),
            rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels,
            out_channels,
            kernel_size=(kernel_size,),
            padding="SAME",
            rngs=rngs,
        )
        self.skip = (
            nnx.Conv(in_channels, out_channels, kernel_size=(1,), rngs=rngs)
            if in_channels != out_channels
            else lambda x: x
        )

        self.act = nnx.swish

    def __call__(self, x: jax.Array, t_emb: jax.Array) -> jax.Array:
        """Forward pass, on features of shape ``(batch, length, channels)``."""
        h = self.conv1(self.act(self.norm1(x)))

        # AdaGN conditioning: a scale and bias per channel, from the time
        # embedding, broadcast across the whole horizon.
        t_proj = self.time_proj(self.act(t_emb))[:, None, :]
        gamma, beta = jnp.split(t_proj, 2, axis=-1)
        h = self.norm2(h) * (1 + gamma) + beta

        h = self.conv2(self.act(h))

        return self.skip(x) + h


class FlowTemporalUNet(nnx.Module):
    """A vector field ``xdot = v(x, t)`` over trajectory windows.

    Encoder-decoder over the horizon axis with skip connections, sinusoidal
    time conditioning, and a zero-initialized output projection so that the
    field starts at zero and the first epochs only have to learn corrections.
    """

    def __init__(
        self,
        data_shape: Tuple[int, int],
        time_embedding_size: int,
        channels: Tuple[int, ...],
        kernel_size: int = 5,
        *,
        rngs: nnx.Rngs,
    ):
        """Create a temporal U-Net flow model.

        Args:
            data_shape: Shape of one window, ``(horizon, transition_dim)``.
                The horizon must be divisible by ``2 ** (len(channels) - 1)``.
            time_embedding_size: Dimension of the sinusoidal time embedding.
            channels: Channel counts at each resolution level, e.g.
                ``(64, 128, 256)``. The number of downsampling steps is
                ``len(channels) - 1``.
            kernel_size: Width of the temporal convolutions, in timesteps.
            rngs: Random keys for weight initialization.
        """
        assert len(data_shape) == 2, \
            "data_shape must be (horizon, transition_dim)"
        horizon = data_shape[0]
        stride = 2 ** (len(channels) - 1)
        assert horizon % stride == 0, (
            f"horizon {horizon} is not divisible by {stride}, the total "
            f"downsampling of {len(channels)} levels"
        )
        self.data_shape = data_shape
        in_channels = data_shape[-1]

        # Time embedding
        time_dim = time_embedding_size * 4
        self.time_embedding = nnx.Sequential(
            SinusoidalPosEmb(time_embedding_size),
            nnx.Linear(time_embedding_size, time_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(time_dim, time_dim, rngs=rngs),
        )

        # Input projection
        self.input_conv = nnx.Conv(
            in_channels,
            channels[0],
            kernel_size=(kernel_size,),
            padding="SAME",
            rngs=rngs,
        )

        # Encoder: a ResBlock per level, then a stride-2 downsample in time
        self.down_blocks = nnx.List()
        self.downsamples = nnx.List()
        ch = channels[0]
        for ch_next in channels[1:]:
            self.down_blocks.append(
                TemporalResBlock(
                    ch, ch_next, time_dim, kernel_size, rngs=rngs
                )
            )
            self.downsamples.append(
                nnx.Conv(
                    ch_next,
                    ch_next,
                    kernel_size=(3,),
                    strides=(2,),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            ch = ch_next

        # Bottleneck. Two blocks, as in Diffuser: this is where the receptive
        # field spans the whole window, so it is the cheapest place to spend
        # capacity on the shape of a stride.
        self.mid_block1 = TemporalResBlock(
            ch, ch, time_dim, kernel_size, rngs=rngs
        )
        self.mid_block2 = TemporalResBlock(
            ch, ch, time_dim, kernel_size, rngs=rngs
        )

        # Decoder: upsample, concatenate the skip, then a ResBlock
        self.upsamples = nnx.List()
        self.up_blocks = nnx.List()
        for ch_skip in reversed(channels[:-1]):
            self.upsamples.append(
                nnx.ConvTranspose(
                    ch,
                    ch,
                    kernel_size=(4,),
                    strides=(2,),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            self.up_blocks.append(
                TemporalResBlock(
                    ch + ch_skip, ch_skip, time_dim, kernel_size, rngs=rngs
                )
            )
            ch = ch_skip

        # Output projection, initialized to zero. A flow model's target,
        # x1 - x0, has zero mean, so an identically zero field is a better
        # starting point than a random one and avoids a large transient in
        # the first few hundred steps.
        self.output_norm = nnx.GroupNorm(
            channels[0],
            num_groups=group_count(channels[0]),
            rngs=rngs,
        )
        self.output_conv = nnx.Conv(
            channels[0],
            in_channels,
            kernel_size=(1,),
            kernel_init=nnx.initializers.zeros_init(),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Compute the vector field ``v(x, t)``.

        Args:
            x: Trajectory windows, shape ``(batch, horizon, transition_dim)``.
            t: Denoising times in ``[0, 1]``, shape ``(batch,)``.

        Returns:
            Predicted velocity, same shape as ``x``.
        """
        t_emb = self.time_embedding(t)

        h = self.input_conv(x)

        # Encoder -- save skip features before each downsample
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            skips.append(h)
            h = block(h, t_emb)
            h = down(h)

        h = self.mid_block2(self.mid_block1(h, t_emb), t_emb)

        # Decoder
        for up_conv, block, skip in zip(
            self.upsamples, self.up_blocks, reversed(skips)
        ):
            h = up_conv(h)
            h = jnp.concatenate([h, skip], axis=-1)
            h = block(h, t_emb)

        h = nnx.swish(self.output_norm(h))
        return self.output_conv(h)

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters, for reporting."""
        return sum(
            math.prod(p.shape)
            for p in jax.tree.leaves(nnx.state(self, nnx.Param))
        )
