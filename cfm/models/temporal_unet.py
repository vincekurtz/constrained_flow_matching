"""Temporal U-Net for trajectory windows, as in Diffuser (arXiv:2205.09991).

1-D convolutions over the horizon, with transition entries as channels.
"""

import math
from typing import Tuple

import jax
import jax.numpy as jnp
from flax import nnx

from cfm.models.flow import SinusoidalPosEmb
from cfm.models.unet import group_count


class TemporalResBlock(nnx.Module):
    """1-D analogue of ``unet.ResBlock``."""

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
        h = self.conv1(self.act(self.norm1(x)))

        t_proj = self.time_proj(self.act(t_emb))[:, None, :]
        gamma, beta = jnp.split(t_proj, 2, axis=-1)
        h = self.norm2(h) * (1 + gamma) + beta
        h = self.conv2(self.act(h))

        return self.skip(x) + h


class FlowTemporalUNet(nnx.Module):
    """Vector field xdot = v(x, t) over trajectory windows.

    Args:
        data_shape: (horizon, transition_dim); horizon divisible by
            2**(len(channels) - 1).
        channels: channel count per resolution level, e.g. (64, 128, 256).
        kernel_size: temporal convolution width.
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

        time_dim = time_embedding_size * 4
        self.time_embedding = nnx.Sequential(
            SinusoidalPosEmb(time_embedding_size),
            nnx.Linear(time_embedding_size, time_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(time_dim, time_dim, rngs=rngs),
        )

        self.input_conv = nnx.Conv(
            in_channels,
            channels[0],
            kernel_size=(kernel_size,),
            padding="SAME",
            rngs=rngs,
        )

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

        self.mid_block1 = TemporalResBlock(
            ch, ch, time_dim, kernel_size, rngs=rngs
        )
        self.mid_block2 = TemporalResBlock(
            ch, ch, time_dim, kernel_size, rngs=rngs
        )

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

        # Zero-init output: the target x1 - x0 has zero mean.
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
        t_emb = self.time_embedding(t)

        h = self.input_conv(x)

        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            skips.append(h)
            h = block(h, t_emb)
            h = down(h)

        h = self.mid_block2(self.mid_block1(h, t_emb), t_emb)

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
        """Total number of trainable parameters."""
        return sum(
            math.prod(p.shape)
            for p in jax.tree.leaves(nnx.state(self, nnx.Param))
        )
