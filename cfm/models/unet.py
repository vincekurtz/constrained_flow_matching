from typing import Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from cfm.models.flow import SinusoidalPosEmb


def group_count(channels: int) -> int:
    """Largest GroupNorm group count in (8, 4, 2, 1) that divides channels."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nnx.Module):
    """Residual conv block with AdaGN time conditioning (arXiv:2105.05233)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
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
            kernel_size=(3, 3),
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
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )
        self.skip = (
            nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)
            if in_channels != out_channels
            else lambda x: x
        )

        self.act = nnx.swish

    def __call__(self, x: jax.Array, t_emb: jax.Array) -> jax.Array:
        h = self.conv1(self.act(self.norm1(x)))

        t_proj = self.time_proj(self.act(t_emb))[:, None, None, :]
        gamma, beta = jnp.split(t_proj, 2, axis=-1)
        h = self.norm2(h) * (1 + gamma) + beta
        h = self.conv2(self.act(h))

        return self.skip(x) + h


class FlowUNet(nnx.Module):
    """U-Net vector field xdot = v(x, t) for images.

    Args:
        data_shape: (H, W, C); H and W divisible by 2**(len(channels) - 1).
        channels: channel count per resolution level, e.g. (64, 128, 256).
    """

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        time_embedding_size: int,
        channels: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        assert len(data_shape) == 3, "data_shape must be (H, W, C)"
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
            kernel_size=(3, 3),
            padding="SAME",
            rngs=rngs,
        )

        self.down_blocks = nnx.List()
        self.downsamples = nnx.List()
        ch = channels[0]
        for ch_next in channels[1:]:
            self.down_blocks.append(ResBlock(ch, ch_next, time_dim, rngs=rngs))
            self.downsamples.append(
                nnx.Conv(
                    ch_next,
                    ch_next,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            ch = ch_next

        self.mid_block = ResBlock(ch, ch, time_dim, rngs=rngs)

        self.upsamples = nnx.List()
        self.up_blocks = nnx.List()
        for ch_skip in reversed(channels[:-1]):
            self.upsamples.append(
                nnx.ConvTranspose(
                    ch,
                    ch,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="SAME",
                    rngs=rngs,
                )
            )
            self.up_blocks.append(
                ResBlock(ch + ch_skip, ch_skip, time_dim, rngs=rngs)
            )
            ch = ch_skip

        self.output_norm = nnx.GroupNorm(
            channels[0],
            num_groups=group_count(channels[0]),
            rngs=rngs,
        )
        self.output_conv = nnx.Conv(
            channels[0],
            in_channels,
            kernel_size=(1, 1),
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

        h = self.mid_block(h, t_emb)

        for up_conv, block, skip in zip(
            self.upsamples, self.up_blocks, reversed(skips)
        ):
            h = up_conv(h)
            h = jnp.concatenate([h, skip], axis=-1)
            h = block(h, t_emb)

        h = nnx.swish(self.output_norm(h))
        return self.output_conv(h)
