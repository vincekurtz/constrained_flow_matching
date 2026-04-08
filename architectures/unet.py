from typing import Tuple

from flax import nnx
import jax
import jax.numpy as jnp

from architectures.flow import SinusoidalPosEmb


def _num_groups(channels: int) -> int:
    """Return a group count for GroupNorm that evenly divides channels."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ResBlock(nnx.Module):
    """Residual convolution block with time conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.norm1 = nnx.GroupNorm(
            in_channels, num_groups=_num_groups(in_channels), rngs=rngs,
        )
        self.conv1 = nnx.Conv(
            in_channels, out_channels,
            kernel_size=(3, 3), padding="SAME", rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(
            out_channels, num_groups=_num_groups(out_channels), rngs=rngs,
        )
        self.conv2 = nnx.Conv(
            out_channels, out_channels,
            kernel_size=(3, 3), padding="SAME", rngs=rngs,
        )
        self.time_proj = nnx.Linear(time_dim, out_channels, rngs=rngs)
        self.skip = (
            nnx.Conv(in_channels, out_channels, kernel_size=(1, 1), rngs=rngs)
            if in_channels != out_channels
            else None
        )

    def __call__(self, x: jax.Array, t_emb: jax.Array) -> jax.Array:
        h = nnx.swish(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(nnx.swish(t_emb))[:, None, None, :]
        h = nnx.swish(self.norm2(h))
        h = self.conv2(h)
        residual = self.skip(x) if self.skip is not None else x
        return residual + h


class FlowUNet(nnx.Module):
    """A simple U-Net vector field xdot = v(x, t) for image data.

    Uses an encoder-decoder structure with skip connections and sinusoidal
    time conditioning, suitable for flow-matching on images with any number
    of channels.
    """

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        time_embedding_size: int,
        channels: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        """Create a U-Net flow model.

        Args:
            data_shape: Shape of a single image ``(H, W, C)``.  Spatial
                dimensions must be divisible by ``2 ** (len(channels) - 1)``.
            time_embedding_size: Dimension of the sinusoidal time embedding.
            channels: Channel counts at each resolution level,
                e.g. ``(64, 128, 256)``.  The number of downsampling /
                upsampling steps equals ``len(channels) - 1``.
            rngs: Random keys for weight initialization.
        """
        assert len(data_shape) == 3, "data_shape must be (H, W, C)"
        self.data_shape = data_shape
        in_channels = data_shape[-1]

        # Time embedding
        self.time_embedding = SinusoidalPosEmb(time_embedding_size)
        time_dim = time_embedding_size * 4
        self.time_dense1 = nnx.Linear(
            time_embedding_size, time_dim, rngs=rngs
        )
        self.time_dense2 = nnx.Linear(time_dim, time_dim, rngs=rngs)

        # Input projection
        self.input_conv = nnx.Conv(
            in_channels, channels[0],
            kernel_size=(3, 3), padding="SAME", rngs=rngs,
        )

        # Encoder: each level has a ResBlock followed by stride-2 downsample
        self.down_blocks = nnx.List()
        self.downsamples = nnx.List()
        ch = channels[0]
        for ch_next in channels[1:]:
            self.down_blocks.append(
                ResBlock(ch, ch_next, time_dim, rngs=rngs)
            )
            self.downsamples.append(
                nnx.Conv(
                    ch_next, ch_next,
                    kernel_size=(3, 3), strides=(2, 2),
                    padding="SAME", rngs=rngs,
                )
            )
            ch = ch_next

        # Bottleneck
        self.mid_block = ResBlock(ch, ch, time_dim, rngs=rngs)

        # Decoder: upsample, concatenate skip, then ResBlock
        self.upsamples = nnx.List()
        self.up_blocks = nnx.List()
        for ch_skip in reversed(channels[:-1]):
            self.upsamples.append(
                nnx.Conv(
                    ch, ch,
                    kernel_size=(3, 3), padding="SAME", rngs=rngs,
                )
            )
            self.up_blocks.append(
                ResBlock(ch + ch_skip, ch_skip, time_dim, rngs=rngs)
            )
            ch = ch_skip

        # Output projection
        self.output_norm = nnx.GroupNorm(
            channels[0], num_groups=_num_groups(channels[0]), rngs=rngs,
        )
        self.output_conv = nnx.Conv(
            channels[0], in_channels, kernel_size=(1, 1), rngs=rngs,
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Compute the vector field v(x, t).

        Args:
            x: Input images, shape ``(batch, H, W, C)``.
            t: Time steps in ``[0, 1]``, shape ``(batch,)``.

        Returns:
            Predicted velocity, same shape as ``x``.
        """
        # Time conditioning
        t_emb = nnx.swish(self.time_dense1(self.time_embedding(t)))
        t_emb = self.time_dense2(t_emb)

        h = self.input_conv(x)

        # Encoder — save skip features before each downsample
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            skips.append(h)
            h = block(h, t_emb)
            h = down(h)

        h = self.mid_block(h, t_emb)

        # Decoder
        for up_conv, block, skip in zip(
            self.upsamples, self.up_blocks, reversed(skips)
        ):
            h = jax.image.resize(
                h,
                (h.shape[0], skip.shape[1], skip.shape[2], h.shape[3]),
                method="nearest",
            )
            h = up_conv(h)
            h = jnp.concatenate([h, skip], axis=-1)
            h = block(h, t_emb)

        h = nnx.swish(self.output_norm(h))
        return self.output_conv(h)
