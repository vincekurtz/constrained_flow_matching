import math

from flax import nnx
import jax
import jax.numpy as jnp
from typing import Tuple

from architectures.mlp import MLP

class SinusoidalPosEmb(nnx.Module):
    """A basic positional embedding."""

    def __init__(self, dim: int):
        """Create a position embedding of the given dimension."""
        assert dim > 2, "Positional embedding dimension must be greater than 2"
        assert dim % 2 == 0, "Positional embedding dimension must be even"
        self.half_dim = dim // 2

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the positional embedding to the input x."""
        emb = jnp.log(10000) / (self.half_dim - 1)
        emb = jnp.exp(jnp.arange(self.half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class FlowMLP(nnx.Module):
    """A simple vector field xdot = v(x, t) based on a simple MLP backend."""

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        time_embedding_size: int,
        hidden_sizes: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        """Create a flow MLP with the given dimensions.

        Args:
            data_shape: The shape of a single data sample (excluding batch),
                e.g. (2,) for 2-D vectors or (28, 28, 1) for MNIST images.
            time_embedding_size: The dimension of the time embedding.
            hidden_sizes: A tuple specifying the size of each hidden layer.
            rngs: Random keys for weight initialization.
        """
        self.data_shape = data_shape
        self.time_embedding = SinusoidalPosEmb(time_embedding_size)

        flat_data_size = math.prod(data_shape)
        input_size = flat_data_size + time_embedding_size
        output_size = flat_data_size

        self.mlp = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Run a forward pass through the network.

        Args:
            x: The input to the network, shape ``(batch, *data_shape)``.
            t: The denoising time step, in [0, 1], shape ``(batch,)``.

        Returns:
            The output of the network xdot = v(x, t), same shape as ``x``.
        """
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        t_emb = self.time_embedding(t)
        xt = jnp.concatenate((x_flat, t_emb), axis=-1)
        xdot_flat = self.mlp(xt)
        xdot = xdot_flat.reshape(x.shape)
        return xdot
