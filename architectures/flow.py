from flax import nnx
import jax
import jax.numpy as jnp
from typing import Tuple

from architectures.mlp import MLP

class SinusoidalPosEmb(nnx.Module):
    """A basic positional embedding."""

    def __init__(self, dim: int):
        """Create a position embedding of the given dimension."""
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
        data_size: int,
        time_embedding_size: int,
        hidden_sizes: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        """Create a flow MLP with the given dimensions.

        Args:
            data_size: The dimension of the input data x.
            time_embedding_size: The dimension of the time embedding.
            hidden_sizes: A tuple specifying the size of each hidden layer.
            rngs: Random keys for weight initialization.
        """
        self.time_embedding = SinusoidalPosEmb(time_embedding_size)

        input_size = data_size + time_embedding_size
        output_size = data_size

        self.mlp = MLP(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        """Run a forward pass through the network.

        Args:
            x: The input to the network.
            t: The denoising time step, in [0, 1].

        Returns:
            The output of the network xdot = v(x, t).
        """
        t_emb = self.time_embedding(t)
        xt = jnp.concatenate((x, t_emb), axis=-1)
        return self.mlp(xt)
