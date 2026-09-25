import math

from flax import nnx
import jax
import jax.numpy as jnp
from typing import Tuple

from cfm.models.mlp import MLP

class SinusoidalPosEmb(nnx.Module):
    """Sinusoidal embedding of t in [0, 1], log-spaced frequencies 1-1000 Hz."""

    def __init__(self, dim: int):
        assert dim > 2, "Positional embedding dimension must be greater than 2"
        assert dim % 2 == 0, "Positional embedding dimension must be even"
        max_frequency = 1000
        half_dim = dim // 2
        exponent = jnp.arange(half_dim) / (half_dim - 1)
        self.freqs = jnp.power(max_frequency, exponent)

    def __call__(self, x: jax.Array) -> jax.Array:
        emb = 2 * jnp.pi * x[:, None] * self.freqs[None, :]
        emb = jnp.concatenate((jnp.sin(emb), jnp.cos(emb)), axis=-1)
        return emb


class FlowMLP(nnx.Module):
    """MLP vector field xdot = v(x, t) on samples of shape ``data_shape``."""

    def __init__(
        self,
        data_shape: Tuple[int, ...],
        time_embedding_size: int,
        hidden_sizes: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
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
        batch = x.shape[0]
        x_flat = x.reshape(batch, -1)
        t_emb = self.time_embedding(t)
        xt = jnp.concatenate((x_flat, t_emb), axis=-1)
        xdot_flat = self.mlp(xt)
        xdot = xdot_flat.reshape(x.shape)
        return xdot
