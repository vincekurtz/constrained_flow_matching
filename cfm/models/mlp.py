from typing import Tuple

from flax import nnx
import jax

class MLP(nnx.Module):
    """Feed-forward network with swish activations."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List()
        in_size = input_size
        for size in hidden_sizes:
            self.layers.append(nnx.Linear(in_size, size, rngs=rngs))
            in_size = size
        self.layers.append(nnx.Linear(in_size, output_size, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.swish(x)
        return self.layers[-1](x)
