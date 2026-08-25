from typing import Tuple

from flax import nnx
import jax

class MLP(nnx.Module):
    """A basic multi-layer perceptron (MLP) network."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: Tuple[int, ...],
        *,
        rngs: nnx.Rngs,
    ):
        """Create a simple feed-forward network with the given dimensions.

        Args:
            input_size: The dimension of the input to the network.
            output_size: The dimension of the output of the network.
            hidden_sizes: A tuple specifying the size of each hidden layer.
            rngs: Random keys for weight initialization.
        """
        self.layers = nnx.List()
        in_size = input_size
        for i, size in enumerate(hidden_sizes):
            self.layers.append(nnx.Linear(in_size, size, rngs=rngs))
            in_size = size
        self.layers.append(nnx.Linear(in_size, output_size, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Run a forward pass through the network."""
        for layer in self.layers[:-1]:
            x = layer(x)
            x = nnx.swish(x)
        return self.layers[-1](x)
