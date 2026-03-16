from typing import Tuple

import flax.linen as nn
import jax


class MLP(nn.Module):
    """A basic multi-layer perceptron (MLP) network."""

    output_size: int
    hidden_sizes: Tuple[int, ...]

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Run a forward pass through the network.

        Args:
            x: The input to the network.

        Returns:
            The output of the network.
        """
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.swish(x)
        return nn.Dense(self.output_size)(x)
