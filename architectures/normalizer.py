from flax import nnx
import jax
import jax.numpy as jnp

class Normalizer(nnx.Module):
    """A simple data normalizer.

    Scales input data to have zero mean and unit variance, based on statistics
    accumulated from the training data.
    """
    def __init__(self, data_size: int, rngs: nnx.Rngs):
        """Create a normalizer for data of the given dimension."""
        self.batch_norm = nnx.BatchNorm(
            data_size, use_scale=False, use_bias=False, momentum=0.9, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Normalize the input data x."""
        return self.batch_norm(x)

    def unnormalize(self, x: jax.Array) -> jax.Array:
        """Un-normalize the input data x."""
        mean = self.batch_norm.mean
        var = self.batch_norm.var
        std = jnp.sqrt(var + self.batch_norm.epsilon)
        return x * std + mean
