from typing import Iterable

import jax
import jax.numpy as jnp


class Normalizer:
    """A simple data normalizer.

    Scales input data to have zero mean and unit variance, based on statistics
    computed from the training data. Supports data of any shape.
    """

    def __init__(self, mean: jax.Array, std: jax.Array):
        """Create a normalizer with pre-computed statistics.

        Args:
            mean: The mean of the data, shape matching a single data point.
            std: The standard deviation of the data, same shape as mean.
        """
        self.mean = mean
        self.std = std

    @classmethod
    def from_dataloader(cls, dataloader: Iterable[jax.Array]) -> "Normalizer":
        """Compute mean and std by iterating over batches from a dataloader.

        Uses an online algorithm so the full dataset need not fit in memory.

        Args:
            dataloader: An iterable yielding batches of shape (batch, ...).

        Returns:
            A Normalizer with the computed mean and std.
        """
        count = 0
        running_sum = None
        running_sum_sq = None

        for batch in dataloader:
            batch_size = batch.shape[0]
            batch_sum = jnp.sum(batch, axis=0)
            batch_sum_sq = jnp.sum(batch**2, axis=0)

            if running_sum is None:
                running_sum = batch_sum
                running_sum_sq = batch_sum_sq
            else:
                running_sum = running_sum + batch_sum
                running_sum_sq = running_sum_sq + batch_sum_sq

            count += batch_size

        mean = running_sum / count
        var = running_sum_sq / count - mean**2
        std = jnp.maximum(jnp.sqrt(var), 1e-6)

        return cls(mean, std)

    def __call__(self, x: jax.Array) -> jax.Array:
        """Normalize the input data x."""
        return (x - self.mean) / self.std

    def unnormalize(self, x: jax.Array) -> jax.Array:
        """Un-normalize the input data x."""
        return x * self.std + self.mean
