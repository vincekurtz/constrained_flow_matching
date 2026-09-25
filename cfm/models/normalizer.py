from typing import Iterable

import jax
import jax.numpy as jnp


class Normalizer:
    """Per-element zero-mean, unit-variance scaling."""

    def __init__(self, mean: jax.Array, std: jax.Array):
        self.mean = mean
        self.std = std

    @classmethod
    def from_dataloader(cls, dataloader: Iterable[jax.Array]) -> "Normalizer":
        """Compute mean and std from running sums over batches."""
        count = 0
        running_sum = 0
        running_sum_sq = 0

        for batch in dataloader:
            batch_size = batch.shape[0]
            batch_sum = jnp.sum(batch, axis=0)
            batch_sum_sq = jnp.sum(batch**2, axis=0)

            running_sum += batch_sum
            running_sum_sq += batch_sum_sq
            count += batch_size

        mean = running_sum / count
        var = running_sum_sq / count - mean**2
        std = jnp.maximum(jnp.sqrt(var), 1e-6)

        return cls(mean, std)

    def normalize(self, x: jax.Array) -> jax.Array:
        return (x - self.mean) / self.std

    def unnormalize(self, x: jax.Array) -> jax.Array:
        return x * self.std + self.mean
