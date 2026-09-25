"""Flow-matching training."""

from datetime import datetime
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from torch.utils.data import DataLoader, Dataset, default_collate

from cfm.models.normalizer import Normalizer

SCHEDULES = ("constant", "cosine")

# Cosine schedule: fraction of steps spent in warmup, and final LR / peak LR.
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.05


def loss_fn(
    model: nnx.Module, x1: jax.Array, x0: jax.Array, t: jax.Array
) -> jax.Array:
    """Flow-matching loss ||v(xt, t) - (x1 - x0)||^2, xt = t x1 + (1 - t) x0."""
    t_bc = t.reshape((t.shape[0],) + (1,) * (x1.ndim - 1))
    xt = t_bc * x1 + (1 - t_bc) * x0
    target = x1 - x0
    pred = model(xt, t)
    return jnp.mean(jnp.square(pred - target))


def make_schedule(
    learning_rate: float, num_steps: int, schedule: str
) -> optax.Schedule:
    """Constant LR, or ``"cosine"``: linear warmup then cosine decay."""
    if schedule == "constant":
        return optax.constant_schedule(learning_rate)
    if schedule != "cosine":
        raise ValueError(
            f"unknown schedule {schedule!r}; expected one of {SCHEDULES}"
        )
    warmup = max(1, int(WARMUP_FRACTION * num_steps))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup,
        decay_steps=max(num_steps, warmup + 1),
        end_value=FINAL_LR_FRACTION * learning_rate,
    )


@jax.jit
def ema_update(averaged, params, decay: float):
    """One EMA step over a parameter pytree."""
    return jax.tree.map(
        lambda a, p: decay * a + (1.0 - decay) * p, averaged, params
    )


@nnx.jit
def train_step(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    batch: jax.Array,
    rng: jax.Array,
) -> jax.Array:
    """One in-place optimizer step; returns the batch loss."""
    batch_size = batch.shape[0]
    x1 = batch

    noise_rng, t_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, x1.shape)
    t = jax.random.uniform(t_rng, (batch_size,))
    loss, grad = nnx.value_and_grad(loss_fn)(model, x1, x0, t)
    optimizer.update(model, grad)

    return loss


def train(
    dataset: Dataset,
    model: nnx.Module,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int = 0,
    print_frequency: int = 1,
    schedule: str = "constant",
    ema_decay: Optional[float] = None,
) -> Tuple[nnx.Module, Normalizer]:
    """Train a flow model on normalized data.

    Args:
        learning_rate: peak learning rate when a schedule is used.
        print_frequency: epochs between progress prints.
        schedule: one of ``SCHEDULES``.
        ema_decay: if set (e.g. 0.999), return EMA weights instead of the
            last iterate.

    Returns:
        The trained model and the normalizer it expects inputs through.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: jax.tree.map(
            jnp.array, default_collate(batch)
        ),
    )

    batch = next(iter(dataloader))
    assert isinstance(batch, jax.Array), "Batch should be a jax.Array"
    assert batch.shape[0] == batch_size, (
        f"Batch size {batch.shape[0]} does not match expected {batch_size}"
    )

    learning_rate = make_schedule(
        learning_rate, num_epochs * len(dataloader), schedule
    )
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
    rng = jax.random.key(seed)

    averaged = (
        nnx.state(model, nnx.Param) if ema_decay is not None else None
    )

    normalizer = Normalizer.from_dataloader(dataloader)
    jit_normalize = jax.jit(normalizer.normalize)

    start_time = datetime.now()
    for epoch in range(num_epochs):
        loss = 0.0

        for batch in dataloader:
            rng, step_rng = jax.random.split(rng)
            batch = jit_normalize(batch)
            batch_loss = train_step(model, optimizer, batch, step_rng)
            loss += batch_loss

            if averaged is not None:
                averaged = ema_update(
                    averaged, nnx.state(model, nnx.Param), ema_decay
                )

        if (epoch + 1) % print_frequency == 0 or epoch == 0:
            loss = loss / len(dataloader)
            elapsed = datetime.now() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs}"
                f" | Loss {loss:.4f}"
                f" | Time {elapsed}"
            )

    if averaged is not None:
        nnx.update(model, averaged)

    return model, normalizer
