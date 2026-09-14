"""Flow-matching training.

The loop is deliberately plain. Two options exist because the locomotion
examples need them and a constant learning rate with the final weights was
not enough there: a cosine learning-rate schedule, and an exponential moving
average of the parameters. Both default to off, so every other problem
trains exactly as before.
"""

from datetime import datetime
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from torch.utils.data import DataLoader, Dataset, default_collate

from cfm.models.normalizer import Normalizer

SCHEDULES = ("constant", "cosine")

# Fraction of training spent warming the learning rate up from zero, and the
# fraction of the peak it decays to. Standard values; the cosine schedule is
# insensitive to both.
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.05


def loss_fn(
    model: nnx.Module, x1: jax.Array, x0: jax.Array, t: jax.Array
) -> jax.Array:
    """Compute the flow-maching loss for a given batch of data.

    The flow-matching loss is given by

        L = || v(xt, t) - (x1 - x0) ||^2

    where v(xt, t) is the model's prediction and

        xt = t * x1 + (1 - t) * x0.

    Args:
        model: The flow model xdot = v(x, t) to train.
        x1: The target data points, size (batch, data_size).
        x0: The initial noise, size (batch, data_size).
        t: The denoising time step, in [0, 1], size (batch,).

    Returns:
        The flow-matching loss L.
    """
    t_bc = t.reshape((t.shape[0],) + (1,) * (x1.ndim - 1))
    xt = t_bc * x1 + (1 - t_bc) * x0
    target = x1 - x0
    pred = model(xt, t)
    return jnp.mean(jnp.square(pred - target))


def make_schedule(
    learning_rate: float, num_steps: int, schedule: str
) -> optax.Schedule:
    """Build the learning-rate schedule.

    Args:
        learning_rate: Peak learning rate.
        num_steps: Total number of optimizer steps the run will take.
        schedule: ``"constant"``, or ``"cosine"`` for a warmup followed by
            cosine decay to a small fraction of the peak.

    Returns:
        An optax schedule mapping step count to learning rate.
    """
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
    """Fold one step's parameters into the running average."""
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
    """Perform a single optimization step on a batch of data.

    Args:
        model: The flow model xdot = v(x, t) to train.
        optimizer: The optimizer used to update the model parameters.
        batch: A batch of samples from the target distribution,
               size (batch, data_size).
        rng: A random key for sampling noise and time steps.

    Returns:
        The flow-matching loss for this batch.
    """
    batch_size = batch.shape[0]
    x1 = batch  # Target data points

    # Sample random noise x0 and time steps t
    noise_rng, t_rng = jax.random.split(rng)
    x0 = jax.random.normal(noise_rng, x1.shape)
    t = jax.random.uniform(t_rng, (batch_size,))

    # Compute loss and gradients
    loss, grad = nnx.value_and_grad(loss_fn)(model, x1, x0, t)

    # Optimization step. Model and optimizer parameters are updated in-place.
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
    """Train a simple flow-matching policy on the given dataset.

    Args:
        dataset: A PyTorch Dataset providing samples from the data distribution.
        model: The flow model xdot = v(x, t) to train.
        num_epochs: The number of training epochs.
        batch_size: The size of each training batch.
        learning_rate: The learning rate for the optimizer, or the peak rate
            when a schedule is used.
        seed: A random seed for reproducibility.
        print_frequency: How often to print training progress (in epochs).
        schedule: Learning-rate schedule, one of ``SCHEDULES``.
        ema_decay: Decay of an exponential moving average over the
            parameters, e.g. 0.999. The averaged weights are what is
            returned. None keeps the last iterate, which is noisier: SGD is
            still bouncing around the minimum at the final step, and on a
            flow model that noise shows up directly in the samples.

    Returns:
        The trained flow model v(x, t).
        The normalizer used to pre-process data passed to the flow model.
    """
    # Create a dataloader that automatically shuffles the data and provides
    # batches of jax arrays.
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: jax.tree.map(
            jnp.array, default_collate(batch)
        ),
    )

    # Check that the dataloader produces batches of the correct shape and type
    batch = next(iter(dataloader))
    assert isinstance(batch, jax.Array), "Batch should be a jax.Array"
    assert batch.shape[0] == batch_size, (
        "Batch size {batch.shape[0]} does not match expected {batch_size}"
    )

    learning_rate = make_schedule(
        learning_rate, num_epochs * len(dataloader), schedule
    )
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate), wrt=nnx.Param)
    rng = jax.random.key(seed)

    # The moving average starts at the initial weights. Early on it therefore
    # lags badly, but with thousands of steps to go that bias is long gone by
    # the time training ends.
    averaged = (
        nnx.state(model, nnx.Param) if ema_decay is not None else None
    )

    # Compute normalizer stats from the full dataset before training
    normalizer = Normalizer.from_dataloader(dataloader)
    jit_normalize = jax.jit(normalizer.normalize)

    # Training loop: optimizer and model parameters are updated in-place.
    start_time = datetime.now()
    for epoch in range(num_epochs):
        loss = 0.0

        for batch in dataloader:
            rng, step_rng = jax.random.split(rng)

            # Normalize the batch using pre-computed stats.
            batch = jit_normalize(batch)

            # Perform a SGD step, updating model parameters in-place.
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
