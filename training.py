from flax import nnx
from typing import Tuple
import jax.numpy as jnp
import jax
import optax
from torch.utils.data import Dataset, DataLoader, default_collate
from architectures.normalizer import Normalizer
from datetime import datetime


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
) -> Tuple[nnx.Module, Normalizer]:
    """Train a simple flow-matching policy on the given dataset.

    Args:
        dataset: A PyTorch Dataset providing samples from the data distribution.
        model: The flow model xdot = v(x, t) to train.
        num_epochs: The number of training epochs.
        batch_size: The size of each training batch.
        learning_rate: The learning rate for the optimizer.
        seed: A random seed for reproducibility.
        print_frequency: How often to print training progress (in epochs).

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

    optimizer = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)
    rng = jax.random.key(seed)

    # Compute normalizer stats from the full dataset before training
    normalizer = Normalizer.from_dataloader(dataloader)

    # Training loop: optimizer and model parameters are updated in-place.
    start_time = datetime.now()
    for epoch in range(num_epochs):
        loss = 0.0

        for batch in dataloader:
            rng, step_rng = jax.random.split(rng)

            # Normalize the batch using pre-computed stats.
            batch = normalizer(batch)

            # Perform a SGD step, updating model parameters in-place.
            batch_loss = train_step(model, optimizer, batch, step_rng)
            loss += batch_loss

        if (epoch + 1) % print_frequency == 0 or epoch == 0:
            loss = loss / len(dataloader)
            elapsed = datetime.now() - start_time
            print(
                f"Epoch {epoch + 1}/{num_epochs}"
                f" | Loss {loss:.4f}"
                f" | Time {elapsed}"
            )

    return model, normalizer
