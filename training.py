from flax import nnx
import jax.numpy as jnp
import jax

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
    xt = t[:, None] * x1 + (1 - t)[:, None] * x0
    target = x1 - x0
    pred = model(xt, t)
    return jnp.mean(jnp.square(pred - target))

def train_step(batch, model, optimizer, rng):
    pass

def train(dataloader, model, num_epochs, rng):
    pass
