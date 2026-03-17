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


def train(dataloader, model, num_epochs, rng):
    pass
