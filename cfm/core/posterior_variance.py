"""Isotropic posterior variance r_t^2 of the clean-sample estimate.

Peng et al. 2024, https://arxiv.org/abs/2402.02149 (Sec. 4.2): approximate
p(x1 | x_t) by N(mu, r_t^2 I) with the maximum-likelihood isotropic
variance

    r_t^2 = E ||x1 - mu(x_t)||^2 / d,    mu = x_t + (1 - t) v(x_t, t),

estimated offline by Monte Carlo over data. Here t = 0 is noise and t = 1
is data, with x_t = t x1 + (1 - t) x0 as in training.
"""

import json
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np


def estimate(
    model, normalizer, data: jax.Array, num_times: int = 101,
    num_noise: int = 1, batch_size: int = 1024, seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Monte Carlo estimate of r_t^2 on an even grid over [0, 1].

    Args:
        data: unnormalized samples, ``(n, *model.data_shape)``.
        num_noise: noise draws per data sample and time.

    Returns:
        ``(ts, r2)``, both of shape ``(num_times,)``, in normalized space.
    """
    x1_all = normalizer.normalize(jnp.asarray(data))
    d = int(np.prod(model.data_shape))
    ts = np.linspace(0.0, 1.0, num_times)

    @jax.jit
    def batch_error(t, x1, rng):
        x0 = jax.random.normal(rng, x1.shape)
        xt = t * x1 + (1 - t) * x0
        mu = xt + (1 - t) * model(xt, jnp.full((x1.shape[0],), t))
        return jnp.sum((x1 - mu) ** 2)

    rng = jax.random.key(seed)
    r2 = np.zeros(num_times)
    n = x1_all.shape[0]
    for i, t in enumerate(ts):
        total = 0.0
        for _ in range(num_noise):
            for start in range(0, n, batch_size):
                rng, key = jax.random.split(rng)
                total += float(batch_error(
                    t, x1_all[start:start + batch_size], key))
        r2[i] = total / (num_noise * n * d)
    return ts, r2


def save(path, ts, r2, **metadata) -> None:
    """Write a variance table as JSON with provenance metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        **metadata, "ts": [float(t) for t in ts],
        "r2": [float(r) for r in r2],
    }, indent=2) + "\n")


def load(path) -> Tuple[jax.Array, jax.Array]:
    """Read ``(ts, r2)`` from a table written by ``save``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; estimate it with "
            "`uv run -m experiments.posterior_variance --problem <name>`"
        )
    table = json.loads(path.read_text())
    return jnp.asarray(table["ts"]), jnp.asarray(table["r2"])
