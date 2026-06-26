"""Tests for the D3IL-style avoiding task: dataset, constraint, and sampler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from architectures.flow import FlowMLP
from datasets.avoiding import (
    AvoidingDataset,
    START,
    TRAJ_LEN,
    _trajectory_collides,
)
from examples.avoiding import (
    FEASIBILITY_TOL,
    avoiding_constraint,
    avoiding_penalty,
    _violation,
)
from generation import generate, generate_constrained
import training


@pytest.fixture(scope="module")
def dataset():
    return AvoidingDataset(num_samples=256)


def test_dataset_shape_and_start(dataset):
    """Each demo is a (TRAJ_LEN, 2) trajectory anchored at START."""
    assert dataset.data.shape == (256, TRAJ_LEN, 2)
    starts = dataset.data.numpy()[:, 0, :]
    assert np.allclose(starts, np.array(START), atol=1e-4)


def test_demos_avoid_original_pillars(dataset):
    """Training demos are collision-free w.r.t. the original pillars."""
    assert not any(_trajectory_collides(t) for t in dataset.data.numpy())


def test_constraint_shape_and_differentiable():
    """avoiding_constraint stacks per-waypoint terms with a finite Jacobian."""
    x = jnp.zeros((TRAJ_LEN, 2))
    h = avoiding_constraint(x)
    # One or more constraint terms per waypoint (circle and/or funnel lines).
    assert h.ndim == 1 and h.shape[0] % TRAJ_LEN == 0
    jac = jax.jacobian(avoiding_constraint)(x)
    assert jac.shape == (h.shape[0], TRAJ_LEN, 2)
    assert jnp.all(jnp.isfinite(jac))


def test_constraints_are_new(dataset):
    """The inference constraints are genuinely unseen: some demos violate them.

    The whole point of the experiment is that the demonstrations never obeyed
    the inference-time constraint set, so it is non-trivial w.r.t. the data.
    """
    viol = jax.vmap(_violation)(jnp.array(dataset.data.numpy()))
    assert float((viol > FEASIBILITY_TOL).mean()) > 0.0


def test_constrained_sampler_enforces_feasibility(dataset):
    """Soft-penalty equality flow lowers violation vs. the plain flow."""
    model = FlowMLP(
        data_shape=(TRAJ_LEN, 2),
        time_embedding_size=8,
        hidden_sizes=(64, 64),
        rngs=nnx.Rngs(0),
    )
    model, normalizer = training.train(
        dataset=dataset,
        model=model,
        num_epochs=30,
        batch_size=64,
        learning_rate=1e-3,
        print_frequency=30,
    )

    x_unc, _ = generate(model, normalizer, num_samples=32, dt=0.02)
    x, _, _ = generate_constrained(
        model,
        normalizer,
        avoiding_penalty,
        num_samples=32,
        dt=0.02,
        penalty_weight=20.0,
        rescale_factor=1.0,
        rescale_exponent=2.0,
    )
    assert jnp.all(jnp.isfinite(x))
    # The penalty flow should lower the mean violation relative to no penalty.
    assert float(jax.vmap(_violation)(x).mean()) < float(
        jax.vmap(_violation)(x_unc).mean()
    )
