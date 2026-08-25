"""Shared fixtures for the test suite."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cfm.core.constraints import equality, inequality
from cfm.models.flow import FlowMLP
from cfm.models.normalizer import Normalizer

# An untrained model has an arbitrary vector field, so the constrained flows
# go unstable under the penalty weights the real examples use. These are the
# strongest settings that stay finite here.
PENALTY_WEIGHT = 1.5
DT = 0.02
NUM_SAMPLES = 6


@pytest.fixture
def model():
    """A tiny 2-D flow model."""
    return FlowMLP(
        data_shape=(2,),
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture
def normalizer():
    """A non-trivial normalizer, so unnormalize() is actually exercised."""
    return Normalizer(mean=jnp.array([0.5, -0.25]), std=jnp.array([1.2, 0.8]))


@pytest.fixture
def identity_normalizer():
    return Normalizer(mean=jnp.zeros(2), std=jnp.ones(2))


@pytest.fixture
def circle():
    """Equality constraint: samples must land on the unit circle."""
    return equality(lambda x: jnp.sum(x**2, axis=-1) - 1.0, name="||x|| = 1")


@pytest.fixture
def right_half():
    """Inequality constraint: samples must have x[0] >= 0."""
    return inequality(lambda x: jnp.atleast_1d(-x[0]), name="x[0] >= 0")


@pytest.fixture
def rng():
    return jax.random.key(0)


def constraint_for(kind, circle, right_half):
    """Pick the fixture matching a method's supported constraint kind."""
    return circle if kind == "equality" else right_half
