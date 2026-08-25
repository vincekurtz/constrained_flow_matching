"""Shared fixtures for the test suite.

# Why the fixtures are session-scoped

Every ``generate`` call costs roughly a second of XLA compilation, and that
cost is flat: it does not care how many integration steps or samples the call
makes. The suite's runtime is therefore set by the number of *distinct*
generator calls it makes, not by their size. Shrinking ``dt`` or
``NUM_SAMPLES`` buys nothing; not calling the generator twice for the same
result buys a second each time.

So the model, normalizer and constraints are built once per session and the
``run_once`` cache lets several tests share one generator result. Fixtures
here are treated as read-only: nothing in the suite trains or mutates them.

For the same reason the suite keeps XLA's compiled kernels on disk between
runs, which roughly halves a re-run. The cache is keyed on the HLO, the
backend and the JAX version, so it invalidates itself when the code changes;
delete CACHE_DIR if you ever want to force a cold run.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from cfm.core.constraints import equality, inequality
from cfm.models.flow import FlowMLP
from cfm.models.normalizer import Normalizer

CACHE_DIR = ".jax_cache"

jax.config.update("jax_compilation_cache_dir", CACHE_DIR)
# The defaults skip anything that compiles in under a second, which is most
# of what this suite builds.
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)

# An untrained model has an arbitrary vector field, so the constrained flows
# go unstable under the penalty weights the real examples use. These are the
# strongest settings that stay finite here.
PENALTY_WEIGHT = 1.5
DT = 0.02
NUM_SAMPLES = 6
SEED = 0


@pytest.fixture(scope="session")
def model():
    """A tiny 2-D flow model."""
    return FlowMLP(
        data_shape=(2,),
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )


@pytest.fixture(scope="session")
def normalizer():
    """A non-trivial normalizer, so unnormalize() is actually exercised."""
    return Normalizer(mean=jnp.array([0.5, -0.25]), std=jnp.array([1.2, 0.8]))


@pytest.fixture(scope="session")
def identity_normalizer():
    return Normalizer(mean=jnp.zeros(2), std=jnp.ones(2))


@pytest.fixture(scope="session")
def circle():
    """Equality constraint: samples must land on the unit circle."""
    return equality(lambda x: jnp.sum(x**2, axis=-1) - 1.0, name="||x|| = 1")


@pytest.fixture(scope="session")
def right_half():
    """Inequality constraint: samples must have x[0] >= 0."""
    return inequality(lambda x: jnp.atleast_1d(-x[0]), name="x[0] >= 0")


@pytest.fixture(scope="session")
def feasible():
    """An inequality no sample can violate, for "leaves it alone" checks."""
    return inequality(
        lambda x: jnp.atleast_1d(jnp.sum(x**2) - 1e6), name="always feasible"
    )


@pytest.fixture
def rng():
    return jax.random.key(SEED)


@pytest.fixture(scope="session")
def run_once():
    """Memoize generator results across tests, keyed by an explicit label.

    Use it when a test only *inspects* a result that another test also needs.
    Never use it where repeating the call is the point of the test -- a
    determinism check has to actually run the generator twice.
    """
    cache = {}

    def get(key, build):
        if key not in cache:
            cache[key] = build()
        return cache[key]

    return get


@pytest.fixture(scope="session")
def baseline(run_once, model, normalizer):
    """The unconstrained flow at test settings.

    Several tests compare against it, so it is generated once and shared.
    """
    from cfm.methods.ldf import generate_unconstrained

    return run_once(
        "unconstrained",
        lambda: generate_unconstrained(
            model, normalizer, num_samples=NUM_SAMPLES, dt=DT, seed=SEED
        ),
    )


def constraint_for(kind, circle, right_half):
    """Pick the fixture matching a method's supported constraint kind."""
    return circle if kind == "equality" else right_half
