"""Fixed generator calls whose outputs are pinned in ``tests/goldens/``."""

import jax
import jax.numpy as jnp
from flax import nnx

from cfm.core.constraints import equality, inequality
from cfm.methods.cbf import generate as generate_cbf
from cfm.methods.ldf import generate, generate_unconstrained
from cfm.methods.pcfm import generate as generate_pcfm
from cfm.methods.pigdm import generate as generate_pigdm
from cfm.models.flow import FlowMLP
from cfm.models.normalizer import Normalizer

GOLDEN_DIR = "tests/goldens"

NUM_SAMPLES = 8
DT = 0.01

# The untrained model diverges under the examples' penalty weights.
PENALTY_WEIGHT = 1.5
RESCALE_FACTOR = 1.0
INEQ_RESCALE_FACTOR = 10.0


def make_model():
    """A tiny 2-D flow model."""
    return FlowMLP(
        data_shape=(2,),
        time_embedding_size=8,
        hidden_sizes=(16, 16),
        rngs=nnx.Rngs(0),
    )


def make_normalizer():
    """A non-trivial normalizer, so unnormalize() is actually exercised."""
    return Normalizer(mean=jnp.array([0.5, -0.25]), std=jnp.array([1.2, 0.8]))


def circle_constraint(x):
    """Equality g(x) = ||x||^2 - 1."""
    return jnp.sum(x**2, axis=-1) - 1.0


def right_half_constraint(x):
    """Inequality h(x) = -x[0] <= 0."""
    return jnp.atleast_1d(-x[0])


def circle():
    return equality(circle_constraint, name="unit circle")


def right_half():
    return inequality(right_half_constraint, name="x[0] >= 0")


def _rng():
    return jax.random.key(0)


# name -> zero-arg callable returning the final samples. Changing an
# argument invalidates its golden.
CASES = {}


def case(name):
    def register(fn):
        CASES[name] = fn
        return fn
    return register


@case("unconstrained")
def _unconstrained():
    return generate_unconstrained(
        make_model(), make_normalizer(), num_samples=NUM_SAMPLES, dt=DT, seed=0
    ).x


@case("ldf_equality")
def _ldf_equality():
    return generate(
        make_model(), make_normalizer(), circle(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=RESCALE_FACTOR,
        rescale_exponent=2.0,
    ).x


@case("ldf_equality_projected")
def _ldf_equality_projected():
    return generate(
        make_model(), make_normalizer(), circle(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=RESCALE_FACTOR,
        num_projection_iters=2,
    ).x


@case("penalty_equality")
def _penalty_equality():
    return generate(
        make_model(), make_normalizer(), circle(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=0.0,
    ).x


@case("ldf_inequality_closed_form")
def _ldf_inequality_closed_form():
    return generate(
        make_model(), make_normalizer(), right_half(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=INEQ_RESCALE_FACTOR,
        slack="closed_form",
    ).x


@case("ldf_inequality_ode")
def _ldf_inequality_ode():
    return generate(
        make_model(), make_normalizer(), right_half(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=INEQ_RESCALE_FACTOR,
        slack="ode",
    ).x


@case("ldf_inequality_projected")
def _ldf_inequality_projected():
    return generate(
        make_model(), make_normalizer(), right_half(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=INEQ_RESCALE_FACTOR,
        num_projection_iters=3,
    ).x


@case("penalty_inequality")
def _penalty_inequality():
    return generate(
        make_model(), make_normalizer(), right_half(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        penalty_weight=PENALTY_WEIGHT, rescale_factor=0.0, slack="closed_form",
    ).x


@case("pcfm")
def _pcfm():
    return generate_pcfm(
        make_model(), make_normalizer(), circle(),
        num_samples=NUM_SAMPLES, num_steps=20, rng=_rng(),
        correction_weight=0.1,
    ).x


@case("pigdm")
def _pigdm():
    return generate_pigdm(
        make_model(), make_normalizer(), circle(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        guidance_scale=1.0, eps_reg=1e-4,
    ).x


@case("cbf")
def _cbf():
    return generate_cbf(
        make_model(), make_normalizer(), right_half(),
        num_samples=NUM_SAMPLES, dt=DT, rng=_rng(),
        phi0=1.0, omega=4.0, qp="elastic",
    ).x
