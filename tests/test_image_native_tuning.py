"""Native state and experimental initialization preserve the LDF equations."""
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfm.methods import ldf
from cfm.models.normalizer import Normalizer
from experiments.imagenet_native_tuning import initialized_ldf, resolved
from problems.imagenet import fixed_pixel_constraint


class ToyFlow:
    data_shape = (2, 2, 1)

    def __call__(self, x, t):
        return -.3 * x + .1 * jnp.sin(x.mean(axis=(1, 2, 3), keepdims=True))


@pytest.mark.parametrize("solver", [diffrax.Midpoint(), diffrax.Heun()])
def test_zero_initial_duals_reproduce_public_ldf(solver):
    model, normalizer = ToyFlow(), Normalizer(.5, 1.)
    reference = np.array([.1, .3, .5, .7]).reshape(2, 2, 1)
    constraint = fixed_pixel_constraint(reference, np.array([[1, 0], [1, 0]]))
    key = jax.random.key(5)
    cfg = dict(dt=.01, penalty_weight=2., rescale_factor=.3,
               rescale_exponent=1.5, solver=solver)
    expected = ldf.generate(model, normalizer, constraint,
                            num_samples=1, rng=key, **cfg)
    image, steps = initialized_ldf(model, normalizer, constraint, key,
                                   dual_cancel=0., **cfg)
    np.testing.assert_allclose(image, expected.x[0], atol=1e-6)
    assert int(steps) == int(expected.num_steps)


def test_initial_duals_cancel_only_the_initial_penalty_force():
    model = ToyFlow()
    x0 = jax.random.normal(jax.random.key(1), (1, *model.data_shape))
    def g(x):
        return 3 * (x[jnp.array([0, 2])] - jnp.array([.1, .5]))
    residual = jax.vmap(g)(x0.reshape(1, -1))
    ode = ldf._equality_flow(model, g, .7, 2.)
    xdot, dualdot = ode(0., (x0, -residual), None)
    np.testing.assert_allclose(xdot, model(x0, jnp.array([0.])), atol=1e-7)
    np.testing.assert_allclose(dualdot, .7 * residual / (1 + 1e-6)**2,
                               atol=1e-6)


def test_plan_rejects_unsupported_or_mistyped_settings():
    with pytest.raises(ValueError, match="Unknown settings"):
        resolved(dict(name="trial", method="ldf", data_scale=.5))
    assert resolved(dict(name="trial", method="pigdm", solver="heun"))[
        "solver"] == "heun"
    with pytest.raises(ValueError, match="both rtol and atol"):
        resolved(dict(name="trial", method="ldf", rtol=.001))


@pytest.mark.parametrize("method", ["unconstrained", "ldf", "pigdm"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_snapshot_saving_preserves_samples_with_custom_solver(method, adaptive):
    from cfm import methods

    model, normalizer = ToyFlow(), Normalizer(.5, 1.)
    constraint = fixed_pixel_constraint(
        np.zeros((2, 2, 1)), np.array([[1, 0], [1, 0]]),
    )
    cfg = dict(num_samples=1, seed=3, dt=.01, solver=diffrax.Heun())
    if adaptive:
        cfg.update(solver=diffrax.Tsit5(), stepsize_controller=
                   diffrax.PIDController(rtol=1e-5, atol=1e-6))
    if method == "unconstrained":
        def run(**kw):
            return ldf.generate_unconstrained(model, normalizer, **kw)
    else:
        if method == "ldf":
            cfg.update(initial_dual_scale=-1., penalty_weight=2.,
                       rescale_exponent=0.)
        def run(**kw):
            return methods.get(method).run(model, normalizer, constraint, **kw)
    full = run(**cfg)
    final = run(**cfg, save_trajectory=False)
    np.testing.assert_allclose(final.x, full.x, atol=1e-6)
    assert final.xs.shape[0] == (1 if method == "pigdm" else 2)
    assert final.xs.shape[0] < full.xs.shape[0]


def test_initial_duals_cancel_a_stationary_penalty_flow():
    from cfm.core.constraints import equality, inequality
    from cfm.core.solve import initial_noise

    class Stationary:
        data_shape = (2,)

        def __call__(self, x, t):
            return jnp.zeros_like(x)

    model, normalizer = Stationary(), Normalizer(0., 1.)
    constraint = equality(lambda x: jnp.atleast_1d(x[0] - 1.5))
    key = jax.random.key(3)
    x0 = initial_noise(key, 1, model.data_shape)
    cfg = dict(num_samples=1, rng=key, dt=.01, penalty_weight=2.,
               rescale_factor=0., save_trajectory=False)
    canceled = ldf.generate(model, normalizer, constraint,
                            initial_dual_scale=-1., **cfg)
    default = ldf.generate(model, normalizer, constraint, **cfg)
    np.testing.assert_allclose(canceled.x, x0, atol=1e-6)
    assert abs(float(default.x[0, 0]) - 1.5) < abs(float(x0[0, 0]) - 1.5)
    with pytest.raises(ValueError, match="equality constraints only"):
        ldf.generate(model, normalizer, inequality(constraint.fn),
                     initial_dual_scale=-1., **cfg)
