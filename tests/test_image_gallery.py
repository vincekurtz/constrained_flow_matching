"""Random gallery reproducibility and dynamic fixed-pixel constraints."""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cfm import methods
from cfm.models.normalizer import Normalizer
from experiments import imagenet_inpainting_gallery as gallery
from problems import imagenet


def test_random_cases_are_reproducible_and_independent():
    cases = gallery.make_cases(20261107)
    assert cases == gallery.make_cases(20261107)
    assert cases != gallery.make_cases(20261108)
    seeds = [c[k] for c in cases for k in ("reference_seed", "sample_seed")]
    assert len(set(seeds)) == 10
    assert len({tuple(c["hole"]) for c in cases}) == 5
    for case in cases:
        observed = gallery.observed_mask(case)
        assert observed.sum() == 512**2 - 192**2
        assert observed[:32].all() and observed[-32:].all()
        assert observed[:, :32].all() and observed[:, -32:].all()


def test_existing_plan_cannot_be_silently_replaced(tmp_path):
    source = tmp_path / "model.pkl"
    source.write_bytes(b"checkpoint identity")
    root = tmp_path / "results"
    plan = gallery.initialize(root, source, 5)
    assert gallery.initialize(root, source, 5) == plan
    with pytest.raises(ValueError, match="Existing plan differs"):
        gallery.initialize(root, source, 6)
    source.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="Existing plan differs"):
        gallery.initialize(root, source, 5)


def test_figure_cannot_silently_drop_cases(tmp_path):
    gallery.write_json(tmp_path / "plan.json", {
        "experiment_seed": 1, "cases": gallery.make_cases(1)[:-1],
    })
    with pytest.raises(ValueError, match="all five planned"):
        gallery.comparison(tmp_path, tmp_path / "figure.png")
    assert not (tmp_path / "figure.png").exists()


@pytest.mark.parametrize("stage", ["pigdm", "ldf"])
def test_dynamic_indices_match_existing_sampler_constraints(stage):
    class Model:
        data_shape = (2, 2, 1)

        def __call__(self, x, t):
            return -.3 * x + .1 * jnp.sin(x.mean(
                axis=(1, 2, 3), keepdims=True,
            ))

    model, normalizer = imagenet.PROBLEM.model_for(
        stage, Model(), Normalizer(.5, 2.),
    )
    method = methods.get(stage)
    reduction = "norm" if stage == "pigdm" else "pixels"
    traces = []

    @jax.jit
    def dynamic(reference, indices):
        traces.append(True)
        constraint = imagenet.indexed_pixel_constraint(
            reference, indices, reduction=reduction,
        )
        return method.run(model, normalizer, constraint, num_samples=1,
                          seed=2, dt=.02).x

    for observed, values in [
        ([[True, False], [True, False]], [.2, 0, .6, 0]),
        ([[False, True], [False, True]], [0, .3, 0, .7]),
    ]:
        observed = np.array(observed)
        reference = np.array(values, dtype=np.float32).reshape(2, 2, 1)
        indices = jnp.array(np.flatnonzero(observed), dtype=jnp.int32)
        constraint = imagenet.fixed_pixel_constraint(
            reference, observed, reduction=reduction,
        )
        expected = method.run(model, normalizer, constraint, num_samples=1,
                              seed=2, dt=.02).x
        np.testing.assert_allclose(dynamic(reference, indices), expected,
                                   atol=1e-6, rtol=1e-5)
    assert len(traces) == 1


def test_record_validation_detects_replaced_reference_or_sample():
    plan = {"checkpoint_sha256": "checkpoint", "settings": gallery.settings()}
    case = gallery.make_cases(1)[0]
    reference = np.zeros(imagenet.IMAGE_SHAPE, dtype=np.float32)
    observed = gallery.observed_mask(case)
    image = reference.copy()
    record = {
        "checkpoint_sha256": "checkpoint", "case": case, "method": "ldf",
        "settings": plan["settings"]["ldf"], "coordinates": "ldf",
        "reference_sha256": gallery.digest(reference),
        "mask_sha256": gallery.digest(observed),
        "sample_sha256": gallery.digest(image),
    }
    gallery.validate_record(record, plan, case, "ldf", reference,
                            observed, image)
    changed = copy.deepcopy(record)
    changed["case"]["sample_seed"] += 1
    with pytest.raises(ValueError, match="Mismatched inputs"):
        gallery.validate_record(changed, plan, case, "ldf", reference,
                                observed, image)
    image[0, 0, 0] = .1
    with pytest.raises(ValueError, match="Sample hash mismatch"):
        gallery.validate_record(record, plan, case, "ldf", reference,
                                observed, image)
