"""Study integrity and metric correctness, without pretrained downloads."""

import json

import numpy as np
import pytest
from scipy.linalg import sqrtm

from experiments import imagenet_inpainting_gallery as gallery
from experiments import imagenet_inpainting_metrics as metrics
from experiments import imagenet_inpainting_study as study


def test_low_rank_fid_matches_covariance_formula():
    rng = np.random.default_rng(10)
    for n, m, d in [(7, 5, 12), (30, 22, 4)]:
        x, y = rng.normal(size=(n, d)), rng.normal(size=(m, d)) + .3
        a, b = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
        expected = np.sum((x.mean(0) - y.mean(0))**2)
        expected += np.trace(a + b - 2 * sqrtm(a @ b).real)
        np.testing.assert_allclose(metrics.frechet_distance(x, y), expected,
                                   atol=1e-6)
        assert metrics.frechet_distance(x, x) < 1e-10
    with pytest.raises(ValueError, match="at least two"):
        metrics.frechet_distance(x[:1], y)


def test_paired_kid_excludes_dependent_cross_pairs():
    x = np.array([[1., 0.], [2., 1.], [-1., 2.]])
    y = x + np.array([.2, -.1])
    values = []
    for i in range(3):
        for j in range(3):
            if i != j:
                def kernel(a, b):
                    return (a @ b / 2 + 1)**3
                values.append(kernel(x[i], x[j]) + kernel(y[i], y[j])
                              - 2 * kernel(x[i], y[j]))
    np.testing.assert_allclose(metrics.paired_kid(x, y), np.mean(values))
    assert metrics.paired_kid(x, x) == 0
    with pytest.raises(ValueError, match="equal feature"):
        metrics.paired_kid(x, y[:2])


def test_study_plan_is_frozen_and_methods_share_cases(tmp_path):
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"model")
    root = tmp_path / "study"
    plan = study.initialize(root, checkpoint, count=8)
    assert study.initialize(root, checkpoint, count=8) == plan
    assert len(plan["cases"]) == 8
    assert plan["representative_ids"] == [1, 2, 3, 4, 5]
    assert set(plan["settings"]) == {"references", *study.METHODS}
    assert plan["settings"]["pcfm"]["num_final_projection_iters"] == 0
    with pytest.raises(ValueError, match="Existing study differs"):
        study.initialize(root, checkpoint, count=9)


def test_raw_errors_are_not_hidden_by_display_clipping():
    ref = np.ones((2, 2, 3), dtype=np.float32)
    sample = ref.copy()
    sample[0] = 1.2
    mask = np.array([[True, True], [False, False]])
    result = study.sample_metrics(sample, ref, mask)
    assert result["observed_mae"] == pytest.approx(.2)
    assert result["out_of_range_fraction"] == .5
    assert (metrics.png_array(sample) == 255).all()


def test_resume_rejects_replaced_samples_and_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(study.imagenet, "IMAGE_SHAPE", (2, 2, 3))
    case = {"id": 1}
    plan = {"cases": [case]}
    folder = tmp_path / "case_1"
    folder.mkdir()
    image = np.zeros((2, 2, 3), dtype=np.float32)
    np.save(folder / "reference.npy", image)
    record = {"plan_sha256": study.plan_hash(plan), "case_id": 1,
              "stage": "references", "sample_sha256": gallery.digest(image)}
    gallery.write_json(folder / "reference.json", record)
    study.load_sample(tmp_path, plan, case, "references")
    with pytest.raises(ValueError, match="Invalid or mismatched"):
        study.load_sample(tmp_path, {**plan, "changed": True},
                          case, "references")
    image[0] = 1
    np.save(folder / "reference.npy", image)
    with pytest.raises(ValueError, match="Invalid or mismatched"):
        study.load_sample(tmp_path, plan, case, "references")


def test_evaluation_refuses_unfinished_study(tmp_path):
    gallery.write_json(tmp_path / "plan.json", {"cases": [{"id": 1}]})
    with pytest.raises(ValueError, match="Study incomplete"):
        metrics.evaluate(tmp_path, tmp_path / "report")
    assert not (tmp_path / "report/results.csv").exists()


def test_real_reference_manifest_is_deterministic(tmp_path):
    from PIL import Image
    images = tmp_path / "images"
    images.mkdir()
    for i in range(4):
        Image.new("RGB", (16, 16), (i, 0, 0)).save(images / f"{i}.png")
    checkpoint = tmp_path / "model.pkl"
    checkpoint.write_bytes(b"model")
    plan = study.initialize(tmp_path / "study", checkpoint, count=3,
                            reference_dir=images)
    assert plan["source"]["kind"] == "real"
    assert len({c["source_sha256"] for c in plan["cases"]}) == 3
    persisted = json.loads((tmp_path / "study/plan.json").read_text())
    assert persisted == plan


@pytest.mark.parametrize("method_name", ["pcfm", "pigdm", "ldf"])
def test_batching_keeps_independent_constraints_and_random_keys(
        method_name, monkeypatch):
    import jax
    import jax.numpy as jnp
    from cfm import methods
    from cfm.models.normalizer import Normalizer
    from problems import imagenet

    class Model:
        data_shape = (2, 2, 1)

        def __call__(self, x, t):
            return -.3 * x + .1 * jnp.sin(x.mean(
                axis=(1, 2, 3), keepdims=True,
            ))

    refs = {i: np.full((2, 2, 1), i / 10, dtype=np.float32)
            for i in range(1, 4)}
    masks = {i: np.array([[True, False], [i % 2 == 0, i % 2 == 1]])
             for i in refs}
    cases = [{"id": i, "sample_seed": i + 10} for i in refs]
    monkeypatch.setattr(study, "load_sample", lambda root, plan, case, stage:
                        (refs[case["id"]], {}))
    monkeypatch.setattr(gallery, "observed_mask",
                        lambda case: masks[case["id"]])
    cfg = ({"num_steps": 4, "num_correction_iters": 1,
            "num_projection_iters": 1, "num_final_projection_iters": 0}
           if method_name == "pcfm" else
           {"dt": .05, "save_trajectory": False})
    if method_name == "ldf":
        cfg["rescale_exponent"] = 0.
    if method_name == "pigdm":
        import diffrax
        cfg["solver"] = diffrax.Tsit5()
        cfg["stepsize_controller"] = diffrax.PIDController(
            rtol=1e-6, atol=1e-7,
        )

    @jax.jit
    def sample(key, reference, indices):
        constraint = imagenet.indexed_pixel_constraint(
            reference, indices,
            reduction="pixels" if method_name == "ldf" else "norm",
        )
        return methods.get(method_name).run(
            Model(), Normalizer(.5, 1.), constraint, num_samples=1,
            rng=key, **cfg,
        ).x[0]

    jobs = list(study.batched_sample_jobs(
        sample, cases, None, {}, method_name, batch_size=2,
    ))
    assert len(jobs) == 3  # Padding is never retained.
    for case, image, seconds, compilation, batch_size, error in jobs:
        assert error is None and batch_size == 2 and seconds >= 0
        expected = sample(jax.random.key(case["sample_seed"]),
                          jnp.asarray(refs[case["id"]]), jnp.asarray(
                              np.flatnonzero(masks[case["id"]])))
        np.testing.assert_allclose(image, expected, atol=1e-5)
    assert jobs[0][3] and jobs[1][3] and not jobs[2][3]
