"""Paired, resumable inpainting study using the public sampler APIs.

Initialize once, then run references, pcfm, pigdm, ldf, and evaluate.
Synthetic references are a model-consistency pilot, not a real-data benchmark.

Baselines added after a study was run (DPS, OPC) are registered with
``--stage extend``. Their settings live in ``extensions.json`` next to the
frozen plan, so existing samples stay valid; each added sample records the
hash of its own settings.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import subprocess
import sys

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

from cfm import methods
from cfm.core import checkpoint
from cfm.methods.ldf import generate_unconstrained
from cfm.models.jit_image import JiTImageFlow
from experiments import imagenet_inpainting_gallery as gallery
from problems import imagenet

METHODS = ("pcfm", "pigdm", "ldf")
EXTRA_METHODS = ("dps", "opc")
STAGES = ("references", *METHODS)
EXTENSIONS = "extensions.json"


def extension_settings(dps_overrides=None, opc_overrides=None):
    """Settings for the added baselines, from the ImageNet problem gains.

    OPC keeps PiGDM's guidance settings but integrates with fixed Midpoint
    steps (see ``problems/imagenet.py``).
    """
    result = {name: getattr(methods, name.upper()).config(
        **imagenet.PROBLEM.gains_for(name)) for name in EXTRA_METHODS}
    for name, overrides in (("dps", dps_overrides), ("opc", opc_overrides)):
        if overrides:
            if set(overrides) - (set(result[name]) | {"solver", "rtol",
                                                      "atol", "max_steps"}):
                raise ValueError(f"Unsupported {name} study setting")
            result[name].update(overrides)
        result[name]["save_trajectory"] = False
    return result


def extend(root, settings):
    """Register added baselines; refuse to change ones already registered."""
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    path = root / EXTENSIONS
    current = json.loads(path.read_text()) if path.exists() else {
        "plan_sha256": plan_hash(plan), "settings": {}}
    if current["plan_sha256"] != plan_hash(plan):
        raise ValueError("Extensions belong to a different plan")
    for name, cfg in settings.items():
        if name not in EXTRA_METHODS:
            raise ValueError(f"Unknown added baseline: {name}")
        if name in current["settings"] and current["settings"][name] != cfg:
            raise ValueError(f"{name} already registered with other settings")
        current["settings"][name] = cfg
    gallery.write_json(path, current)
    return current


def study_settings(root, plan):
    """Plan settings plus registered extensions, with their hashes."""
    settings = dict(plan.get("settings", {}))
    path = Path(root) / EXTENSIONS
    hashes = {}
    if path.exists():
        extensions = json.loads(path.read_text())
        if extensions["plan_sha256"] != plan_hash(plan):
            raise ValueError("Extensions belong to a different plan")
        for name, cfg in extensions["settings"].items():
            settings[name] = cfg
            hashes[name] = plan_hash(cfg)
    return settings, hashes


def study_methods(root, plan):
    """Base methods, then registered extensions in a fixed order."""
    _, hashes = study_settings(root, plan)
    return (*METHODS, *(m for m in EXTRA_METHODS if m in hashes))


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()


def initialize(root, checkpoint_path, count=256, seed=20261108,
               reference_dir=None, source_gallery=None, pcfm_overrides=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = gallery.settings(pigdm_overrides=json.loads(Path(
        "experiments/imagenet_native/gallery_pigdm.json"
    ).read_text()))
    cfg["references"]["save_trajectory"] = False
    cfg["pcfm"] = methods.PCFM.config(**imagenet.PROBLEM.gains_for("pcfm"))
    if pcfm_overrides:
        allowed = set(cfg["pcfm"]) | {"num_projection_iters"}
        if set(pcfm_overrides) - allowed:
            raise ValueError("Unsupported PCFM study setting")
        if pcfm_overrides.get("num_final_projection_iters", 0) != 0:
            raise ValueError("Study excludes additional final projection")
        cfg["pcfm"].update(pcfm_overrides)
    cases = gallery.make_cases(seed, count=count)
    checkpoint_sha256 = file_hash(checkpoint_path)
    source = {"kind": "synthetic"}
    if reference_dir and source_gallery:
        raise ValueError("Choose either real references or a source gallery")
    if source_gallery:
        original = json.loads((Path(source_gallery) / "plan.json").read_text())
        if original["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("Source gallery uses a different checkpoint")
        cases = original["cases"]
        seed = original["experiment_seed"]
        source = {"kind": "existing_gallery", "path": str(
            Path(source_gallery).resolve()), "plan_sha256": plan_hash(original)}
        for method in ("pigdm", "ldf"):
            cfg[method] = original["settings"][method]
    elif reference_dir:
        files = sorted(p for p in Path(reference_dir).rglob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(files) < count:
            raise ValueError(f"Need {count} images, found {len(files)}")
        selected = np.random.default_rng(seed).choice(
            len(files), count, replace=False,
        )
        source = {"kind": "real", "path": str(Path(reference_dir).resolve())}
        for case, index in zip(cases, selected):
            case["reference_path"] = str(files[index].resolve())
            case["source_sha256"] = file_hash(files[index])
    plan = {
        "schema": 1, "seed": seed, "checkpoint": str(
            Path(checkpoint_path).resolve()),
        "checkpoint_sha256": checkpoint_sha256, "source": source,
        "cases": cases, "settings": cfg,
        "representative_ids": [c["id"] for c in cases[:5]],
        "protocol": {
            "selection": "All planned cases; first five shown unfiltered",
            "paired_noise": True, "mask": "192px square at random position",
            "output_compositing": False, "pcfm_final_projection_iters": 0,
            "pcfm_per_step_projection": True,
            "evaluation": "Clean-FID Inception pool3, 2048 features; uint8 PNG",
            "regions": ["full", "hole"],
        },
    }
    path = root / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing study differs; use a new output directory")
    gallery.write_json(path, plan)
    return plan


def load_sample(root, plan, case, stage):
    folder = Path(root) / f"case_{case['id']}"
    stem = folder / ("reference" if stage == "references" else stage)
    record = json.loads(stem.with_suffix(".json").read_text())
    image = np.load(stem.with_suffix(".npy"))
    if (record.get("plan_sha256") != plan_hash(plan)
            or record.get("case_id") != case["id"]
            or record.get("stage") != stage
            or record.get("sample_sha256") != gallery.digest(image)
            or image.shape != imagenet.IMAGE_SHAPE
            or not np.isfinite(image).all()):
        raise ValueError(f"Invalid or mismatched sample: {stem}")
    if stage in EXTRA_METHODS:
        _, hashes = study_settings(root, plan)
        if record.get("settings_sha256") != hashes.get(stage):
            raise ValueError(f"Settings changed: {stem}")
    if stage != "references":
        reference, _ = load_sample(root, plan, case, "references")
        if record.get("reference_sha256") != gallery.digest(reference):
            raise ValueError(f"Reference changed: {stem}")
    return image, record


def runtime_settings(cfg):
    cfg = dict(cfg)
    if "solver" in cfg:
        cfg["solver"] = gallery.SOLVERS[cfg["solver"]]()
    rtol, atol = cfg.pop("rtol", None), cfg.pop("atol", None)
    if (rtol is None) != (atol is None):
        raise ValueError("Specify both rtol and atol")
    if rtol is not None:
        cfg["stepsize_controller"] = diffrax.PIDController(rtol=rtol, atol=atol)
    return cfg


def sample_metrics(image, reference, observed):
    error = np.asarray(image, dtype=np.float64) - reference
    return {
        "observed_mae": float(np.abs(error[observed]).mean()),
        "observed_max_error": float(np.abs(error[observed]).max()),
        "hole_rmse": float(np.sqrt(np.mean(error[~observed] ** 2))),
        "out_of_range_fraction": float(np.mean((image < 0) | (image > 1))),
    }


def batched_sample_jobs(sample, cases, root, plan, stage, batch_size):
    """Vectorize independent public solves; retry a failed batch individually.

    Each solve retains its own random key, constraints, and adaptive solver
    state. Padding the last batch is discarded, never counted as samples.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    batch_sample = jax.jit(jax.vmap(sample))
    compiled = False
    single_compiled = False

    def inputs(chunk):
        key_name = "reference_seed" if stage == "references" else "sample_seed"
        keys = jnp.stack([jax.random.key(c[key_name]) for c in chunk])
        if stage == "references":
            return (keys,)
        references = [load_sample(root, plan, c, "references")[0]
                      for c in chunk]
        indices = [np.flatnonzero(np.broadcast_to(
            gallery.observed_mask(c)[..., None], ref.shape,
        )).astype(np.int32) for c, ref in zip(chunk, references)]
        return keys, jnp.asarray(np.stack(references)), jnp.asarray(indices)

    for start_index in range(0, len(cases), batch_size):
        chunk = cases[start_index:start_index + batch_size]
        padded = chunk + [chunk[-1]] * (batch_size - len(chunk))
        args = inputs(padded)
        start = time.perf_counter()
        try:
            images = np.asarray(batch_sample(*args))
        except Exception:
            # One failed adaptive solve must not discard the other cases.
            for case in chunk:
                args_single = tuple(x[0] for x in inputs([case]))
                single_start = time.perf_counter()
                try:
                    image = np.asarray(sample(*args_single))
                    error = None
                except Exception as exc:
                    image, error = None, exc
                yield (case, image, time.perf_counter() - single_start,
                       not single_compiled, 1, error)
                single_compiled = True
        else:
            elapsed = time.perf_counter() - start
            for case, image in zip(chunk, images):
                yield (case, image, elapsed / len(chunk), not compiled,
                       batch_size, None)
        compiled = True


def generate(root, stage, limit=None, batch_size=1):
    root = Path(root)
    plan = json.loads((root / "plan.json").read_text())
    if file_hash(plan["checkpoint"]) != plan["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed")
    cases = plan["cases"][:limit]
    settings, settings_hashes = study_settings(root, plan)
    if stage not in settings:
        raise ValueError(f"Stage {stage} is not registered in this study")
    pending = []
    for case in cases:
        name = "reference" if stage == "references" else stage
        if (root / f"case_{case['id']}" / f"{name}.json").exists():
            load_sample(root, plan, case, stage)
        else:
            pending.append(case)
    if not pending:
        return
    if stage != "references":
        for case in pending:
            load_sample(root, plan, case, "references")
    importing = (plan["source"]["kind"] == "existing_gallery"
                 and stage in ("references", "pigdm", "ldf"))
    real = plan["source"]["kind"] == "real" and stage == "references"
    model = None
    if not importing and not real:
        model, normalizer = checkpoint.load(plan["checkpoint"])
        if not isinstance(model, JiTImageFlow):
            raise ValueError("Expected native JiTImageFlow")
        if jax.config.jax_use_simplified_jaxpr_constants:
            # JAX 0.9.1 cannot hoist NumPy TypedNdArray constants. Device
            # arrays avoid that compiler bug; the weights are unchanged.
            model.net.p = jax.device_put(model.net.p)
        if np.ndim(normalizer.std) != 0:
            raise ValueError("Residual-norm equivalence needs scalar std")
        cfg = runtime_settings(settings[stage])
        if stage == "references":
            sample = jax.jit(lambda key: generate_unconstrained(
                model, normalizer, num_samples=1, rng=key, **cfg,
            ).x[0])
        else:
            @jax.jit
            def sample(key, reference, indices):
                constraint = imagenet.indexed_pixel_constraint(
                    reference, indices,
                    reduction="pixels" if stage == "ldf" else "norm",
                )
                return methods.get(stage).run(
                    model, normalizer, constraint, num_samples=1,
                    rng=key, **cfg,
                ).x[0]
    compiled = False
    if model is not None and batch_size > 1:
        jobs = batched_sample_jobs(
            sample, pending, root, plan, stage, batch_size,
        )
    else:
        jobs = ((c, None, None, None, 1, None) for c in pending)
    for case, batched_image, elapsed, compilation, actual_batch, error in jobs:
        folder = root / f"case_{case['id']}"
        folder.mkdir(exist_ok=True)
        stem = folder / ("reference" if stage == "references" else stage)
        observed = gallery.observed_mask(case)
        start = time.perf_counter()
        imported_record = None
        try:
            if error is not None:
                raise error
            if batched_image is not None:
                image = batched_image
                if stage == "references":
                    image = np.rint(np.clip(image, 0, 1) * 255).astype(
                        np.float32) / 255
            elif importing:
                old_root = Path(plan["source"]["path"])
                old_plan = json.loads((old_root / "plan.json").read_text())
                if plan_hash(old_plan) != plan["source"]["plan_sha256"]:
                    raise ValueError("Source gallery plan changed")
                old_folder = old_root / f"case_{case['id']}"
                image = np.load(old_folder / stem.with_suffix(".npy").name)
                imported_record = json.loads((
                    old_folder / stem.with_suffix(".json").name
                ).read_text())
                ref, mask = gallery.reference_and_mask(old_root, case)
                gallery.validate_record(imported_record, old_plan, case,
                                        stage, ref, mask, image)
                if stage == "references":
                    image = ref  # exactly the same uint8 reference as before
            elif real:
                if file_hash(case["reference_path"]) != case["source_sha256"]:
                    raise ValueError("Source image changed")
                image, _ = imagenet.reference_and_mask(
                    reference=case["reference_path"],
                )
            elif stage == "references":
                image = np.asarray(sample(
                    jax.random.key(case["reference_seed"]),
                ))
                # Freeze the actual conditioning target at PNG precision.
                image = np.rint(np.clip(image, 0, 1) * 255).astype(
                    np.float32) / 255
            else:
                reference, _ = load_sample(root, plan, case, "references")
                indices = np.flatnonzero(np.broadcast_to(
                    observed[..., None], reference.shape,
                )).astype(np.int32)
                image = np.asarray(sample(
                    jax.random.key(case["sample_seed"]),
                    jnp.asarray(reference), jnp.asarray(indices),
                ))
            if (image.shape != imagenet.IMAGE_SHAPE
                    or not np.isfinite(image).all()):
                raise ValueError("Non-finite or incorrectly shaped sample")
        except Exception as exc:
            gallery.write_json(stem.with_suffix(".failure.json"), {
                "plan_sha256": plan_hash(plan), "case_id": case["id"],
                "stage": stage, "error": str(exc),
                "seconds": time.perf_counter() - start,
            })
            print(f"FAILED {stage} case {case['id']}: {exc}", flush=True)
            continue
        record = {
            "plan_sha256": plan_hash(plan), "case_id": case["id"],
            "stage": stage, "sample_sha256": gallery.digest(image),
            "seconds": (time.perf_counter() - start if elapsed is None
                        else elapsed),
            "includes_compilation": (not compiled and model is not None
                                     if compilation is None else compilation),
            "batch_size": actual_batch,
            "runtime": {
                "jax": jax.__version__,
                "simplified_constants": (
                    jax.config.jax_use_simplified_jaxpr_constants),
            },
            "imported": importing,
            "model": (None if model is None else {
                "class_index": model.label, "cfg": model.cfg,
                "noise_scale": model.noise_scale, "t_eps": model.t_eps,
            }),
        }
        compiled = True
        if stage in settings_hashes:
            record["settings_sha256"] = settings_hashes[stage]
        if importing:
            record["source_record"] = imported_record
            record["seconds"] = imported_record["seconds"]
            record["includes_compilation"] = case["id"] == 1
        if stage != "references":
            reference, _ = load_sample(root, plan, case, "references")
            record["reference_sha256"] = gallery.digest(reference)
            record.update(sample_metrics(image, reference, observed))
        gallery.save_image(stem, image)
        gallery.write_json(stem.with_suffix(".json"), record)
        stem.with_suffix(".failure.json").unlink(missing_ok=True)
        print(json.dumps(record), flush=True)


def main():
    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=(
        "init", "all", "extend", *STAGES, *EXTRA_METHODS, "evaluate",
    ), required=True)
    parser.add_argument("--output-dir",
                        default="results/imagenet_study_256_selected")
    parser.add_argument("--checkpoint", default=imagenet.CHECKPOINT_PATH)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--reference-dir")
    parser.add_argument("--source-gallery")
    parser.add_argument("--pcfm-settings", help="JSON PCFM overrides")
    parser.add_argument("--dps-settings", help="JSON DPS overrides (extend)")
    parser.add_argument("--opc-settings", help="JSON OPC overrides (extend)")
    parser.add_argument("--methods", default=",".join(EXTRA_METHODS),
                        help="Comma-separated baselines to add (extend)")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Independent solves per batch; all keeps adaptive PiGDM at 1",
    )
    parser.add_argument("--limit", type=int,
                        help="Run only the first N planned cases (smoke test)")
    parser.add_argument("--report-dir", default="plots/imagenet_study_256")
    args = parser.parse_args()
    if args.stage == "init":
        initialize(args.output_dir, args.checkpoint, args.count, args.seed,
                   args.reference_dir, args.source_gallery,
                   None if args.pcfm_settings is None else json.loads(
                       Path(args.pcfm_settings).read_text(),
                   ))
    elif args.stage == "extend":
        def overrides(path):
            return None if path is None else json.loads(Path(path).read_text())
        settings = extension_settings(
            overrides(args.dps_settings), overrides(args.opc_settings),
        )
        extend(args.output_dir, {name: settings[name]
                                 for name in args.methods.split(",")})
    elif args.stage == "all":
        root = Path(args.output_dir)
        if not (root / "plan.json").exists():
            raise ValueError("Initialize the study before running all stages")
        status = {"state": "running", "started": time.time(),
                  "completed_stages": [], "report_dir": args.report_dir}
        for stage in (*STAGES, "evaluate"):
            status["stage"] = stage
            gallery.write_json(root / "run_status.json", status)
            manifest_path = root / "run_manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                changed = [name for name, expected in manifest.get(
                    "source_sha256", {},
                ).items() if not Path(name).exists()
                    or file_hash(name) != expected]
                if changed:
                    status.update(state="failed", finished=time.time(),
                                  error=f"Source files changed: {changed}")
                    gallery.write_json(root / "run_status.json", status)
                    raise SystemExit(status["error"])
            result = subprocess.run([
                sys.executable, "-m", "experiments.imagenet_inpainting_study",
                "--stage", stage, "--output-dir", args.output_dir,
                "--report-dir", args.report_dir,
                "--batch-size", str(1 if stage == "pigdm" else args.batch_size),
            ], check=False)
            if result.returncode:
                status.update(state="failed", returncode=result.returncode,
                              finished=time.time())
                gallery.write_json(root / "run_status.json", status)
                raise SystemExit(result.returncode)
            status["completed_stages"].append(stage)
        status.update(state="complete", finished=time.time())
        gallery.write_json(root / "run_status.json", status)
    elif args.stage == "evaluate":
        from experiments.imagenet_inpainting_metrics import evaluate
        evaluate(args.output_dir, args.report_dir)
    else:
        generate(args.output_dir, args.stage, args.limit, args.batch_size)


if __name__ == "__main__":
    main()
