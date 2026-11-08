"""Controlled solver/gain/dual-initialization experiments on plain JiT.

A JSON plan is a list of jobs with unique ``name`` and ``method`` fields.
All jobs use the same reference/mask and raw whole-image sampler output.
No coordinate changes, time changes, final projection, or compositing.
Nonzero initial duals and solver controls are passed through the public
sampler APIs. The primal-dual and PiGDM vector fields are unchanged.
"""

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cfm import methods
from cfm.core import checkpoint
from cfm.methods import ldf
from cfm.models.jit_image import JiTImageFlow
from problems import imagenet

SOLVERS = {"midpoint": diffrax.Midpoint, "heun": diffrax.Heun,
           "euler": diffrax.Euler, "tsit5": diffrax.Tsit5}


def initialized_ldf(model, normalizer, constraint, key, *, dt,
                    penalty_weight, rescale_factor, rescale_exponent,
                    dual_cancel, solver, stepsize_controller=None):
    """Original LDF equations with lambda(0) = -dual_cancel * G(x(0)).

    G is the weighted fixed-pixel residual. dual_cancel=1 cancels only
    the initial quadratic-penalty force; it changes neither the Gaussian
    initial image nor any subsequent primal/dual equation. Zero recovers
    the public sampler. Keep its last-saved-time convention (1-dt).
    """
    result = ldf.generate(
        model, normalizer, constraint, num_samples=1, rng=key, dt=dt,
        penalty_weight=penalty_weight, rescale_factor=rescale_factor,
        rescale_exponent=rescale_exponent, initial_dual_scale=-dual_cancel,
        solver=solver, stepsize_controller=stepsize_controller,
        save_trajectory=False,
    )
    return result.x[0], result.num_steps


def resolved(job):
    """Record every setting, including defaults, before running a job."""
    method = job["method"]
    if method not in ("unconstrained", "pigdm", "ldf"):
        raise ValueError(f"Unknown method: {method}")
    cfg = dict(name=job["name"], method=method, seed=2, dt=.002,
               solver="midpoint", cfg=1., t_eps=.05)
    if method == "ldf":
        cfg.update(penalty_weight=10., rescale_factor=1.,
                   rescale_exponent=2., dual_cancel=0., rtol=None, atol=None)
    elif method == "pigdm":
        cfg.update(guidance_scale=.6, eps_reg=1e-8, rtol=None, atol=None,
                   max_steps=4096)
    unknown = set(job) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg.update(job)
    if cfg["solver"] not in SOLVERS:
        raise ValueError("Unknown solver")
    if not 0 < cfg["dt"] < 1:
        raise ValueError("dt must be between 0 and 1")
    if method in ("ldf", "pigdm") and (
            (cfg["rtol"] is None) != (cfg["atol"] is None)):
        raise ValueError("Specify both rtol and atol")
    if (Path(cfg["name"]).name != cfg["name"]
            or "." in cfg["name"] or not cfg["name"]):
        raise ValueError("Job name must be a filename")
    return cfg


def compile_sampler(model, normalizer, constraint, cfg):
    method, dt = cfg["method"], cfg["dt"]
    model.cfg, model.t_eps = cfg["cfg"], cfg["t_eps"]
    solver = SOLVERS[cfg["solver"]]()
    controller = (None if cfg.get("rtol") is None else
                  diffrax.PIDController(rtol=cfg["rtol"], atol=cfg["atol"]))
    if method == "unconstrained":
        def run(key, params):
            result = ldf.generate_unconstrained(
                model, normalizer, num_samples=1, rng=key, dt=dt,
                solver=solver, save_trajectory=False,
            )
            return result.x[0], result.num_steps
    elif method == "pigdm":
        def run(key, params):
            result = methods.PIGDM.run(
                model, normalizer, constraint, num_samples=1, rng=key,
                dt=dt, guidance_scale=params[0], eps_reg=params[1],
                solver=solver, stepsize_controller=controller,
                max_steps=cfg["max_steps"], save_trajectory=False,
            )
            # The legacy PiGDM tuple API does not expose Diffrax statistics.
            steps = round(1 / dt) if controller is None else -1
            return result.x[0], jnp.asarray(steps)
    else:
        def run(key, params):
            return initialized_ldf(
                model, normalizer, constraint, key, dt=dt,
                penalty_weight=params[0], rescale_factor=params[1],
                rescale_exponent=params[2], dual_cancel=params[3],
                solver=solver, stepsize_controller=controller,
            )
    return jax.jit(run)


def run_plan(plan_path, output_dir, checkpoint_path, options):
    jobs = [resolved(job) for job in json.loads(Path(plan_path).read_text())]
    if len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate job names")
    reference, observed = imagenet.reference_and_mask(**options)
    with Path(checkpoint_path).open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    identity = {
        "checkpoint_sha256": checkpoint_hash,
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
        "coordinates": "native", "normalizer": {"mean": .5, "std": 1.},
        "projection_iters": 0,
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    plan = {**identity, "jobs": jobs}
    plan_file = root / "plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError("Existing plan differs; use a new output directory")
    plan_file.write_text(json.dumps(plan, indent=2) + "\n")
    Image.fromarray(np.rint(reference * 255).astype(np.uint8)).save(
        root / "reference.png")
    Image.fromarray(observed.astype(np.uint8) * 255).save(root / "mask.png")
    Image.fromarray(np.rint(np.where(observed[..., None], reference, .5)
                           * 255).astype(np.uint8)).save(root / "masked.png")
    model, normalizer = checkpoint.load(checkpoint_path)
    if not isinstance(model, JiTImageFlow):
        raise ValueError("Prepare the plain JiT velocity checkpoint first")
    if (float(normalizer.mean) != .5 or float(normalizer.std) != 1.
            or model.noise_scale != 2.):
        raise ValueError("Expected native 512px normalization")
    identity.update(class_index=model.label, noise_scale=model.noise_scale)
    signature, sample = None, None
    for cfg in jobs:
        stem = root / cfg["name"]
        if (stem.with_suffix(".json").exists()
                or stem.with_suffix(".failure.json").exists()):
            raise ValueError(f"Refusing to overwrite existing result: {stem}")
        current = tuple(cfg.get(k) for k in (
            "method", "dt", "solver", "cfg", "t_eps", "rtol", "atol",
            "max_steps"))
        if current != signature:
            sample = None
            gc.collect()
            jax.clear_caches()
            constraint = imagenet.fixed_pixel_constraint(
                reference, observed,
                reduction="norm" if cfg["method"] == "pigdm" else "pixels",
            )
            sample = compile_sampler(model, normalizer, constraint, cfg)
            signature = current
        keys = (("guidance_scale", "eps_reg") if cfg["method"] == "pigdm"
                else ("penalty_weight", "rescale_factor", "rescale_exponent",
                      "dual_cancel"))
        params = jnp.array([cfg.get(k, 0.) for k in keys], dtype=jnp.float32)
        start = time.perf_counter()
        try:
            image, steps = sample(jax.random.key(cfg["seed"]), params)
            image = np.asarray(image)
        except Exception as error:
            if "maximum number of solver steps" not in str(error):
                raise
            record = {**identity, "settings": cfg, "finite": False,
                      "status": "solver_step_limit",
                      "max_steps": cfg.get("max_steps", 10000),
                      "seconds": time.perf_counter() - start}
            stem.with_suffix(".failure.json").write_text(
                json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            continue
        finite = bool(np.isfinite(image).all())
        record = {**identity, "settings": cfg, "finite": finite,
                  "seconds": time.perf_counter() - start,
                  "steps": int(steps) if int(steps) >= 0 else None,
                  "sample_sha256": hashlib.sha256(image.tobytes()).hexdigest()}
        np.save(stem.with_suffix(".npy"), image)
        if finite:
            residual = np.abs(image[observed] - reference[observed])
            record.update(observed_mae=float(residual.mean()),
                          observed_max_error=float(residual.max()),
                          min=float(image.min()), max=float(image.max()))
            Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(
                np.uint8)).save(stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(
            json.dumps(record, indent=2) + "\n")
        print(json.dumps(record), flush=True)


def plot_plan(output_dir, figure_path):
    """Show every planned trial, including failed runs, with its raw errors."""
    root = Path(output_dir)
    plan = json.loads((root / "plan.json").read_text())
    with Image.open(root / "reference.png") as image:
        reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    with Image.open(root / "mask.png") as image:
        observed = np.asarray(image.convert("L")) >= 128
    for name, array in (("reference", reference), ("mask", observed)):
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        if digest != plan[name + "_sha256"]:
            raise ValueError(f"Changed {name}")
    panels = [(reference, "Reference"),
              (np.where(observed[..., None], reference, .5), "Observed pixels")]
    records = []
    for cfg in plan["jobs"]:
        stem = root / cfg["name"]
        failed = stem.with_suffix(".failure.json").exists()
        record = json.loads(stem.with_suffix(
            ".failure.json" if failed else ".json").read_text())
        if (record["settings"] != cfg or
                any(record[k] != plan[k] for k in plan if k != "jobs")):
            raise ValueError(f"Changed trial: {stem}")
        records.append(record)
        title = cfg["name"]
        if failed:
            image = np.full(reference.shape, .5)
            title += "\nSOLVER STEP LIMIT (no sample)"
        else:
            image = np.load(stem.with_suffix(".npy"))
            if (hashlib.sha256(image.tobytes()).hexdigest()
                    != record["sample_sha256"]):
                raise ValueError(f"Changed sample: {stem}")
        if not failed and record["finite"]:
            title += (f"\nknown MAE {record['observed_mae']:.3g}, "
                      f"max {record['observed_max_error']:.3g}")
        elif not failed:
            title += "\nNON-FINITE OUTPUT"
            image = np.full(reference.shape, .5)
        panels.append((image, title))
    columns = min(4, math.ceil(math.sqrt(len(panels))))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4 * columns, 4.4 * rows))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, .025, 1, 1))
    fig.text(.5, .008, "Plain JiT velocity; all planned trials retained. "
             "No pixel pasting or final projection.", ha="center")
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    path.with_suffix(".json").write_text(
        json.dumps({"plan": plan, "records": records}, indent=2) + "\n")


def main():
    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--figure",
                        help="Plot all saved jobs instead of sampling")
    parser.add_argument("--checkpoint", default=imagenet.CHECKPOINT_PATH)
    for flag, kwargs in imagenet.PROBLEM.options.items():
        parser.add_argument(flag, **kwargs)
    args = parser.parse_args()
    if args.figure:
        plot_plan(args.output_dir, args.figure)
        return
    if args.plan is None:
        parser.error("--plan is required for sampling")
    options = {flag[2:].replace("-", "_"): getattr(
        args, flag[2:].replace("-", "_"))
        for flag in imagenet.PROBLEM.options}
    run_plan(args.plan, args.output_dir, args.checkpoint, options)


if __name__ == "__main__":
    main()
