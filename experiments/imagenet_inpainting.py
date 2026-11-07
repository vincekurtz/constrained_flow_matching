"""Reproduce the pretrained image example one validation stage at a time.

    python -m experiments.imagenet_inpainting --stage unconstrained
    python -m experiments.imagenet_inpainting --stage unconstrained-ldf
    python -m experiments.imagenet_inpainting --stage pigdm
    python -m experiments.imagenet_inpainting --stage ldf
    python -m experiments.imagenet_inpainting --stage figure

Inspect each sampling stage before running the next. All three use the same
checkpoint. LDF uses an augmented image-coordinate wrapper; its native
initial noise matches the other samplers when sampling one seed at a time.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cfm import methods
from cfm.core import checkpoint
from cfm.methods.ldf import generate_unconstrained
from cfm.models.jit_image import DataScaleFlow
from problems import imagenet

STAGES = ("unconstrained", "pigdm", "ldf")
LABELS = {"unconstrained": "Unconstrained", "pigdm": "PiGDM", "ldf": "LDF"}


def _identity(checkpoint_path, reference, observed):
    with Path(checkpoint_path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "checkpoint_sha256": digest,
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
    }


def generate(stage, checkpoint_path, output_dir, seeds, options):
    """Call the existing public sampler API; save raw whole-image outputs."""
    reference, observed = imagenet.reference_and_mask(**options)
    identity = _identity(checkpoint_path, reference, observed)
    model, normalizer = checkpoint.load(checkpoint_path)
    if (not isinstance(model, DataScaleFlow)
            or model.data_shape != imagenet.IMAGE_SHAPE):
        raise ValueError("Re-run experiments.prepare_imagenet for the time "
                         "adapter validated with this example")
    unconstrained = stage in ("unconstrained", "unconstrained-ldf")
    coordinates = "ldf" if stage in ("ldf", "unconstrained-ldf") else "native"
    adapter = {"coordinates": coordinates,
               "data_scale": model.data_scale,
               "class_index": model.model.label,
               "cfg": model.model.cfg,
               "noise_scale": model.model.noise_scale,
               "t_eps": model.model.t_eps, "precision": "highest"}
    model, normalizer = imagenet.PROBLEM.model_for(
        "ldf" if coordinates == "ldf" else stage, model, normalizer,
    )
    if coordinates == "ldf":
        adapter["initial_image_scale"] = model.initial_image_scale
    constraint = (None if unconstrained else
                  imagenet.PROBLEM.constraint_for(stage, **options))
    cfg = ({"dt": .01, **imagenet.PROBLEM.unconstrained_gains}
           if unconstrained else
           methods.get(stage).config(**imagenet.PROBLEM.gains_for(stage)))
    folder = Path(output_dir) / stage
    folder.mkdir(parents=True, exist_ok=True)
    def sample(key):
        if unconstrained:
            return generate_unconstrained(
                model, normalizer, num_samples=1, rng=key, **cfg,
            ).x[0]
        return methods.get(stage).run(
            model, normalizer, constraint, num_samples=1, rng=key, **cfg,
        ).x[0]

    sample = jax.jit(sample)
    for seed in seeds:
        start = time.perf_counter()
        image = np.asarray(sample(jax.random.key(seed)))
        if not np.isfinite(image).all():
            raise RuntimeError(f"{stage}, seed {seed}: non-finite image")
        record = {
            **identity, "method": stage, "seed": seed, "settings": cfg,
            "adapter": adapter,
            "seconds": time.perf_counter() - start,
            "min": float(image.min()), "max": float(image.max()),
        }
        if not unconstrained:
            record["constraint"] = constraint.name
            residual = np.abs(image[observed] - reference[observed])
            record.update(observed_max_error=float(residual.max()),
                          observed_mae=float(residual.mean()))
        np.save(folder / f"seed_{seed}.npy", image)
        # Only display quantization; reference pixels are never composited.
        Image.fromarray(np.round(np.clip(image, 0, 1) * 255).astype(
            np.uint8
        )).save(folder / f"seed_{seed}.png")
        (folder / f"seed_{seed}.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        print(json.dumps(record), flush=True)


def comparison(output_dir, seeds, options, figure_path):
    """Assemble saved raw samples, rejecting mismatched experiment inputs."""
    reference, observed = imagenet.reference_and_mask(**options)
    expected = {
        "reference_sha256": hashlib.sha256(reference.tobytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(observed.tobytes()).hexdigest(),
    }
    images, records = {}, []
    for stage in STAGES:
        for seed in seeds:
            stem = Path(output_dir) / stage / f"seed_{seed}"
            record = json.loads(stem.with_suffix(".json").read_text())
            if not records:
                expected["checkpoint_sha256"] = record["checkpoint_sha256"]
            if any(record.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Mismatched experiment inputs: {stem}")
            if record["method"] != stage or record["seed"] != seed:
                raise ValueError(f"Mismatched method/seed: {stem}")
            coordinates = "ldf" if stage == "ldf" else "native"
            if record.get("adapter", {}).get("coordinates") != coordinates:
                raise ValueError(f"Mismatched model coordinates: {stem}")
            image = np.load(stem.with_suffix(".npy"))
            if image.shape != imagenet.IMAGE_SHAPE or not np.isfinite(
                image
            ).all():
                raise ValueError(f"Invalid sample: {stem}")
            images[stage, seed] = image
            records.append(record)

    fig = plt.figure(figsize=(3.6 * (len(seeds) + 1), 10.8))
    grid = fig.add_gridspec(3, len(seeds) + 1, wspace=.03, hspace=.12)
    context = grid[:, 0].subgridspec(2, 1, hspace=.05)
    for i, (im, title) in enumerate([
        (reference, "Reference"),
        (np.where(observed[..., None], reference, .5), "Observed pixels"),
    ]):
        ax = fig.add_subplot(context[i])
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    for row, stage in enumerate(STAGES):
        for col, seed in enumerate(seeds, 1):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(np.clip(images[stage, seed], 0, 1))
            ax.set_title(f"{LABELS[stage]} · seed {seed}", fontsize=11)
            ax.axis("off")
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(.5, .07, "Same pretrained weights; LDF uses image coordinates. "
             "No pixel compositing or final projection.",
             ha="center", fontsize=10)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    path.with_suffix(".json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"Saved {path}")


def main():
    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=(*STAGES, "unconstrained-ldf", "figure"),
        required=True,
    )
    parser.add_argument("--checkpoint", default=imagenet.CHECKPOINT_PATH)
    parser.add_argument("--output-dir", default="results/imagenet")
    parser.add_argument(
        "--figure", default="plots/figures/imagenet_inpainting.png"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[2])
    for flag, kwargs in imagenet.PROBLEM.options.items():
        parser.add_argument(flag, **kwargs)
    args = parser.parse_args()
    options = {flag[2:].replace("-", "_"): getattr(
        args, flag[2:].replace("-", "_")
    ) for flag in imagenet.PROBLEM.options}
    if args.stage == "figure":
        comparison(args.output_dir, args.seeds, options, args.figure)
    else:
        generate(args.stage, args.checkpoint, args.output_dir,
                 args.seeds, options)


if __name__ == "__main__":
    main()
