"""Five unselected random references and masks using the committed settings.

Run stages in order: references, pigdm, ldf, figure. Every planned case is
included, including artifacts. No per-case tuning or output compositing.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from cfm import methods
from cfm.core import checkpoint
from cfm.methods.ldf import generate_unconstrained
from cfm.models.jit_image import DataScaleFlow
from problems import imagenet

STAGES = ("references", "pigdm", "ldf", "figure")


def make_cases(seed, count=5, size=512, hole=192, margin=32):
    """Draw references, independent sampler seeds, and interior square masks."""
    if count < 1 or hole < 1 or size - hole < 2 * margin:
        raise ValueError("Invalid case count or mask geometry")
    rng = np.random.default_rng(seed)
    seeds = rng.choice(2**31, size=2 * count, replace=False).tolist()
    positions = set()
    cases = []
    available = (size - hole - 2 * margin + 1)**2
    if count > available:
        raise ValueError("Too many cases for distinct mask positions")
    for i in range(count):
        while True:
            top, left = rng.integers(margin, size - hole - margin + 1, 2)
            if (int(top), int(left)) not in positions:
                positions.add((int(top), int(left)))
                break
        cases.append({
            "id": i + 1, "reference_seed": seeds[i],
            "sample_seed": seeds[count + i],
            "hole": [int(top), int(left), hole, hole],
        })
    return cases


def observed_mask(case, size=512):
    top, left, height, width = case["hole"]
    observed = np.ones((size, size), dtype=bool)
    observed[top:top + height, left:left + width] = False
    return observed


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def save_image(stem, image):
    """Save raw model output alongside an 8-bit RGB PNG."""
    np.save(stem.with_suffix(".npy"), image)
    Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(
        np.uint8
    )).save(stem.with_suffix(".png"))


def settings():
    return {
        "references": dict(imagenet.PROBLEM.unconstrained_gains),
        **{name: methods.get(name).config(
            **imagenet.PROBLEM.gains_for(name)
        ) for name in ("pigdm", "ldf")},
    }


def initialize(root, checkpoint_path, seed):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with Path(checkpoint_path).open("rb") as stream:
        checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    plan = {
        "experiment_seed": seed, "checkpoint_sha256": checkpoint_hash,
        "selection": "All five cases drawn before sampling; no filtering",
        "image_shape": list(imagenet.IMAGE_SHAPE),
        "settings": settings(), "cases": make_cases(seed),
    }
    path = root / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError(
                "Existing plan differs; use a new output directory"
            )
    else:
        write_json(path, plan)
    return plan


def reference_and_mask(root, case):
    folder = Path(root) / f"case_{case['id']}"
    with Image.open(folder / "reference.png") as image:
        reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if reference.shape != imagenet.IMAGE_SHAPE:
        raise ValueError(f"Invalid reference shape in {folder}")
    return reference, observed_mask(case)


def validate_record(record, plan, case, stage, reference, observed, image):
    expected = {
        "checkpoint_sha256": plan["checkpoint_sha256"],
        "case": case, "method": stage, "settings": plan["settings"][stage],
        "reference_sha256": digest(reference),
        "mask_sha256": digest(observed),
        "coordinates": "ldf" if stage == "ldf" else "native",
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Mismatched inputs for case {case['id']}, {stage}")
    if image.shape != imagenet.IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"Invalid sample for case {case['id']}, {stage}")
    if record.get("sample_sha256") != digest(image):
        raise ValueError(f"Sample hash mismatch for case {case['id']}, {stage}")


def generate(stage, root, checkpoint_path, seed):
    plan = initialize(root, checkpoint_path, seed)
    model, normalizer = checkpoint.load(checkpoint_path)
    if (not isinstance(model, DataScaleFlow)
            or model.data_shape != imagenet.IMAGE_SHAPE):
        raise ValueError("Expected the prepared JiT image checkpoint")
    model, normalizer = imagenet.PROBLEM.model_for(stage, model, normalizer)
    cfg = plan["settings"][stage]
    if stage == "references":
        sample = jax.jit(lambda key: generate_unconstrained(
            model, normalizer, num_samples=1, rng=key, **cfg,
        ).x[0])
    else:
        method = methods.get(stage)

        @jax.jit
        def sample(key, reference, indices):
            constraint = imagenet.indexed_pixel_constraint(
                reference, indices,
                reduction="norm" if stage == "pigdm" else "pixels",
            )
            return method.run(
                model, normalizer, constraint, num_samples=1, rng=key, **cfg,
            ).x[0]

    for case in plan["cases"]:
        folder = Path(root) / f"case_{case['id']}"
        folder.mkdir(parents=True, exist_ok=True)
        name = "reference" if stage == "references" else stage
        stem = folder / name
        start = time.perf_counter()
        if stage == "references":
            image = np.asarray(sample(jax.random.key(case["reference_seed"])))
        else:
            reference, observed = reference_and_mask(root, case)
            validate_record(
                json.loads((folder / "reference.json").read_text()),
                plan, case, "references", reference, observed,
                np.load(folder / "reference.npy"),
            )
            indices = np.flatnonzero(np.broadcast_to(
                observed[..., None], reference.shape,
            )).astype(np.int32)
            image = np.asarray(sample(
                jax.random.key(case["sample_seed"]),
                jnp.asarray(reference), jnp.asarray(indices),
            ))
        if not np.isfinite(image).all():
            raise RuntimeError(
                f"Non-finite output in case {case['id']}, {stage}"
            )
        seconds = time.perf_counter() - start
        save_image(stem, image)
        if stage == "references":
            reference, observed = reference_and_mask(root, case)
            Image.fromarray(observed.astype(np.uint8) * 255).save(
                folder / "mask.png"
            )
            Image.fromarray(np.rint(np.where(
                observed[..., None], reference, .5,
            ) * 255).astype(np.uint8)).save(folder / "masked.png")
        record = {
            "checkpoint_sha256": plan["checkpoint_sha256"],
            "case": case, "method": stage, "settings": cfg,
            "coordinates": "ldf" if stage == "ldf" else "native",
            "reference_sha256": digest(reference),
            "mask_sha256": digest(observed), "sample_sha256": digest(image),
            "seconds": seconds, "min": float(image.min()),
            "max": float(image.max()),
        }
        if stage != "references":
            error = np.abs(image[observed] - reference[observed])
            record.update(observed_mae=float(error.mean()),
                          observed_max_error=float(error.max()))
        write_json(stem.with_suffix(".json"), record)
        print(json.dumps(record), flush=True)


def comparison(root, figure_path):
    plan = json.loads((Path(root) / "plan.json").read_text())
    if plan["cases"] != make_cases(plan["experiment_seed"]):
        raise ValueError("Figure must include all five planned random cases")
    fig, axes = plt.subplots(len(plan["cases"]), 4, squeeze=False,
                             figsize=(14, 3.65 * len(plan["cases"])))
    records = []
    for row, case in enumerate(plan["cases"]):
        folder = Path(root) / f"case_{case['id']}"
        reference, observed = reference_and_mask(root, case)
        ref_record = json.loads((folder / "reference.json").read_text())
        validate_record(ref_record, plan, case, "references", reference,
                        observed, np.load(folder / "reference.npy"))
        records.append(ref_record)
        panels = [reference, np.where(observed[..., None], reference, .5)]
        for stage in ("pigdm", "ldf"):
            image = np.load(folder / f"{stage}.npy")
            record = json.loads((folder / f"{stage}.json").read_text())
            validate_record(record, plan, case, stage, reference,
                            observed, image)
            records.append(record)
            panels.append(image)
        for col, image in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(np.clip(image, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                title = ("Reference", "Masked reference", "PiGDM", "LDF")[col]
                ax.set_title(title, fontsize=16)
        axes[row, 0].set_ylabel(f"Case {case['id']}", fontsize=14)
        axes[row, 0].set_xlabel(f"Reference seed {case['reference_seed']}",
                                fontsize=8)
        for col, record in zip((2, 3), records[-2:]):
            axes[row, col].set_xlabel(
                f"Known MAE {record['observed_mae']:.2g} · "
                f"max {record['observed_max_error']:.2g}", fontsize=9,
            )
    fig.subplots_adjust(left=.045, right=.995, top=.97, bottom=.055,
                        wspace=.025, hspace=.13)
    fig.text(.5, .012, "Five random cases, no selection or per-case tuning. "
             "Same pretrained weights; LDF uses image coordinates.\n"
             "Gray regions are missing. Raw sampler outputs; no pixel "
             "compositing or final projection.", ha="center", fontsize=10)
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    write_json(path.with_suffix(".json"), {"plan": plan, "records": records})
    print(f"Saved {path}")


def main():
    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--checkpoint", default=imagenet.CHECKPOINT_PATH)
    parser.add_argument("--output-dir", default="results/imagenet_gallery")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--figure",
                        default="plots/figures/imagenet_inpainting_five.png")
    args = parser.parse_args()
    if args.stage == "figure":
        comparison(args.output_dir, args.figure)
    else:
        generate(args.stage, args.output_dir, args.checkpoint, args.seed)


if __name__ == "__main__":
    main()
