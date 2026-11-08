"""Compare the native-flow baseline and initial-dual study from raw samples."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments import imagenet_inpainting_gallery as gallery

# These are named experimental conditions, not a search over sample seeds.
TRIALS = (
    ("prior_validation/prior_seed2", "Unconstrained · dt=0.002"),
    ("pigdm_refine/pigdm_g06_smallstep", "PiGDM · guidance 0.6, dt=0.0005"),
    ("ldf_sweep/c10p0_r1", "LDF · zero duals\nc=10, p=2, dt=0.002"),
    ("ldf_refine/c40_p1_cancel1_smallstep",
     "LDF · λ₀=−G(x₀)\nc=40, p=1, dt=0.0005"),
)


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def comparison(root, figure_path):
    root = Path(root)
    with Image.open(root / "ldf_refine/reference.png") as image:
        reference = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    with Image.open(root / "ldf_refine/mask.png") as image:
        observed = np.asarray(image.convert("L")) >= 128
    panels = [(reference, "Reference · seed 123"),
              (np.where(observed[..., None], reference, .5), "Observed pixels")]
    expected = {"reference_sha256": digest(reference),
                "mask_sha256": digest(observed), "coordinates": "native",
                "normalizer": {"mean": .5, "std": 1.},
                "projection_iters": 0}
    records = []
    for relative, title in TRIALS:
        stem = root / relative
        record = json.loads(stem.with_suffix(".json").read_text())
        image = np.load(stem.with_suffix(".npy"))
        if not records:
            expected["checkpoint_sha256"] = record["checkpoint_sha256"]
        if (any(record.get(k) != v for k, v in expected.items())
                or record["sample_sha256"] != digest(image)
                or not np.isfinite(image).all()
                or image.shape != reference.shape):
            raise ValueError(f"Inconsistent trial: {stem}")
        records.append(record)
        if record["settings"]["method"] != "unconstrained":
            title += (f"\nknown MAE {record['observed_mae']:.3g}, "
                      f"max {record['observed_max_error']:.3g}")
        panels.append((image, title))
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.95, bottom=.08,
                        wspace=.04, hspace=.28)
    fig.text(.5, .015,
             "Same JiT velocity model and ordinary image state throughout. "
             "Compared samplers share seed 2; the reference uses seed 123.\n"
             "Raw outputs; no observed-pixel pasting or final projection. "
             "This is a tuning case, not a robustness claim.",
             ha="center", fontsize=10)
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    path.with_suffix(".json").write_text(
        json.dumps(records, indent=2) + "\n")


def dog_prior_comparison(root, figure_path):
    """Keep all five paired class-guidance probes, including poor priors."""
    root = Path(root)
    weak = root.parent / "imagenet_dog_gallery"
    strong = root / "dog_prior_cfg23"
    plan = json.loads((weak / "plan.json").read_text())
    if plan["cases"] != gallery.make_cases(plan["experiment_seed"]):
        raise ValueError("Expected all five prior cases")
    records = []
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.6))
    for col, case in enumerate(plan["cases"]):
        reference, observed = gallery.reference_and_mask(weak, case)
        for row, stem in enumerate((
            weak / f"case_{case['id']}/reference",
            strong / f"prior_case_{case['id']}",
        )):
            image = np.load(stem.with_suffix(".npy"))
            record = json.loads(stem.with_suffix(".json").read_text())
            if row == 0:
                gallery.validate_record(record, plan, case, "references",
                                        reference, observed, image)
                model = record["model"]
            else:
                model = {**record, "cfg": record["settings"]["cfg"]}
                if (record["settings"]["seed"] != case["reference_seed"]
                        or record["settings"]["method"] != "unconstrained"
                        or record["checkpoint_sha256"] !=
                        plan["checkpoint_sha256"]
                        or record["sample_sha256"] != digest(image)
                        or not np.isfinite(image).all()):
                    raise ValueError(f"Inconsistent prior: {stem}")
            cfg = 1. if row == 0 else 2.3
            if model["class_index"] != 207 or model["cfg"] != cfg:
                raise ValueError(f"Unexpected class/guidance: {stem}")
            records.append({**record, "source": str(stem)})
            axes[row, col].imshow(np.clip(image, 0, 1))
            axes[row, col].axis("off")
            axes[row, col].set_title(
                f"Seed {case['reference_seed']}\nCFG {cfg:g}", fontsize=10,
            )
    fig.subplots_adjust(left=.01, right=.99, top=.94, bottom=.08,
                        wspace=.02, hspace=.15)
    fig.text(.5, .018,
             "Native JiT-H/32, class 207 (golden retriever). "
             "Same five seeds; no constraints.\n"
             "Midpoint dt=0.002, original time and image state. "
             "All draws retained.", ha="center", fontsize=11)
    path = Path(figure_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    path.with_suffix(".json").write_text(
        json.dumps(records, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/imagenet_native")
    parser.add_argument("--figure")
    parser.add_argument("--dog-priors", action="store_true")
    args = parser.parse_args()
    name = "dog_priors" if args.dog_priors else "comparison"
    figure = args.figure or f"plots/figures/imagenet_native_{name}.png"
    plot = dog_prior_comparison if args.dog_priors else comparison
    plot(args.output_dir, figure)


if __name__ == "__main__":
    main()
