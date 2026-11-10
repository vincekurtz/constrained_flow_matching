"""Clean-FID features, distribution metrics, tables, and unselected figures."""

import csv
import importlib.metadata
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from experiments import imagenet_inpainting_gallery as gallery
from experiments.imagenet_inpainting_study import (
    load_sample, plan_hash, sample_metrics, study_methods,
)


def frechet_distance(x, y):
    """Exact empirical FID using thin covariance factors (also for n < d).

    The covariance square-root trace is the nuclear norm of X @ Y.T
    for centered, variance-normalized feature matrices. Avoids a singular
    2048x2048 matrix square root for small pilot studies.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("Expected feature matrices with equal feature width")
    if min(len(x), len(y)) < 2:
        raise ValueError("FID needs at least two samples per distribution")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Features must be finite")
    mx, my = x.mean(0), y.mean(0)
    a = (x - mx) / np.sqrt(len(x) - 1)
    b = (y - my) / np.sqrt(len(y) - 1)
    if max(len(x), len(y)) <= x.shape[1]:
        trace = np.linalg.svd(a @ b.T, compute_uv=False).sum()
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(a.T @ a)
        sqrt_a = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0)))
        sqrt_a = sqrt_a @ eigenvectors.T
        middle = sqrt_a @ (b.T @ b) @ sqrt_a
        trace = np.sqrt(np.maximum(np.linalg.eigvalsh(
            (middle + middle.T) / 2,
        ), 0)).sum()
    return float(max(0., np.sum((mx - my)**2)
                     + np.sum(a*a) + np.sum(b*b) - 2 * trace))


def paired_kid(x, y):
    """Polynomial MMD, excluding matched pairs from all kernel sums.

    Reference/output features are paired, not independent samples. Dropping
    the cross diagonal avoids dependence bias from each conditioning image.
    Values can be negative, as for other unbiased MMD estimates.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError("Paired KID needs equal feature matrices with n >= 2")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Features must be finite")
    k = (x @ x.T / x.shape[1] + 1)**3
    k += (y @ y.T / x.shape[1] + 1)**3
    k -= 2 * (x @ y.T / x.shape[1] + 1)**3
    return float((k.sum() - np.trace(k)) / (len(x) * (len(x) - 1)))


LABELS = {"pcfm": "PCFM", "pigdm": "PiGDM", "opc": "OPC", "dps": "DPS",
          "ldf": "LDF"}


def png_array(image):
    return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)


def extract_features(root, plan, cases, region, stage, model):
    import torch
    from cleanfid import fid

    name = "reference" if stage == "references" else stage
    folder = root / "evaluation_images" / region / name
    folder.mkdir(parents=True, exist_ok=True)
    identities = []
    for case in cases:
        image, record = load_sample(root, plan, case, stage)
        identities.append(record["sample_sha256"])
        if region == "hole":
            top, left, height, width = case["hole"]
            image = image[top:top + height, left:left + width]
        Image.fromarray(png_array(image)).save(folder / f"{case['id']:06d}.png")
    expected = {"plan_sha256": plan_hash(plan), "hashes": identities,
                "case_ids": [c["id"] for c in cases], "region": region,
                "cleanfid": importlib.metadata.version("clean-fid"),
                "mode": "clean", "features": "inception_v3_pool3_2048"}
    cache = folder / "features.npz"
    metadata = folder / "features.json"
    if cache.exists() and metadata.exists():
        if json.loads(metadata.read_text()) == expected:
            features = np.load(cache)["features"]
            if features.shape == (len(cases), 2048) and np.isfinite(
                    features).all():
                return features
    # Refuse extra files left behind by a different evaluation population.
    if len(list(folder.glob("*.png"))) != len(cases):
        raise ValueError(f"Unexpected evaluation images in {folder}")
    features = fid.get_folder_features(
        str(folder), model=model, num_workers=0, batch_size=8,
        device=torch.device("cpu"), mode="clean", verbose=False,
    )
    if features.shape != (len(cases), 2048) or not np.isfinite(features).all():
        raise ValueError("Invalid Inception features")
    np.savez(cache, features=features)
    gallery.write_json(metadata, expected)
    return features


def figure(root, plan, output):
    names = study_methods(root, plan)
    cases = [c for c in plan["cases"]
             if c["id"] in plan["representative_ids"]]
    fig, axes = plt.subplots(len(cases), 2 + len(names), squeeze=False,
                             figsize=(3.5 * (2 + len(names)),
                                      3.6 * len(cases)))
    for row, case in enumerate(cases):
        try:
            reference, _ = load_sample(root, plan, case, "references")
        except FileNotFoundError:
            for ax in axes[row]:
                ax.text(.5, .5, "REFERENCE FAILED", ha="center")
                ax.axis("off")
            continue
        observed = gallery.observed_mask(case)
        images = [reference, np.where(observed[..., None], reference, .5)]
        records = []
        for method in names:
            try:
                image, record = load_sample(root, plan, case, method)
            except FileNotFoundError:
                image, record = np.full_like(reference, .5), None
            images.append(image)
            records.append(record)
        for col, image in enumerate(images):
            ax = axes[row, col]
            ax.imshow(np.clip(image, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(("Reference", "Observed", *(
                    LABELS[m] for m in names))[col], fontsize=16)
            if col >= 2:
                record = records[col - 2]
                ax.set_xlabel("FAILED / MISSING" if record is None else
                              f"Known MAE {record['observed_mae']:.2g}")
        axes[row, 0].set_ylabel(f"Case {case['id']}")
    fig.suptitle("First five planned cases, without selection", fontsize=17)
    fig.text(.5, .005, "Whole-image outputs, no compositing. PCFM uses its "
             "built-in per-step projection. Display clipped to [0, 1].",
             ha="center")
    fig.tight_layout(rect=(0, .02, 1, .98))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def evaluate(root, report_dir):
    root, report_dir = Path(root), Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    plan = json.loads((root / "plan.json").read_text())
    names = study_methods(root, plan)
    # Require every planned job to have either a valid output or a failure.
    records, successes = {}, {}
    for stage in ("references", *names):
        records[stage], successes[stage] = {}, set()
        for case in plan["cases"]:
            try:
                image, record = load_sample(root, plan, case, stage)
            except FileNotFoundError:
                name = "reference" if stage == "references" else stage
                failure = root / f"case_{case['id']}" / f"{name}.failure.json"
                if not failure.exists():
                    raise ValueError(f"Study incomplete: {stage}, {case['id']}")
                failed = json.loads(failure.read_text())
                if (failed.get("plan_sha256") != plan_hash(plan)
                        or failed.get("case_id") != case["id"]
                        or failed.get("stage") != stage):
                    raise ValueError("Mismatched failure record")
                continue
            if stage != "references":
                reference, _ = load_sample(root, plan, case, "references")
                record.update(sample_metrics(
                    image, reference, gallery.observed_mask(case),
                ))
            records[stage][case["id"]] = record
            successes[stage].add(case["id"])
    complete = set.intersection(*successes.values())
    cases = [c for c in plan["cases"] if c["id"] in complete]
    if len(cases) < 2:
        raise ValueError("Need at least two jointly successful cases for FID")
    import torch
    from cleanfid import fid

    torch.set_num_threads(4)
    model = fid.build_feature_extractor(
        "clean", device=torch.device("cpu"), use_dataparallel=False,
    )
    rows = []
    distributions = {m: {} for m in names}
    for region in ("full", "hole"):
        reference_features = extract_features(
            root, plan, cases, region, "references", model,
        )
        for method in names:
            features = extract_features(
                root, plan, cases, region, method, model,
            )
            distributions[method][f"fid_{region}"] = frechet_distance(
                reference_features, features,
            )
            distributions[method][f"paired_kid_{region}"] = paired_kid(
                reference_features, features,
            )
    for method in names:
        paired = [records[method][c["id"]] for c in cases]
        timings = [r["seconds"] for r in records[method].values()
                   if not r["includes_compilation"]]
        rows.append({
            "method": method, "planned": len(plan["cases"]),
            "successful": len(successes[method]), "paired_n": len(cases),
            **distributions[method],
            "observed_mae": float(np.mean([r["observed_mae"] for r in paired])),
            "observed_max_error": max(r["observed_max_error"] for r in paired),
            "hole_rmse": float(np.mean([r["hole_rmse"] for r in paired])),
            "out_of_range_fraction": float(np.mean([
                r["out_of_range_fraction"] for r in paired
            ])),
            "seconds_median": float(np.median(timings)) if timings else None,
        })
    report = {
        "plan": plan, "paired_case_ids": sorted(complete), "rows": rows,
        "role": ("calibration" if plan["source"]["kind"] == "existing_gallery"
                 else "evaluation"),
        "excluded_case_ids": sorted(set(c["id"] for c in plan["cases"])
                                    - complete),
        "notes": [
            "Synthetic sources measure model consistency, not real-data FID.",
            "Finite-sample FID is biased; compare identical sample counts.",
            "FID uses the common successful subset; failures are reported.",
            "Paired KID excludes matched cross pairs; may be negative.",
            "FID uses clipped uint8 images; pixel errors use raw floats.",
            "Hole FID uses the 192px missing crop with Clean-FID resizing.",
            "Runtime excludes compilation; budgets differ across methods.",
            "Original conditioning references also form the FID reference set.",
        ],
    }
    if len(cases) < 100:
        report["notes"].append(
            "Small-sample diagnostic only; not a reliable FID ranking.")
    gallery.write_json(report_dir / "results.json", report)
    with (report_dir / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"Inpainting study: {plan['source']['kind']} references",
             "", f"Paired evaluation: {len(cases)}/{len(plan['cases'])} cases.",
             "", "| Method | Success | FID full ↓ | FID hole ↓ | "
             "KID hole ×1000 ↓ | Known MAE ↓ | Hole RMSE ↓ | s/image |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        seconds = row['seconds_median']
        timing = "—" if seconds is None else f"{seconds:.1f}"
        lines.append(
            f"| {row['method']} | {row['successful']}/{row['planned']} | "
            f"{row['fid_full']:.2f} | {row['fid_hole']:.2f} | "
            f"{1000 * row['paired_kid_hole']:.2f} | "
            f"{row['observed_mae']:.5f} | {row['hole_rmse']:.4f} | {timing} |"
        )
    lines += ["", *[f"- {note}" for note in report["notes"]], ""]
    (report_dir / "results.md").write_text("\n".join(lines))
    figure(root, plan, report_dir / "representatives.png")
    print("\n".join(lines), flush=True)
