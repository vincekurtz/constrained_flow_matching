"""Shared helpers for the ImageNet inpainting study.

Case and mask generation, record hashing and validation, image saving,
and the default PiGDM/LDF settings used by ``imagenet_inpainting_study``.
"""

import hashlib
import json
from pathlib import Path

import diffrax
import numpy as np
from PIL import Image

from cfm import methods
from problems import imagenet

SOLVERS = {"midpoint": diffrax.Midpoint, "heun": diffrax.Heun,
           "euler": diffrax.Euler, "tsit5": diffrax.Tsit5}


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


def settings(ldf_overrides=None, pigdm_overrides=None):
    result = {
        "references": dict(imagenet.PROBLEM.unconstrained_gains),
        **{name: methods.get(name).config(
            **imagenet.PROBLEM.gains_for(name)
        ) for name in ("pigdm", "ldf")},
    }
    if ldf_overrides:
        allowed = {"dt", "penalty_weight", "rescale_factor",
                   "rescale_exponent", "initial_dual_scale"}
        if set(ldf_overrides) - allowed:
            raise ValueError("Unsupported LDF gallery setting")
        result["ldf"].update(ldf_overrides)
    if pigdm_overrides:
        allowed = {"dt", "guidance_scale", "eps_reg", "solver", "rtol",
                   "atol", "max_steps"}
        if set(pigdm_overrides) - allowed:
            raise ValueError("Unsupported PiGDM gallery setting")
        if (pigdm_overrides.get("rtol") is None) != (
                pigdm_overrides.get("atol") is None):
            raise ValueError("Specify both rtol and atol")
        if pigdm_overrides.get("solver", "midpoint") not in SOLVERS:
            raise ValueError("Unknown PiGDM solver")
        result["pigdm"].update(pigdm_overrides)
    return result


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
        "coordinates": "native",
    }
    if any(record.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Mismatched inputs for case {case['id']}, {stage}")
    if image.shape != imagenet.IMAGE_SHAPE or not np.isfinite(image).all():
        raise ValueError(f"Invalid sample for case {case['id']}, {stage}")
    if record.get("sample_sha256") != digest(image):
        raise ValueError(f"Sample hash mismatch for case {case['id']}, {stage}")
