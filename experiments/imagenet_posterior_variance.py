"""Estimate the OPC posterior-variance table for the pretrained JiT model.

    uv run -m experiments.imagenet_posterior_variance

The JiT checkpoint has no bundled training set, so ``x1`` samples are fresh
unconstrained model samples (the same sampler as the study references,
quantized to 8 bits). Their seeds are drawn from a dedicated generator and
checked against the study and gallery plans, so no inpainting target is
reused. Writes ``experiments/posterior_variance/imagenet.json``.
"""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cfm.core import checkpoint
from cfm.core import posterior_variance as pv
from cfm.methods.ldf import generate_unconstrained
from problems import imagenet

OUTPUT = Path("experiments/posterior_variance/imagenet.json")
EXCLUDED_PLANS = (
    "results/imagenet_study_256_selected/plan.json",
    "results/imagenet_native_gallery/plan.json",
)


def used_seeds():
    seeds = set()
    for path in EXCLUDED_PLANS:
        if Path(path).exists():
            for case in json.loads(Path(path).read_text())["cases"]:
                seeds.update((case["reference_seed"], case["sample_seed"]))
    return seeds


def main():
    jax.config.update("jax_compilation_cache_dir", ".jax_cache")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-data", type=int, default=64)
    parser.add_argument("--num-noise", type=int, default=4)
    parser.add_argument("--num-times", type=int, default=101)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261110)
    args = parser.parse_args()

    model, normalizer = checkpoint.load(imagenet.CHECKPOINT_PATH)
    if jax.config.jax_use_simplified_jaxpr_constants:
        model.net.p = jax.device_put(model.net.p)
    excluded = used_seeds()
    seeds = [s for s in np.random.default_rng(args.seed).choice(
        2**31, size=args.num_data + len(excluded), replace=False,
    ).tolist() if s not in excluded][:args.num_data]

    gains = imagenet.PROBLEM.unconstrained_gains
    sample = jax.jit(jax.vmap(lambda key: generate_unconstrained(
        model, normalizer, num_samples=1, rng=key, save_trajectory=False,
        **gains,
    ).x[0]))
    images = []
    for start in range(0, len(seeds), args.batch_size):
        keys = jnp.stack([jax.random.key(s)
                          for s in seeds[start:start + args.batch_size]])
        batch = np.asarray(sample(keys))
        images.append(np.rint(np.clip(batch, 0, 1) * 255) / 255)
        print(f"generated {start + len(batch)}/{len(seeds)}", flush=True)
    data = jnp.asarray(np.concatenate(images), dtype=jnp.float32)

    ts, r2 = pv.estimate(model, normalizer, data, num_times=args.num_times,
                         num_noise=args.num_noise, batch_size=args.batch_size,
                         seed=args.seed)
    with open(imagenet.CHECKPOINT_PATH, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    pv.save(OUTPUT, ts, r2, problem="imagenet",
            checkpoint=imagenet.CHECKPOINT_PATH, checkpoint_sha256=sha,
            data="unconstrained model samples (8-bit), disjoint seeds",
            data_sampler=gains, data_seeds=seeds,
            num_data=len(seeds), num_noise=args.num_noise, seed=args.seed)
    print(f"Wrote {OUTPUT}")
    for t, r in zip(ts[::10], r2[::10]):
        prior = (1 - t) ** 2 / (t ** 2 + (1 - t) ** 2)
        print(f"  t={t:.1f}  r2={r:.5f}  (PiGDM prior {prior:.5f})")


if __name__ == "__main__":
    main()
