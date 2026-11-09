"""Estimate the OPC posterior-variance table for one problem's checkpoint.

    uv run -m experiments.posterior_variance --problem star

Writes ``experiments/posterior_variance/<problem>.json``, which the
problem's ``opc`` gains point to. Re-run after retraining the checkpoint.
"""

import argparse
import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

import problems
from cfm.core import checkpoint
from cfm.core import posterior_variance as pv

OUT_DIR = Path("experiments/posterior_variance")


def training_samples(problem, max_samples: int) -> jnp.ndarray:
    """The first ``max_samples`` of the problem's training set, in order."""
    loader = DataLoader(problem.make_dataset(), batch_size=1024,
                        shuffle=False, collate_fn=default_collate)
    chunks, count = [], 0
    for batch in loader:
        chunks.append(np.asarray(batch))
        count += len(batch)
        if count >= max_samples:
            break
    return jnp.asarray(np.concatenate(chunks)[:max_samples])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--num-noise", type=int, default=8)
    parser.add_argument("--num-times", type=int, default=101)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    problem = problems.get(args.problem)
    model, normalizer = checkpoint.load(problem.checkpoint_path)
    data = training_samples(problem, args.max_samples)
    data = data.reshape((data.shape[0],) + tuple(model.data_shape))
    ts, r2 = pv.estimate(model, normalizer, data, num_times=args.num_times,
                         num_noise=args.num_noise, seed=args.seed)
    with open(problem.checkpoint_path, "rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    path = OUT_DIR / f"{problem.name}.json"
    pv.save(path, ts, r2, problem=problem.name,
            checkpoint=problem.checkpoint_path, checkpoint_sha256=sha,
            num_data=int(data.shape[0]), num_noise=args.num_noise,
            seed=args.seed)
    print(f"Wrote {path}")
    for t, r in zip(ts[::10], r2[::10]):
        pigdm = (1 - t) ** 2 / (t ** 2 + (1 - t) ** 2)
        print(f"  t={t:.1f}  r2={r:.4f}  (PiGDM prior {pigdm:.4f})")


if __name__ == "__main__":
    main()
