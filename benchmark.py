"""Benchmark constrained generation algorithms one sample at a time.

Loads a pretrained model (star or MNIST), JIT-compiles a single-sample
generator for the chosen method, and reports per-sample wall-clock time
and constraint violation.

Usage examples:

    python benchmark.py --example star --method pcfm --num-samples 20
    python benchmark.py --example star --method pigdm --num-samples 20 \\
        --guidance-scale 1.0 --eps-reg 1e-4
    python benchmark.py --example mnist --method ours --num-samples 5 \\
        --penalty-weight 10.0 --rescale-factor 1.0
"""

import argparse
import json
import time
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp

from datasets.mnist import MNISTDataset
from generation import generate_constrained
from pcfm import generate_pcfm
from pi_gdm import generate_pigdm


def build_constraint(example: str):
    """Return (constraint_fn, scalar_violation_fn) for the given example.

    The constraint_fn operates on a single *unnormalized* sample and matches
    the convention used by the generation algorithms. The violation_fn maps
    a single sample to a scalar magnitude for reporting.
    """
    if example == "star":
        def unit_circle_constraint(x):
            return jnp.sum(x**2, axis=-1) - 1.0

        def violation_fn(x):
            return jnp.abs(unit_circle_constraint(x))

        return unit_circle_constraint, violation_fn

    if example == "mnist":
        # Top-half inpainting on a fixed reference digit, identical to the
        # constraint used in examples/mnist.py.
        ds = MNISTDataset(train=False, digit=5)
        reference = jnp.array(ds[0])  # (28, 28, 1)
        mask = jnp.zeros((28, 28, 1), dtype=bool).at[:14, :, :].set(True)
        observed_indices = jnp.where(mask.ravel())[0]
        y = reference.ravel()[observed_indices]
        n_pixels = 28 * 28 * 1
        A = jnp.eye(n_pixels)[observed_indices]

        def inpainting_constraint(x):
            return A @ x.ravel() - y

        def violation_fn(x):
            return jnp.max(jnp.abs(inpainting_constraint(x)))

        return inpainting_constraint, violation_fn

    raise ValueError(f"Unknown example: {example}")


def build_generator(method: str, model, normalizer, constraint_fn, args):
    """JIT-compile a function ``rng -> single_sample`` for the chosen method."""
    if method == "ours":
        def _gen(rng):
            x, _ = generate_constrained(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                dt=args.dt,
                rng=rng,
                penalty_weight=args.penalty_weight,
                rescale_factor=args.rescale_factor,
                rescale_exponent=args.rescale_exponent,
            )
            return x[0]
        return jax.jit(_gen)

    if method == "pigdm":
        def _gen(rng):
            x, _ = generate_pigdm(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                dt=args.dt,
                rng=rng,
                guidance_scale=args.guidance_scale,
                eps_reg=args.eps_reg,
            )
            return x[0]
        return jax.jit(_gen)

    if method == "pcfm":
        def _gen(rng):
            x, _ = generate_pcfm(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                num_steps=args.num_steps,
                rng=rng,
                penalty_weight=args.pcfm_penalty_weight,
                num_correction_iters=args.num_correction_iters,
                correction_lr=args.correction_lr,
                num_final_projection_iters=args.num_final_projection_iters,
                eps_reg=args.pcfm_eps_reg,
            )
            return x[0]
        return jax.jit(_gen)

    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--example", choices=["star", "mnist"], required=True,
        help="Which pretrained model + constraint to benchmark."
    )
    parser.add_argument(
        "--method", choices=["ours", "pigdm", "pcfm"], required=True,
        help="ours = primal-dual generate_constrained; "
             "pigdm = PiGDM; pcfm = Physics-Constrained Flow Matching."
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; samples use jax.random.split off it.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Override the default data/<example>_model.pkl.")
    parser.add_argument("--out", type=str, default=None,
                        help="If set, save per-sample timings, violations, "
                             "and hyperparameters as JSON for later plotting.")

    # Diffrax step size (used by pd and pigdm).
    parser.add_argument("--dt", type=float, default=0.01)

    # Primal-dual (generate_constrained).
    parser.add_argument("--penalty-weight", type=float, default=5.0)
    parser.add_argument("--rescale-factor", type=float, default=1.0)
    parser.add_argument("--rescale-exponent", type=float, default=2.0)

    # PiGDM.
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--eps-reg", type=float, default=1e-4)

    # PCFM.
    parser.add_argument("--num-steps", type=int, default=100,
                        help="Number of outer integration steps for PCFM.")
    parser.add_argument("--pcfm-penalty-weight", type=float, default=0.1,
                        help="lambda in PCFM relaxed correction.")
    parser.add_argument("--num-correction-iters", type=int, default=10)
    parser.add_argument("--correction-lr", type=float, default=0.1)
    parser.add_argument("--num-final-projection-iters", type=int, default=20)
    parser.add_argument("--pcfm-eps-reg", type=float, default=1e-10,
                        help="Tikhonov regulariser for PCFM projection solve.")

    args = parser.parse_args()

    save_path = (Path(args.save_path) if args.save_path
                 else Path(f"data/{args.example}_model.pkl"))
    print(f"Loading model from {save_path}")
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    constraint_fn, violation_fn = build_constraint(args.example)
    gen = build_generator(args.method, model, normalizer, constraint_fn, args)

    base_rng = jax.random.key(args.seed)
    rngs = jax.random.split(base_rng, args.num_samples)

    # Warm-up: trigger JIT compilation with the first RNG.
    print(f"Compiling {args.method} for {args.example} ...")
    t0 = time.perf_counter()
    first = gen(rngs[0])
    jax.block_until_ready(first)
    compile_time = time.perf_counter() - t0
    print(f"  compile + first sample: {compile_time:.2f} s\n")

    # Per-sample timing loop. Sample 0 was already produced above, but we
    # re-run it here so all reported times come from the post-compile regime.
    per_sample_times = []
    per_sample_violations = []
    print(f"{'i':>4} | {'time (ms)':>10} | {'violation':>11}")
    print("-" * 32)
    for i, rng in enumerate(rngs):
        t0 = time.perf_counter()
        sample = gen(rng)
        jax.block_until_ready(sample)
        elapsed = time.perf_counter() - t0
        v = float(violation_fn(sample))
        per_sample_times.append(elapsed)
        per_sample_violations.append(v)
        print(f"{i:>4d} | {elapsed * 1000:>10.2f} | {v:>11.3e}")

    times = jnp.array(per_sample_times)
    violations = jnp.array(per_sample_violations)
    print("\nSummary")
    print(f"  method:    {args.method}")
    print(f"  example:   {args.example}")
    print(f"  N samples: {args.num_samples}")
    print(f"  time/sample (ms): mean={float(times.mean()) * 1000:7.2f}  "
          f"std={float(times.std()) * 1000:6.2f}  "
          f"min={float(times.min()) * 1000:6.2f}  "
          f"max={float(times.max()) * 1000:6.2f}")
    print(f"  violation:        mean={float(violations.mean()):.3e}  "
          f"max={float(violations.max()):.3e}")

    if args.out is not None:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "method": args.method,
            "example": args.example,
            "num_samples": args.num_samples,
            "compile_time_s": compile_time,
            "times_s": per_sample_times,
            "violations": per_sample_violations,
            "params": vars(args),
        }
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"\nSaved per-sample results to {out_path}")


if __name__ == "__main__":
    main()
