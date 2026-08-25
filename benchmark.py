"""Benchmark constrained generation algorithms one sample at a time.

Loads a pretrained model (star, MNIST, or obstacle avoidance), JIT-compiles a
single-sample generator for the chosen method, and reports per-sample
wall-clock time and constraint violation.

Not every method handles every example. The star and MNIST constraints are
equalities, which the CBF safety filter does not support; the obstacle
constraint is an inequality, which PCFM and PiGDM do not support:

    example     | ours | pigdm | pcfm | cbf
    ------------+------+-------+------+-----
    star        |  x   |   x   |  x   |
    mnist       |  x   |   x   |  x   |
    obstacle    |  x   |       |      |  x

Usage examples:

    python benchmark.py --example star --method pcfm --num-samples 20
    python benchmark.py --example star --method pigdm --num-samples 20 \\
        --guidance-scale 1.0 --eps-reg 1e-4
    python benchmark.py --example mnist --method ours --num-samples 5 \\
        --penalty-weight 10.0 --rescale-factor 1.0
    python benchmark.py --example obstacle --method cbf --num-samples 20 \\
        --num-obstacles 2 --scene-seed 0
"""

import argparse
import json
import time
from pathlib import Path

import cloudpickle
import diffrax
import jax
import jax.numpy as jnp

from cbf import generate_cbf
from datasets.mnist import MNISTDataset
from examples.obstacle_scene import (
    SLACK_GAINS,
    make_constraint_fn,
    sample_scene,
)
from generation import generate_constrained, generate_inequality_constrained
from pcfm import generate_pcfm
from pi_gdm import generate_pigdm

# Which methods can handle each example's constraint. The CBF safety filter
# only supports inequalities, and PCFM / PiGDM only support equalities.
SUPPORTED_METHODS = {
    "star": ("ours", "pigdm", "pcfm"),
    "mnist": ("ours", "pigdm", "pcfm"),
    "obstacle": ("ours", "cbf"),
}

# Defaults for our method's gains, per example. The obstacle scene puts many
# constraints in play at once and needs much stiffer settings than the star and
# MNIST constraints do; --penalty-weight / --rescale-factor override either.
DEFAULT_GAINS = {"penalty_weight": 5.0, "rescale_factor": 1.0}


def build_constraint(example: str, args):
    """Return (constraint_fn, scalar_violation_fn) for the given example.

    The constraint_fn operates on a single *unnormalized* sample and matches
    the convention used by the generation algorithms. The violation_fn maps
    a single sample to a scalar magnitude for reporting: the residual
    magnitude for an equality constraint, and the amount by which the
    inequality is exceeded for an inequality one.
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

    if example == "obstacle":
        # Obstacle avoidance h(x) <= 0 on a freshly sampled scene, identical
        # to the constraint used in examples/obstacles.py.
        centers, radii = sample_scene(args.scene_seed, args.num_obstacles)
        obstacle_constraint = make_constraint_fn(centers, radii)

        def violation_fn(x):
            return jnp.max(jnp.maximum(obstacle_constraint(x), 0.0))

        return obstacle_constraint, violation_fn

    raise ValueError(f"Unknown example: {example}")


def resolve_gains(args):
    """Return (penalty_weight, rescale_factor) for our method."""
    defaults = (
        SLACK_GAINS[args.slack] if args.example == "obstacle"
        else DEFAULT_GAINS
    )
    penalty_weight = (
        args.penalty_weight if args.penalty_weight is not None
        else defaults["penalty_weight"]
    )
    rescale_factor = (
        args.rescale_factor if args.rescale_factor is not None
        else defaults["rescale_factor"]
    )
    return penalty_weight, rescale_factor


def build_generator(method: str, model, normalizer, constraint_fn, args):
    """JIT-compile a function ``rng -> single_sample`` for the chosen method."""
    if method == "ours":
        penalty_weight, rescale_factor = resolve_gains(args)

        if args.example == "obstacle":
            # The obstacle constraint is an inequality, so it goes through the
            # slack formulation rather than the equality flow.
            def _gen(rng):
                x, _ = generate_inequality_constrained(
                    model,
                    normalizer,
                    constraint_fn,
                    num_samples=1,
                    dt=args.dt,
                    rng=rng,
                    penalty_weight=penalty_weight,
                    rescale_factor=rescale_factor,
                    slack=args.slack,
                )
                return x[0]
            return jax.jit(_gen)

        def _gen(rng):
            x, _, _ = generate_constrained(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                dt=args.dt,
                rng=rng,
                penalty_weight=penalty_weight,
                rescale_factor=rescale_factor,
                rescale_exponent=args.rescale_exponent,
            )
            return x[0]
        return jax.jit(_gen)

    if method == "cbf":
        # SafeFlow's Algorithm 1 integrates with an embedded RK pair and error
        # control, which --adaptive reproduces. The omega / (1 - t)^2 schedule
        # is genuinely stiff at the end of the horizon, so the fixed midpoint
        # steps used elsewhere are not always enough to hold on to it.
        integrator = (
            {
                "solver": diffrax.Tsit5(),
                "stepsize_controller": diffrax.PIDController(
                    rtol=1e-5, atol=1e-7
                ),
            }
            if args.adaptive
            else {}
        )

        def _gen(rng):
            x, _ = generate_cbf(
                model,
                normalizer,
                constraint_fn,
                num_samples=1,
                dt=args.dt,
                rng=rng,
                phi0=args.phi0,
                omega=args.omega,
                qp=args.qp,
                qp_penalty=args.qp_penalty,
                num_terminal_iters=args.num_terminal_iters,
                **integrator,
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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--example", choices=["star", "mnist", "obstacle"], required=True,
        help="Which pretrained model + constraint to benchmark. star and "
             "mnist impose equality constraints, obstacle an inequality."
    )
    parser.add_argument(
        "--method", choices=["ours", "pigdm", "pcfm", "cbf"], required=True,
        help="ours = primal-dual flow; pigdm = PiGDM; pcfm = "
             "Physics-Constrained Flow Matching; cbf = SafeFlow control "
             "barrier function filter (inequality constraints only, so "
             "--example obstacle)."
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; samples use jax.random.split off it.")
    parser.add_argument("--save-path", type=str, default=None,
                        help="Override the default data/<example>_model.pkl.")
    parser.add_argument("--out", type=str, default=None,
                        help="If set, save per-sample timings, violations, "
                             "and hyperparameters as JSON for later plotting.")

    # Diffrax step size (used by ours, pigdm, and cbf).
    parser.add_argument("--dt", type=float, default=0.01)

    # Primal-dual (ours). The penalty and rescale defaults depend on the
    # example; see DEFAULT_GAINS and SLACK_GAINS.
    parser.add_argument("--penalty-weight", type=float, default=None)
    parser.add_argument("--rescale-factor", type=float, default=None)
    parser.add_argument("--rescale-exponent", type=float, default=2.0,
                        help="Equality-constrained flow only.")
    parser.add_argument(
        "--slack", choices=["closed_form", "ode"], default="closed_form",
        help="How the inequality solver handles the slack variable "
             "(--example obstacle only)."
    )

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

    # CBF safety filter (SafeFlow).
    parser.add_argument("--phi0", type=float, default=1.0,
                        help="Class-K gain used where the sample is feasible.")
    parser.add_argument("--omega", type=float, default=4.0,
                        help="Blow-up gain omega / (1 - t)^2 used where the "
                             "sample is infeasible.")
    parser.add_argument("--qp", choices=["exact", "elastic"],
                        default="elastic",
                        help="Solve the CBF quadratic program in its elastic "
                             "relaxation (default), which tolerates mutually "
                             "infeasible barrier conditions, or exactly as "
                             "written, which raises when they cannot all be "
                             "met at once -- as happens on many samples of "
                             "the obstacle scene.")
    parser.add_argument("--qp-penalty", type=float, default=1e4,
                        help="Price per unit of barrier-condition violation "
                             "(--qp elastic only).")
    parser.add_argument("--num-terminal-iters", type=int, default=20,
                        help="Gauss-Newton iterations for the terminal safety "
                             "filter at t = 1. 0 disables it.")
    parser.add_argument("--adaptive", action="store_true",
                        help="Integrate the CBF flow with Tsit5 and PID error "
                             "control instead of fixed midpoint steps, as in "
                             "SafeFlow's Algorithm 1.")

    # Obstacle-avoidance scene.
    parser.add_argument("--num-obstacles", type=int, default=2,
                        help="Number of obstacles in the test scene "
                             "(--example obstacle only).")
    parser.add_argument("--scene-seed", type=int, default=0,
                        help="Random seed for the test scene "
                             "(--example obstacle only).")

    args = parser.parse_args()

    if args.method not in SUPPORTED_METHODS[args.example]:
        parser.error(
            f"method {args.method!r} does not support example "
            f"{args.example!r}; supported methods are "
            f"{', '.join(SUPPORTED_METHODS[args.example])}."
        )

    # Fill in the example-dependent gain defaults now, so that --out records
    # the values actually used rather than a null.
    args.penalty_weight, args.rescale_factor = resolve_gains(args)

    save_path = (Path(args.save_path) if args.save_path
                 else Path(f"data/{args.example}_model.pkl"))
    print(f"Loading model from {save_path}")
    with open(save_path, "rb") as f:
        data = cloudpickle.load(f)
    model = data["model"]
    normalizer = data["normalizer"]

    constraint_fn, violation_fn = build_constraint(args.example, args)
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
