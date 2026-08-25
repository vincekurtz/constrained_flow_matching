"""Command line entry point.

Replaces the seven per-example argparse scripts that used to live in
``examples/``. Because it reads the problem and method registries, adding an
example or a baseline makes it appear here automatically.

    uv run -m cfm.cli train    --problem star
    uv run -m cfm.cli generate --problem star
    uv run -m cfm.cli generate --problem star --method ldf
    uv run -m cfm.cli generate --problem obstacles --method cbf \\
        --num-obstacles 3 --scene-seed 7
    uv run -m cfm.cli benchmark --problem star --method ldf --num-samples 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import problems
from cfm import methods, sweep, training
from cfm.core import checkpoint
from cfm.methods.ldf import generate_unconstrained

# Hyperparameters that any method may accept. Each is passed through to the
# method only when the user actually set it, so registry defaults stay in
# charge otherwise.
GENERATION_FLAGS = {
    "--dt": {"type": float, "help": "ODE step size."},
    "--num-steps": {"type": int, "help": "Fixed step count (PCFM)."},
    "--penalty-weight": {"type": float, "help": "Constraint penalty weight."},
    "--correction-weight": {
        "type": float,
        "help": "PCFM relaxed-correction weight (distinct from "
                "--penalty-weight, which is LDF's constraint scale).",
    },
    "--rescale-factor": {
        "type": float,
        "help": "Multiplier flow rescaling. Zero is the penalty-only "
                "ablation, which --method penalty fixes.",
    },
    "--rescale-exponent": {"type": float, "help": "Equality flow exponent p."},
    "--slack": {
        "choices": ["closed_form", "ode"],
        "help": "Slack handling for inequality constraints.",
    },
    "--num-projection-iters": {
        "type": int, "help": "Gauss-Newton polish iterations."
    },
    "--guidance-scale": {"type": float, "help": "PiGDM guidance weight."},
    "--eps-reg": {"type": float, "help": "Tikhonov regulariser."},
    "--phi0": {"type": float, "help": "CBF class-K gain."},
    "--omega": {"type": float, "help": "CBF blow-up gain."},
    "--qp": {"choices": ["exact", "elastic"], "help": "CBF QP formulation."},
    "--qp-penalty": {"type": float, "help": "CBF elastic violation price."},
}


def _flag_to_kwarg(flag):
    return flag.lstrip("-").replace("-", "_")


def _add_problem_options(parser, problem):
    """Add the extra options a specific problem declares."""
    for flag, kwargs in problem.options.items():
        parser.add_argument(flag, **kwargs)


def _problem_option_values(args, problem):
    """Collect this problem's own options, for make_constraint."""
    return {
        _flag_to_kwarg(flag): getattr(args, _flag_to_kwarg(flag))
        for flag in problem.options
    }


def _generation_overrides(args):
    """Generation hyperparameters the user actually set."""
    return {
        _flag_to_kwarg(flag): getattr(args, _flag_to_kwarg(flag), None)
        for flag in GENERATION_FLAGS
    }


def _resolve(problem, method, args):
    """Merge problem gains with explicit CLI overrides.

    Precedence is registry defaults, then the problem's per-method gains,
    then whatever the user typed.
    """
    cfg = problem.gains_for(method.name)
    cfg.update({
        k: v for k, v in _generation_overrides(args).items() if v is not None
    })
    # PCFM steps a fixed grid rather than integrating, so dt means nothing.
    if method.name not in methods.USES_DT:
        cfg.pop("dt", None)
    else:
        cfg.pop("num_steps", None)
        cfg.setdefault("dt", 0.01)
    return cfg


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_train(args):
    problem = problems.get(args.problem)
    path = Path(args.save_path or problem.checkpoint_path)
    cfg = problem.train

    print(f"Training {problem.label} ...")
    model, normalizer = training.train(
        dataset=problem.make_dataset(),
        model=problem.make_model(),
        num_epochs=args.num_epochs or cfg.num_epochs,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        seed=cfg.seed,
        print_frequency=cfg.print_frequency,
    )
    checkpoint.save(path, model, normalizer)
    print(f"Saved trained model and normalizer to {path}")


def cmd_generate(args):
    problem = problems.get(args.problem)
    path = Path(args.save_path or problem.checkpoint_path)
    model, normalizer = checkpoint.load(path)
    num_samples = args.num_samples or problem.default_num_samples
    rng = jax.random.key(args.seed)

    if args.method is None:
        print(f"Generating {num_samples} unconstrained samples ...")
        samples = generate_unconstrained(
            model, normalizer, num_samples=num_samples,
            dt=args.dt if args.dt is not None else 0.01, rng=rng,
        )
        constraint = None
    else:
        method = methods.get(args.method)
        constraint = problem.make_constraint(
            **_problem_option_values(args, problem)
        )
        cfg = _resolve(problem, method, args)
        print(f"Generating {num_samples} samples with {method.label}")
        print(f"  constraint: {constraint.name} ({constraint.kind})")
        print(f"  settings:   {cfg}")

        start = time.time()
        samples = method.run(
            model, normalizer, constraint,
            num_samples=num_samples, rng=rng, **cfg,
        )
        jax.block_until_ready(samples.x)
        print(f"  took {time.time() - start:.2f} s")

        v = constraint.violations(samples.x)
        print(f"  violation:  mean={float(jnp.nanmean(v)):.3e} "
              f"max={float(jnp.nanmax(v)):.3e}")

    if not args.no_plot:
        problem.plot(problem, samples, constraint)


def cmd_benchmark(args):
    """Time a method one sample at a time, and record its violation."""
    problem = problems.get(args.problem)
    method = methods.get(args.method)
    model, normalizer = checkpoint.load(
        Path(args.save_path or problem.checkpoint_path)
    )
    constraint = problem.make_constraint(
        **_problem_option_values(args, problem)
    )
    if not method.supports_constraint(constraint):
        raise SystemExit(
            f"method {method.name!r} does not support {constraint.kind} "
            f"constraints; supported here: "
            f"{', '.join(m.name for m in methods.supporting(constraint))}"
        )

    cfg = _resolve(problem, method, args)

    def _gen(rng):
        return method.run(
            model, normalizer, constraint, num_samples=1, rng=rng, **cfg
        ).x[0]

    gen = jax.jit(_gen)
    rngs = jax.random.split(jax.random.key(args.seed), args.num_samples)

    print(f"Compiling {method.name} for {problem.name} ...")
    t0 = time.perf_counter()
    jax.block_until_ready(gen(rngs[0]))
    compile_time = time.perf_counter() - t0
    print(f"  compile + first sample: {compile_time:.2f} s\n")

    times, violations = [], []
    print(f"{'i':>4} | {'time (ms)':>10} | {'violation':>11}")
    print("-" * 32)
    for i, rng in enumerate(rngs):
        t0 = time.perf_counter()
        sample = gen(rng)
        jax.block_until_ready(sample)
        elapsed = time.perf_counter() - t0
        v = float(constraint.violation(sample))
        times.append(elapsed)
        violations.append(v)
        print(f"{i:>4d} | {elapsed * 1000:>10.2f} | {v:>11.3e}")

    t = jnp.array(times)
    v = jnp.array(violations)
    print("\nSummary")
    print(f"  method:    {method.name} ({method.label})")
    print(f"  problem:   {problem.name}")
    print(f"  N samples: {args.num_samples}")
    print(f"  time/sample (ms): mean={float(t.mean()) * 1000:7.2f}  "
          f"std={float(t.std()) * 1000:6.2f}  "
          f"min={float(t.min()) * 1000:6.2f}  "
          f"max={float(t.max()) * 1000:6.2f}")
    print(f"  violation:        mean={float(jnp.nanmean(v)):.3e}  "
          f"max={float(jnp.nanmax(v)):.3e}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "problem": problem.name,
            "method": method.name,
            "num_samples": args.num_samples,
            "compile_time_s": compile_time,
            "times_s": times,
            "violations": violations,
            "config": {k: str(v) for k, v in cfg.items()},
        }, indent=2) + "\n")
        print(f"\nSaved per-sample results to {out}")


def cmd_sweep(args):
    """Run a declarative benchmark sweep."""
    config = sweep.load_config(args.config)
    records = sweep.run_sweep(
        config, results_dir=Path(args.results_dir), force=args.force
    )
    print(f"\n{len(records)} results in {args.results_dir}\n")
    print(sweep.render_table(records))

    if args.check_baseline:
        regressions = sweep.compare_to_baseline(records)
        if regressions:
            print("\nRegressions against the recorded baseline:")
            for line in regressions:
                print(f"  {line}")
            raise SystemExit(1)
        print("\nNo regressions against the recorded baseline.")


def cmd_table(args):
    """Render recorded sweep results as a table."""
    records = sweep.load_results(Path(args.results_dir))
    if not records:
        raise SystemExit(
            f"no results in {args.results_dir}; run `cfm sweep` first"
        )
    print(sweep.render_table(records, fmt=args.format))


def cmd_list(args):
    """Show what is registered, and which methods fit which problem.

    The compatibility table used to be a hand-maintained dict in
    ``benchmark.py``; here it is derived from each constraint's kind.
    """
    del args
    print("Methods")
    for m in methods.METHODS.values():
        kinds = ", ".join(sorted(m.supports))
        print(f"  {m.name:10s} {m.label:34s} [{kinds}]")

    print("\nProblems")
    for name in problems.names():
        p = problems.get(name)
        if not p.constrained:
            print(f"  {name:12s} {p.label:22s} unconstrained only")
            continue
        constraint = p.make_constraint()
        usable = ", ".join(m.name for m in methods.supporting(constraint))
        print(f"  {name:12s} {p.label:22s} {constraint.kind:10s} {usable}")


# --------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        prog="cfm",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_problem_arg(p, required=True):
        p.add_argument(
            "--problem", choices=problems.names(), required=required,
            help="Which example to run.",
        )
        p.add_argument("--save-path", type=str, default=None,
                       help="Override the default checkpoint path.")

    p_train = sub.add_parser("train", help="Train a flow model.")
    add_problem_arg(p_train)
    p_train.add_argument("--num-epochs", type=int, default=None,
                         help="Override the problem's epoch count.")
    p_train.set_defaults(func=cmd_train)

    p_gen = sub.add_parser("generate", help="Generate samples and plot them.")
    add_problem_arg(p_gen)
    p_gen.add_argument(
        "--method", choices=sorted(methods.METHODS), default=None,
        help="Constrained sampler. Omit for unconstrained generation.",
    )
    p_gen.add_argument("--num-samples", type=int, default=None)
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--no-plot", action="store_true",
                       help="Skip plotting; just report the numbers.")
    p_gen.set_defaults(func=cmd_generate)

    p_bench = sub.add_parser(
        "benchmark", help="Time one method, one sample at a time."
    )
    add_problem_arg(p_bench)
    p_bench.add_argument("--method", choices=sorted(methods.METHODS),
                         required=True)
    p_bench.add_argument("--num-samples", type=int, default=20)
    p_bench.add_argument("--seed", type=int, default=0)
    p_bench.add_argument("--out", type=str, default=None,
                         help="Write per-sample results here as JSON.")
    p_bench.set_defaults(func=cmd_benchmark)

    p_sweep = sub.add_parser(
        "sweep", help="Run a declarative benchmark sweep from a TOML config."
    )
    p_sweep.add_argument("config", type=str,
                         help="Path to a sweep TOML, e.g. "
                              "experiments/table1.toml")
    p_sweep.add_argument("--results-dir", type=str, default="results")
    p_sweep.add_argument("--force", action="store_true",
                         help="Re-run cases that already have results.")
    p_sweep.add_argument("--check-baseline", action="store_true",
                         help="Fail if any case regressed against "
                              "experiments/baseline.json.")
    p_sweep.set_defaults(func=cmd_sweep)

    p_table = sub.add_parser(
        "table", help="Render recorded sweep results as a table."
    )
    p_table.add_argument("--results-dir", type=str, default="results")
    p_table.add_argument("--format", choices=["markdown", "latex"],
                         default="markdown")
    p_table.set_defaults(func=cmd_table)

    p_list = sub.add_parser("list", help="List registered methods/problems.")
    p_list.set_defaults(func=cmd_list)

    # Generation hyperparameters and per-problem options apply to both
    # generate and benchmark. Defaults are None so "unset" is detectable.
    for p in (p_gen, p_bench):
        for flag, kwargs in GENERATION_FLAGS.items():
            p.add_argument(flag, default=None, **kwargs)

    return parser, (p_gen, p_bench)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser, needs_problem_opts = build_parser()

    # A problem may declare its own flags (--scene-seed, --constraint, ...).
    # They are only meaningful once we know which problem was chosen, so do a
    # permissive first pass, then add them and parse for real.
    known, _ = parser.parse_known_args(argv)
    name = getattr(known, "problem", None)
    if name in problems.all_problems():
        for p in needs_problem_opts:
            _add_problem_options(p, problems.get(name))

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
