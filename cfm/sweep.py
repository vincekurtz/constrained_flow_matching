"""Declarative benchmark sweeps.

Replaces ``make_benchmark_table.sh``, which re-ran the benchmark and then
recovered the numbers by grepping its stdout with a regex -- while the
benchmark was already able to write them as JSON, and while the figures kept
a second, parallel timing stack of their own.

Here one runner produces one result file per case, and both the table and the
figures read those files. A case is skipped when a result for its exact
configuration already exists, so an interrupted sweep resumes rather than
starting over.

A configuration is a list of ``[[block]]`` sections, each its own
problems x methods x steps grid -- one table needs more than one grid, since
the obstacle rows run a different scene and a much finer step size than the
2-D and MNIST rows. A row within a grid is named by a ``[[variants]]`` entry:
a registered method with some gains pinned, which is how the same method can
appear twice (LDF with and without the final projection) without being
registered twice.
"""

import hashlib
import json
import platform
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp

import problems
from cfm import methods
from cfm.core import checkpoint

RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class Case:
    """One (problem, row, steps) benchmark run.

    A *row* of the table is a method plus the gains that pin it down, which
    is not the same thing as a method: LDF with the final projection and LDF
    without it are one method at two settings, and both are rows. ``variant``
    names the row, and is what keeps the two apart in the results directory.
    """

    problem: str
    method: str
    steps: int
    num_samples: int
    variant: str = ""
    row_label: str = ""
    problem_label: str = ""
    gains: Dict[str, Any] = field(default_factory=dict)
    problem_options: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The row's name: its variant, or the method when it has none."""
        return self.variant or self.method

    @property
    def key(self) -> str:
        """Stable identity: same configuration, same file."""
        payload = json.dumps({
            "problem": self.problem,
            "method": self.method,
            "variant": self.name,
            "steps": self.steps,
            "num_samples": self.num_samples,
            "gains": self.gains,
            "problem_options": self.problem_options,
        }, sort_keys=True)
        digest = hashlib.sha256(payload.encode()).hexdigest()[:10]
        return f"{self.problem}_{self.name}_{self.steps}_{digest}"

    @property
    def label(self) -> str:
        # The scene is in the progress line because two obstacle blocks run
        # the same problem and would otherwise scroll past identically.
        scene = "".join(
            f" {k}={v}" for k, v in sorted(self.problem_options.items())
        )
        return f"{self.problem}{scene}/{self.name}/{self.steps} steps"


@dataclass(frozen=True)
class Variant:
    """A named table row: a registered method with some gains pinned.

    Declaring the two LDF rows as variants gives each its own label and its
    own result file without inventing a second entry in the method registry
    for what is one algorithm with one gain changed.
    """

    name: str
    method: str
    label: str
    gains: Dict[str, Any] = field(default_factory=dict)


def load_config(path) -> Dict[str, Any]:
    """Read a sweep configuration from TOML."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _variants(config: Dict[str, Any]) -> Dict[str, Variant]:
    """The variants a config declares, keyed by the name its rows use."""
    out = {}
    for entry in config.get("variants", []):
        method_name = entry.get("method", entry["name"])
        method = methods.get(method_name)  # fail loudly on a typo
        out[entry["name"]] = Variant(
            name=entry["name"],
            method=method_name,
            label=entry.get("label", method.label),
            gains=dict(entry.get("gains", {})),
        )
    return out


def _blocks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The config's blocks, each its own problems x methods x steps grid.

    A config with no ``[[block]]`` section is itself the single block, so the
    flat form still works.
    """
    shared = {
        k: v for k, v in config.items() if k not in ("block", "variants")
    }
    blocks = config.get("block")
    if not blocks:
        return [shared]
    return [
        # Gain overrides accumulate rather than replace: a block adds its own
        # to whatever the config declared for every block, and wins on a tie
        # by coming later in the list.
        {**shared, **b, "gains": shared.get("gains", []) + b.get("gains", [])}
        for b in blocks
    ]


def _selects(value, *names) -> bool:
    """Does a gain entry's selector cover this row?

    An absent key selects everything, and a list selects any of its entries,
    so one override can name several rows at once.
    """
    if value is None:
        return True
    if isinstance(value, list):
        return any(v in names for v in value)
    return value in names


def build_cases(config: Dict[str, Any]) -> List[Case]:
    """Expand a sweep configuration into individual cases.

    Cases whose method cannot handle the problem's constraint are dropped
    rather than erroring, so one config can span problems with different
    constraint kinds.
    """
    variants = _variants(config)
    cases = []

    for block in _blocks(config):
        num_samples = block.get("num_samples", 20)
        overrides = block.get("gains", [])
        problem_label = block.get("problem_label")
        if problem_label and len(block["problems"]) > 1:
            raise ValueError(
                "problem_label renames one problem's rows, so a block that "
                f"sets it may list only one problem; got {block['problems']}"
            )

        for problem_name in block["problems"]:
            problem = problems.get(problem_name)
            problem_options = block.get("problem_options", {}).get(
                problem_name, {}
            )
            constraint = problem.make_constraint(**problem_options)

            for row_name in block["methods"]:
                variant = variants.get(row_name)
                method_name = variant.method if variant else row_name
                method = methods.get(method_name)
                if not method.supports_constraint(constraint):
                    continue

                for steps in block["steps"]:
                    gains = problem.gains_for(method_name)
                    for entry in overrides:
                        if (_selects(entry.get("method"),
                                     row_name, method_name)
                                and _selects(entry.get("problem"),
                                             problem_name)
                                and _selects(entry.get("steps"), steps)):
                            gains.update({
                                k: v for k, v in entry.items()
                                if k not in ("method", "problem", "steps")
                            })
                    # A variant's own gains are what make it that row, so
                    # they land last: an override aimed at the method as a
                    # whole tunes both LDF rows without collapsing them into
                    # one.
                    if variant:
                        gains.update(variant.gains)

                    cases.append(Case(
                        problem=problem_name,
                        method=method_name,
                        variant=row_name,
                        row_label=variant.label if variant else method.label,
                        problem_label=problem_label or problem.label,
                        steps=steps,
                        num_samples=num_samples,
                        gains=gains,
                        problem_options=problem_options,
                    ))
    return cases


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def run_case(case: Case, results_dir: Path = RESULTS_DIR) -> Dict[str, Any]:
    """Time one method on one problem, one sample at a time."""
    problem = problems.get(case.problem)
    method = methods.get(case.method)
    model, normalizer = checkpoint.load(problem.checkpoint_path)
    constraint = problem.make_constraint(**case.problem_options)

    cfg = dict(case.gains)
    # A step count means dt for the integrating methods and num_steps for
    # PCFM, which walks a fixed grid instead.
    if case.method in methods.USES_DT:
        cfg["dt"] = 1.0 / case.steps
    else:
        cfg["num_steps"] = case.steps

    def _gen(rng):
        return method.run(
            model, normalizer, constraint, num_samples=1, rng=rng, **cfg
        ).x[0]

    gen = jax.jit(_gen)
    rngs = jax.random.split(jax.random.key(0), case.num_samples)

    t0 = time.perf_counter()
    jax.block_until_ready(gen(rngs[0]))
    compile_time = time.perf_counter() - t0

    times, violations = [], []
    for rng in rngs:
        t0 = time.perf_counter()
        sample = gen(rng)
        jax.block_until_ready(sample)
        times.append(time.perf_counter() - t0)
        violations.append(float(constraint.violation(sample)))

    v = jnp.array(violations)
    record = {
        "problem": case.problem,
        "problem_label": case.problem_label or problem.label,
        "problem_options": case.problem_options,
        "method": case.method,
        "variant": case.name,
        "method_label": case.row_label or method.label,
        "steps": case.steps,
        "num_samples": case.num_samples,
        "config": {k: str(val) for k, val in cfg.items()},
        "compile_time_s": compile_time,
        "times_s": times,
        "violations": violations,
        "mean_time_ms": 1000 * sum(times) / len(times),
        "mean_violation": float(jnp.nanmean(v)),
        "max_violation": float(jnp.nanmax(v)),
        "num_nan": int(jnp.sum(jnp.isnan(v))),
        "git_sha": _git_sha(),
        "platform": platform.platform(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{case.key}.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    return record


def run_sweep(
    config: Dict[str, Any],
    results_dir: Path = RESULTS_DIR,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Run every case in a configuration, skipping ones already recorded."""
    cases = build_cases(config)
    print(f"{len(cases)} cases")
    records = []

    for i, case in enumerate(cases, 1):
        path = results_dir / f"{case.key}.json"
        if path.exists() and not force:
            print(f"[{i}/{len(cases)}] {case.label}: cached")
            records.append(json.loads(path.read_text()))
            continue

        print(f"[{i}/{len(cases)}] {case.label}: running ...", flush=True)
        record = run_case(case, results_dir)
        print(f"    {record['mean_time_ms']:8.2f} ms   "
              f"violation mean={record['mean_violation']:.3e}")
        records.append(record)

    return records


def load_results(results_dir: Path = RESULTS_DIR) -> List[Dict[str, Any]]:
    """Every recorded result, newest configuration wins on ties."""
    return [
        json.loads(p.read_text())
        for p in sorted(Path(results_dir).glob("*.json"))
    ]


def render_table(records, fmt: str = "markdown") -> str:
    """Render benchmark records as a table.

    Rows are ordered by problem, then by the problem's label so the two
    obstacle scenes stay in separate stretches, then step count, then the
    order methods are registered, then the row label so a method's variants
    keep a fixed order. The table therefore reads the same way every time it
    is regenerated, from whatever result files happen to be on disk.
    """
    order = list(methods.METHODS)
    rows = sorted(
        records,
        key=lambda r: (
            r["problem"], r.get("problem_label") or "", r["steps"],
            order.index(r["method"]) if r["method"] in order else 99,
            r.get("method_label") or "",
        ),
    )

    header = ["Problem", "Steps", "Method", "Time (ms)", "Violation"]
    body = []
    for r in rows:
        note = f" ({r['num_nan']} NaN)" if r.get("num_nan") else ""
        body.append([
            r.get("problem_label") or problems.get(r["problem"]).label,
            str(r["steps"]),
            r["method_label"],
            f"{r['mean_time_ms']:.2f}",
            f"{r['mean_violation']:.2e}{note}",
        ])

    if fmt == "latex":
        lines = [
            r"\begin{tabular}{lllrr}", r"\toprule",
            " & ".join(header) + r" \\", r"\midrule",
        ]
        lines += [" & ".join(row) + r" \\" for row in body]
        lines += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(lines)

    widths = [
        max(len(header[i]), *(len(row[i]) for row in body))
        for i in range(len(header))
    ]

    def fmt_row(cells):
        return "| " + " | ".join(
            c.ljust(widths[i]) for i, c in enumerate(cells)
        ) + " |"

    rule = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [fmt_row(header), rule]
    lines += [fmt_row(row) for row in body]
    return "\n".join(lines)


def _ran_the_baseline_scenario(record, want) -> bool:
    """Did this case run the problem configuration the baseline recorded?

    The baseline keys on problem and method alone, from before a problem
    could appear under more than one constraint or scene. A case whose
    problem options the baseline never ran -- the star under its inequality,
    the crowded obstacle scene -- is not a comparison, so it is skipped
    rather than measured against the wrong reference.
    """
    params = want.get("params", {})
    return all(
        params.get(k) == v
        for k, v in (record.get("problem_options") or {}).items()
    )


def compare_to_baseline(
    records,
    baseline_path: Optional[Path] = None,
    time_tolerance: float = 1.05,
    baseline_steps: int = 100,
):
    """Check recorded results against the pre-refactor baseline.

    The baseline was recorded at a single resolution (dt = 0.01, and
    num_steps = 100 for PCFM), so only cases at that step count are
    comparable; others are skipped rather than compared against the wrong
    reference.

    Returns a list of human-readable regressions. Violation must be no worse,
    and wall-clock no more than ``time_tolerance`` times slower -- except
    that a case which spends more time and buys a better violation is
    reported as a trade, not a regression.
    """
    baseline_path = Path(baseline_path or "experiments/baseline.json")
    if not baseline_path.exists():
        return [f"no baseline at {baseline_path}"]

    baseline = json.loads(baseline_path.read_text())["results"]
    regressions = []

    for r in records:
        if r["steps"] != baseline_steps:
            continue
        # The baseline predates the final Gauss-Newton projection, so only
        # the rows that skip it are comparable. A projected row lands orders
        # of magnitude tighter and would otherwise read as an improvement in
        # something the baseline never measured.
        if float(r.get("config", {}).get("num_projection_iters", 0)) > 0:
            continue
        # The baseline predates the ours -> ldf rename.
        legacy = {"ldf": "ours"}.get(r["method"], r["method"])
        key = f"{r['problem']}/{legacy}"
        if key not in baseline:
            continue
        want = baseline[key]
        if not _ran_the_baseline_scenario(r, want):
            continue

        worse_violation = (
            not jnp.isnan(want["mean_violation"])
            and r["mean_violation"] > want["mean_violation"]
            # Both sides sit at the float32 noise floor for methods that
            # project to numerical zero; only flag a real change.
            and r["mean_violation"] > 1e-6
        )
        slower = r["mean_time_ms"] > time_tolerance * want["mean_time_ms"]
        better_violation = r["mean_violation"] < want["mean_violation"]

        if worse_violation:
            regressions.append(
                f"{key}: violation {r['mean_violation']:.3e} > baseline "
                f"{want['mean_violation']:.3e}"
            )
        if slower and not better_violation:
            regressions.append(
                f"{key}: {r['mean_time_ms']:.2f} ms > "
                f"{time_tolerance:g}x baseline {want['mean_time_ms']:.2f} ms"
            )
        elif slower and better_violation:
            ratio = r["mean_time_ms"] / want["mean_time_ms"]
            tighter = want["mean_violation"] / max(r["mean_violation"], 1e-30)
            print(
                f"  note: {key} is {ratio:.2f}x slower but {tighter:.0f}x "
                f"tighter ({want['mean_violation']:.2e} -> "
                f"{r['mean_violation']:.2e}); counted as a trade, not a "
                f"regression"
            )
    return regressions
