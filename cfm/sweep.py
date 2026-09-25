"""Declarative benchmark sweeps.

A config is a list of ``[[block]]`` sections, each a problems x methods x steps
grid. Each case writes one result file, and cases already on disk are skipped.
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
from cfm.core import checkpoint, constraints

RESULTS_DIR = Path("results")


@dataclass(frozen=True)
class Case:
    """One (problem, row, steps) benchmark run. ``variant`` names the row."""

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
        """The variant name, or the method when there is none."""
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
        # Include problem options so blocks on the same problem are distinct.
        scene = "".join(
            f" {k}={v}" for k, v in sorted(self.problem_options.items())
        )
        return f"{self.problem}{scene}/{self.name}/{self.steps} steps"


@dataclass(frozen=True)
class Variant:
    """A named table row: a registered method with some gains pinned."""

    name: str
    method: str
    label: str
    gains: Dict[str, Any] = field(default_factory=dict)


def load_config(path) -> Dict[str, Any]:
    """Read a sweep configuration from TOML."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def _variants(config: Dict[str, Any]) -> Dict[str, Variant]:
    """The config's variants, keyed by name."""
    out = {}
    for entry in config.get("variants", []):
        method_name = entry.get("method", entry["name"])
        method = methods.get(method_name)
        out[entry["name"]] = Variant(
            name=entry["name"],
            method=method_name,
            label=entry.get("label", method.label),
            gains=dict(entry.get("gains", {})),
        )
    return out


def _blocks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The config's blocks. A config without ``[[block]]`` is one block."""
    shared = {
        k: v for k, v in config.items() if k not in ("block", "variants")
    }
    blocks = config.get("block")
    if not blocks:
        return [shared]
    return [
        # Block gain overrides append to (and take precedence over) shared ones.
        {**shared, **b, "gains": shared.get("gains", []) + b.get("gains", [])}
        for b in blocks
    ]


def _selects(value, *names) -> bool:
    """Whether a selector (None, a value, or a list) matches any of names."""
    if value is None:
        return True
    if isinstance(value, list):
        return any(v in names for v in value)
    return value in names


def build_cases(config: Dict[str, Any]) -> List[Case]:
    """Expand a config into cases, dropping unsupported method/problem pairs."""
    variants = _variants(config)
    cases = []

    for block in _blocks(config):
        num_samples = block.get("num_samples", 20)
        overrides = block.get("gains", [])
        # A string labels the block's single problem; a table labels each.
        problem_label = block.get("problem_label", {})
        if isinstance(problem_label, str):
            if len(block["problems"]) > 1:
                raise ValueError(
                    "a string problem_label renames one problem's rows, so a "
                    "block that sets it may list only one problem; got "
                    f"{block['problems']} (use a table to rename each)"
                )
            problem_label = {block["problems"][0]: problem_label}

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
                    # Variant gains define the row, so they win.
                    if variant:
                        gains.update(variant.gains)

                    cases.append(Case(
                        problem=problem_name,
                        method=method_name,
                        variant=row_name,
                        row_label=variant.label if variant else method.label,
                        problem_label=problem_label.get(
                            problem_name, problem.label),
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
    """Load every recorded result."""
    return [
        json.loads(p.read_text())
        for p in sorted(Path(results_dir).glob("*.json"))
    ]


def render_table(records, fmt: str = "markdown") -> str:
    """Render benchmark records as a markdown or LaTeX table."""
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
        return render_latex_table(records)

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


# (row name, header) for the LaTeX columns. Unlisted rows are appended.
LATEX_COLUMNS = [
    ("penalty", "Penalty only"),
    ("cbf", r"CBF~\cite{safeflow2025}"),
    ("pcfm", r"PCFM~\cite{utkarsh2025pcfm}"),
    ("pigdm", r"$\Pi$GDM~\cite{pokle2024training}"),
    ("ldf", "LDF (no proj.)"),
    ("ldf_projected", "LDF + proj."),
]

# Bold times are within this factor of the fastest in their row; bold
# violations are below the threshold, near single-precision round-off.
LATEX_TIME_FACTOR = 1.2
LATEX_VIOLATION_THRESHOLD = 1e-6


def _latex_violation(value: float, num_nan: int) -> str:
    r"""Format a violation with the paper's ``\sci``/``\scib`` macros."""
    if value == 0:
        cell = r"\textbf{0}"
    else:
        mantissa, exponent = f"{value:.1e}".split("e")
        macro = r"\scib" if value < LATEX_VIOLATION_THRESHOLD else r"\sci"
        cell = f"{macro}{{{mantissa}}}{{{int(exponent)}}}"
    # Dagger marks a mean taken over non-NaN samples only.
    return cell + r"~$\dagger$" if num_nan else cell


def render_latex_table(records) -> str:
    """Render benchmark records as the paper's LaTeX tabular.

    One row per scenario (equality first), a Time/Viol. pair per method.
    Each scenario must be recorded at a single step count.
    """
    kinds = {}

    def kind(r):
        name = (r["problem"], json.dumps(r.get("problem_options") or {},
                                         sort_keys=True))
        if name not in kinds:
            kinds[name] = problems.get(r["problem"]).make_constraint(
                **(r.get("problem_options") or {})
            ).kind
        return kinds[name]

    cells: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        label = r.get("problem_label") or problems.get(r["problem"]).label
        row = (kind(r) != constraints.EQUALITY, r["problem"], label,
               r["steps"])
        column = r.get("variant") or r["method"]
        if column in cells.setdefault(row, {}):
            raise ValueError(
                f"two results for {column!r} on {label} at {r['steps']} "
                "steps; remove the stale one from the results directory"
            )
        cells[row][column] = r

    columns = list(LATEX_COLUMNS)
    known = {name for name, _ in columns}
    for per_row in cells.values():
        for name, r in per_row.items():
            if name not in known:
                columns.append((name, r["method_label"]))
                known.add(name)

    order = list(problems.all_problems())
    rows = sorted(cells, key=lambda k: (
        k[0], order.index(k[1]) if k[1] in order else 99, k[2], k[3],
    ))
    scenarios = [k[:3] for k in rows]
    for k in rows:
        if scenarios.count(k[:3]) > 1:
            raise ValueError(
                f"{k[2]} was run at more than one step count; the table has "
                "no Steps column, so keep one resolution per scenario"
            )

    groups = "\n".join(
        f"& \\multicolumn{{2}}{{c}}{{{header}}}" for _, header in columns
    )
    rules = "".join(
        f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
        for i in range(len(columns))
    )
    lines = [
        r"\begin{tabular}{l" + "cc" * len(columns) + "}",
        r"\toprule",
        groups + r" \\",
        rules,
        " & ".join(["Problem"] + ["Time", "Viol."] * len(columns)) + r" \\",
        r"\midrule",
    ]

    for i, row in enumerate(rows):
        if i and row[0] != rows[i - 1][0]:
            lines.append(r"\midrule")
        per_row = cells[row]
        fastest = min(r["mean_time_ms"] for r in per_row.values())
        out = [row[2]]
        for name, _ in columns:
            r = per_row.get(name)
            if r is None:
                out += ["---", "---"]
                continue
            time_ms = f"{r['mean_time_ms']:.1f}"
            if r["mean_time_ms"] <= LATEX_TIME_FACTOR * fastest:
                time_ms = rf"\textbf{{{time_ms}}}"
            out += [time_ms,
                    _latex_violation(r["mean_violation"],
                                     r.get("num_nan", 0))]
        lines.append(" & ".join(out) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _ran_the_baseline_scenario(record, want) -> bool:
    """Whether the record's problem options match the baseline's."""
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
    """Compare results to ``experiments/baseline.json``; return regressions.

    Only unprojected cases at ``baseline_steps`` are compared. Slower runs
    with a better violation are reported as a trade, not a regression.
    """
    baseline_path = Path(baseline_path or "experiments/baseline.json")
    if not baseline_path.exists():
        return [f"no baseline at {baseline_path}"]

    baseline = json.loads(baseline_path.read_text())["results"]
    regressions = []

    for r in records:
        if r["steps"] != baseline_steps:
            continue
        if float(r.get("config", {}).get("num_projection_iters", 0)) > 0:
            continue
        # The baseline uses the old method name.
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
            # Ignore changes at the float32 noise floor.
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
