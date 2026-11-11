"""Declarative benchmark sweeps.

A config is a list of ``[[block]]`` sections, each a problems x methods x steps
grid. Each case writes one result file, and cases already on disk are skipped.
"""

import hashlib
import json
import math
import platform
import re
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
                if (problem.supported_methods is not None
                        and method_name not in problem.supported_methods):
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
    constraint = problem.constraint_for(case.method, **case.problem_options)

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
    num_failed = int(jnp.sum(~jnp.isfinite(v)))
    v = jnp.where(jnp.isfinite(v), v, jnp.nan)
    with open(problem.checkpoint_path, "rb") as f:
        checkpoint_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
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
        "num_nan": num_failed,
        "num_failed": num_failed,
        "git_sha": _git_sha(),
        "platform": platform.platform(),
        "devices": [str(d) for d in jax.devices()],
        "checkpoint_sha256": checkpoint_sha256,
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
        note = (f" ({r['num_nan']} failed)"
                if r.get("num_nan") else "")
        violation = (f"{r['mean_violation']:.2e}"
                     if math.isfinite(r["mean_violation"]) else "failed")
        body.append([
            r.get("problem_label") or problems.get(r["problem"]).label,
            str(r["steps"]),
            r["method_label"],
            f"{r['mean_time_ms']:.2f}",
            f"{violation}{note}",
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


# (variant, row label, citation key) for the LaTeX rows, one row per
# variant; unlisted variants are appended. Each ``_projected`` variant sits
# indented under its base method. The paper keeps the citations as LaTeX
# comments so the labels stay short.
LATEX_ROWS = [
    ("penalty", "Penalty", None),
    ("penalty_projected", None, None),
    ("time_varying_penalty", "Time-varying penalty", None),
    ("time_varying_penalty_projected", None, None),
    ("cbf", "CBF", "safeflow2025"),
    ("pcfm", "PCFM", "utkarsh2025pcfm"),
    ("pigdm", r"$\Pi$GDM", "pokle2024training"),
    ("pigdm_projected", None, None),
    ("opc", "OPC", "peng2024improving"),
    ("opc_projected", None, None),
    ("dps", "DPS", "chung2023dps"),
    ("dps_projected", None, None),
    ("ldf", "LDF (ours)", None),
    ("ldf_projected", None, None),
]
LATEX_PROJECTED_LABEL = r"\quad + proj."
# Our method's rows are set off from the baselines by a rule.
LATEX_OURS = "ldf"

# Bold times are within this factor of the fastest for their problem; bold
# violations are below the threshold, near single-precision round-off.
LATEX_TIME_FACTOR = 1.2
LATEX_VIOLATION_THRESHOLD = 1e-6

# The constraint-kind header already says which star variant a column is.
_KIND_SUFFIX = re.compile(r"\s*\((?:eq|ineq)\.\)$")


def _latex_violation(value: float, num_nan: int) -> str:
    r"""Format a violation with the paper's ``\sci``/``\scib`` macros."""
    if not math.isfinite(value):
        return r"\textit{failed}"
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

    One row per method variant, a Time/Viol. column pair per scenario,
    with equality scenarios grouped left of inequality ones. Each
    scenario must be recorded at a single step count.
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
        scenario = (kind(r) != constraints.EQUALITY, r["problem"], label,
                    r["steps"])
        variant = r.get("variant") or r["method"]
        if variant in cells.setdefault(scenario, {}):
            raise ValueError(
                f"two results for {variant!r} on {label} at {r['steps']} "
                "steps; remove the stale one from the results directory"
            )
        cells[scenario][variant] = r

    order = list(problems.all_problems())
    scenarios = sorted(cells, key=lambda k: (
        k[0], order.index(k[1]) if k[1] in order else 99, k[2], k[3],
    ))
    names = [k[:3] for k in scenarios]
    for k in scenarios:
        if names.count(k[:3]) > 1:
            raise ValueError(
                f"{k[2]} was run at more than one step count; the table has "
                "no Steps column, so keep one resolution per scenario"
            )

    rows = [(name, label, cite) for name, label, cite in LATEX_ROWS]
    known = {name for name, _, _ in rows}
    for per_scenario in cells.values():
        for name, r in per_scenario.items():
            if name not in known:
                rows.append((name, r["method_label"], None))
                known.add(name)
    present = {name for per in cells.values() for name in per}
    rows = [row for row in rows if row[0] in present]

    # Header: constraint kind, then problem, then Time/Viol.
    lines = [
        r"\begin{tabular}{@{}l" + "cc" * len(scenarios) + "@{}}",
        r"\toprule",
    ]
    spans, rules, col = [], [], 2
    for inequality in (False, True):
        width = 2 * sum(k[0] == inequality for k in scenarios)
        if not width:
            continue
        title = "Inequality" if inequality else "Equality"
        spans.append(rf"\multicolumn{{{width}}}{{c}}{{{title} constraints}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + width - 1}}}")
        col += width
    lines += [" & ".join([""] + spans) + r" \\", "".join(rules)]
    lines.append(" & ".join([""] + [
        rf"\multicolumn{{2}}{{c}}{{{_KIND_SUFFIX.sub('', k[2])}}}"
        for k in scenarios
    ]) + r" \\")
    lines.append("".join(
        rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}"
        for i in range(len(scenarios))
    ))
    lines += [
        " & ".join(["Method"] + ["Time", "Viol."] * len(scenarios)) + r" \\",
        r"\midrule",
    ]

    fastest = {
        k: min((r["mean_time_ms"] for r in cells[k].values()
                if math.isfinite(r["mean_violation"])), default=math.inf)
        for k in scenarios
    }
    for i, (name, label, cite) in enumerate(rows):
        family = name.removesuffix("_projected")
        if i and family != rows[i - 1][0].removesuffix("_projected"):
            lines.append(r"\midrule" if family == LATEX_OURS
                         else r"\addlinespace")
        if label is None:
            label = LATEX_PROJECTED_LABEL
        if cite is not None:
            label += f"% ~\\citep{{{cite}}}\n"
        out = [label]
        for k in scenarios:
            r = cells[k].get(name)
            if r is None:
                out.append(r"\multicolumn{2}{c}{---}")
                continue
            time_ms = f"{r['mean_time_ms']:.1f}"
            if (math.isfinite(r["mean_violation"])
                    and r["mean_time_ms"] <= LATEX_TIME_FACTOR * fastest[k]):
                time_ms = rf"\textbf{{{time_ms}}}"
            out += [time_ms,
                    _latex_violation(r["mean_violation"],
                                     r.get("num_nan", 0))]
        lines.append(" & ".join(out) + r" \\")

    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


# The paper's Table 1 caption, extended for the added baselines.
LATEX_CAPTION = "\n".join([
    "Mean per-sample generation time (ms) and constraint violation across "
    "{num_samples} samples.",
    r"Violation is $\norm{{g(x_1)}}$ for equality-constrained problems "
    "(left) and",
    r"$\norm{{\relu(h(x_1))}}$ for inequality-constrained problems (right).",
    r"``+ proj.'' rows apply a final Gauss-Newton projection to the samples "
    "of the method above.",
    r"Bold times are within {time_percent}\% of the lowest time for that "
    r"problem; bold violations are below $10^{{{threshold_exponent}}}$",
    "(near machine precision for single-precision float).",
    r"$^\dagger$At least one NaN error produced during generation; "
    "average shown is across non-NaN samples.",
    r"\textit{{failed}}: every sample produced a NaN error; such methods "
    "are excluded from the bold-time comparison.",
    "---: the method does not handle that constraint type.",
])


def render_latex_float(records, num_samples: int) -> str:
    r"""Wrap :func:`render_latex_table` in the paper's ``table*`` float.

    The output pastes into the paper as is; it uses the paper's ``\sci``,
    ``\scib``, ``\norm`` and ``\relu`` macros and ``booktabs``.
    """
    caption = LATEX_CAPTION.format(
        num_samples=num_samples,
        time_percent=round(100 * (LATEX_TIME_FACTOR - 1)),
        threshold_exponent=round(math.log10(LATEX_VIOLATION_THRESHOLD)),
    )
    return "\n".join([
        r"\begin{table*}[t]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{%",
        render_latex_table(records),
        "}",
        rf"\caption{{{caption}}}",
        r"\vspace{-1.05em}",
        r"\label{tab:generation_times}",
        r"\end{table*}",
    ])


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
