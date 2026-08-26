# Constrained Flow Matching via Lagrangian Dual Flows

This repository implements the constrained flow matching method described in the
paper [Constrained Flow Matching via Lagragian Dual
Flows](https://arxiv.org/abs/2607.04513) by Vince Kurtz and Alexander Davydov.

This method, **Lagrangian Dual Flows (LDF)**, takes a pre-trained [flow
matching](https://arxiv.org/abs/2210.02747) model
```math
\dot{x} = v_\theta(x, t)
```
and enforces inference-time constraints
```math
g(x) = 0
```
by augmenting the denoising ODE with Lagrangian dual dynamics
```math
\begin{aligned}
& \dot{x} = v_\theta(x, t) - \nabla g(x)^\top\lambda - \nabla g(x)^\top g(x), \\
& \dot{\lambda} = g(x) / (1-t)^2.
\end{aligned}
```

Inequality constraints are also supported: see the paper for full details.

> [!WARNING]
> This is active research code, not a stable library. Expect rough edges: the
> API may change without notice, some interfaces are undocumented, and things
> may break. It is provided as-is, with no guarantee of support or maintenance.
> Use at your own risk.

## Install

Install dependencies with [uv](https://docs.astral.sh/uv/):
```
uv sync --dev
```

Run unit tests:
```
uv run pytest
```

Run lint checks:
```
uv run ruff check
```

## Layout

```
cfm/            the library
  core/         solver scaffolding, constraints, common utilities
  methods/      LDF and baselines, plus the method registry
  models/       flow architectures (MLP, UNet, normalizer)
  datasets/     training datasets
  cli.py        the command line entry point
  sweep.py      declarative benchmark sweeps
problems/       the example problems, and the problem registry
experiments/    sweep configurations and the recorded performance baseline
plots.py        paper figures
```

Everything is driven by two registries. `cfm/methods/__init__.py` lists the
generation methods; `problems/__init__.py` lists the examples. Adding either
one is a single entry, and the CLI, the benchmark, the sweep and the tests
pick it up automatically.

To see what is registered, and which methods can handle which problem:

```bash
uv run -m cfm.cli list
```

## Methods

| Name | Description | Constraints |
|------|-------------|-------------|
| `ldf` | Lagrangian Dual Flows, the method of the paper | equality, inequality |
| `penalty` | Penalty-only ablation: LDF with the multipliers frozen at zero | equality, inequality |
| `pcfm` | [Physics-constrained flow matching](https://arxiv.org/abs/2506.04171) | equality |
| `pigdm` | [Pseudoinverse guidance](https://arxiv.org/abs/2310.04432) | equality |
| `cbf` | [SafeFlow](https://arxiv.org/abs/2504.08661) control barrier function filter | inequality |

The `penalty` baseline is LDF with `rescale_factor = 0`, which freezes the
Lagrange multipliers at their zero initialization and collapses the drift to
$\dot{x} = v_\theta - \nabla g^\top g$: a pure quadratic penalty with no dual
dynamics. 

## Examples

### Training and generating

```bash
# train
uv run -m cfm.cli train --problem star

# generate (unconstrained)
uv run -m cfm.cli generate --problem star

# generate with a constraint
uv run -m cfm.cli generate --problem star --method ldf
uv run -m cfm.cli generate --problem star --method penalty
uv run -m cfm.cli generate --problem star --method pcfm
```

The 2-D problems (`bimodal`, `spiral`, `star`, `unit_circle`) train in about a
minute on a laptop CPU. `star` and `unit_circle` impose a unit-norm
constraint; `unit_circle` also offers a right-half-plane inequality with
`--constraint right_half`.

Per-problem gains are registered with the problem, so the commands above use
the settings the paper uses. Any of them can be overridden on the command
line (`--penalty-weight`, `--rescale-factor`, `--dt`, ...).

### Obstacle avoidance

A point-mass robot plans a path through a field of circular obstacles. The
flow model is trained *unconditionally* on wiggly start-to-goal paths and never
sees an obstacle. At inference time a brand-new scene is sampled and obstacle
avoidance is imposed with inequality constraints

```math
h(x) = r_j + \text{clearance} - \|p_i - c_j\| \le 0
```

for every point `p_i` sampled along the path and every obstacle `(c_j, r_j)`.
The decision variables are the interior knots of a cubic Bezier spline, so the
robot's path is smooth however the constraints push the knots around. The start
and goal are fixed and shared by every path.

```bash
# train (takes about 30 seconds)
uv run -m cfm.cli train --problem obstacles

# unconditional generation
uv run -m cfm.cli generate --problem obstacles

# plan around a new, randomly generated scene
uv run -m cfm.cli generate --problem obstacles --method ldf --dt 0.002
uv run -m cfm.cli generate --problem obstacles --method ldf --dt 0.002 \
    --num-obstacles 3 --scene-seed 7
```

`--num-obstacles` and `--scene-seed` control the test scene. With the default
closed-form slack, all 64 generated paths clear every obstacle across 1-3
obstacle scenes.

`--slack ode` switches to carrying the slack variable as an extra ODE state
instead of substituting its closed-form minimizer. The slack ODE needs much
gentler gains to stay stable and enforces the constraint less tightly,
especially as obstacles are added.

`--method cbf` runs the SafeFlow control barrier function baseline on the same
scene, with `--phi0` and `--omega` setting the barrier gains. The safety-filter
QP is solved with [qpax](https://github.com/kevin-tracy/qpax); barrier
conditions that cannot all be met at once make it infeasible, which
`--qp exact` raises on and `--qp elastic` relaxes, pricing violation at
`--qp-penalty` per unit. The exact solve is fragile enough that Table 1's
obstacle rows ask for the relaxation: at their `dt = 0.002` the barrier
conditions are mutually infeasible on essentially every sample and the exact
solve fails outright.

```bash
uv run -m cfm.cli generate --problem obstacles --method cbf --qp elastic
```

### MNIST

The MNIST example trains a UNet-based flow-matching model on handwritten digits
and supports inpainting: the top half of each image is fixed to a reference
sample and the model generates plausible completions.

```bash
# train (requires a GPU; takes ~30 minutes)
uv run -m cfm.cli train --problem mnist

# unconditional generation
uv run -m cfm.cli generate --problem mnist

# inpainting: fix top half, generate bottom half
uv run -m cfm.cli generate --problem mnist --method ldf
```

Trained models are saved to `data/<problem>_model.pkl`; `--save-path`
overrides this for every subcommand.

## Benchmarks and experiments

Time a single method, one sample at a time:

```bash
uv run -m cfm.cli benchmark --problem star --method ldf --num-samples 20
```

Reproduce Table 1 with a declarative sweep. Each case writes a JSON result to
`results/`, and the table is rendered from those files rather than scraped
from stdout:

```bash
uv run -m cfm.cli sweep experiments/table1.toml
uv run -m cfm.cli table --format markdown   # or latex
```

Cases whose exact configuration already has a result are skipped, so an
interrupted sweep resumes; pass `--force` to re-run them.

A config is a list of `[[block]]` sections, each its own
problems x methods x steps grid. One table needs more than one grid: the
obstacle rows run their own scenes, at a much finer step size than the 2-D
and MNIST rows, and a block carries its own `problem_options`,
`num_samples` and `problem_label` so the same problem can appear twice under
two scenes. A config with no `[[block]]` is itself the single block.

A row of the table is named by a `[[variants]]` entry: a registered method
with the gains that define the row pinned.

```toml
[[variants]]
name = "ldf_projected"
method = "ldf"
label = "LDF + projection"
gains = { num_projection_iters = 2 }
```

That is how one method appears on two rows -- LDF as the flow alone, and LDF
followed by the Gauss-Newton projection -- without being registered twice.
Each row gets its own label and its own result file. A `[[gains]]` override
may name a variant, or name the method and reach every one of its rows;
`method = ["ldf", "penalty"]` reaches several at once. A variant's own gains
are applied last, so tuning `ldf` tunes both LDF rows rather than collapsing
them into one.

`experiments/baseline.json` records per-method timing and violation from
before the repository was reorganized. `--check-baseline` fails the sweep if
any case regressed:

```bash
uv run -m cfm.cli sweep experiments/table1.toml --check-baseline
```

## Paper reproduction

```bash
# train unconstrained flow matching models
uv run -m cfm.cli train --problem star
uv run -m cfm.cli train --problem mnist
uv run -m cfm.cli train --problem obstacles

# create and save figures to plots/figures
uv run python plots.py --plot all --regenerate

# Table 1
uv run -m cfm.cli sweep experiments/table1.toml
uv run -m cfm.cli table
```

## Notes for contributors

**Adding a method.** Implement
`generate(model, normalizer, constraint, **cfg) -> Samples` in a module under
`cfm/methods/`, then add a `Method` entry to `cfm/methods/__init__.py`. The
CLI, the sweep and the parameterized contract tests in
`tests/test_methods.py` pick it up with no further wiring.

**Adding a problem.** Add a `Problem` entry in a module under `problems/` and
list it in `problems/__init__.py`.

**Golden tests.** `tests/goldens/*.npy` pin the numerical output of every
method on a fixed tiny model and seed. They exist so that refactoring can be
verified rather than reviewed, and a failure means an algorithm changed. If a
change is intended, regenerate with `uv run python -m tests.make_goldens` and
review the array diff like any other change.
