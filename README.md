# Constrained Flow Matching via Lagrangian Dual Flows

Code for [Constrained Flow Matching via Lagrangian Dual
Flows](https://arxiv.org/abs/2607.04513) by Vince Kurtz and Alexander Davydov.

**Lagrangian Dual Flows (LDF)** take a pre-trained [flow
matching](https://arxiv.org/abs/2210.02747) model
```math
\dot{x} = v_\theta(x, t)
```
and enforce inference-time constraints
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

Inequality constraints are also supported; see the paper for details.

> [!WARNING]
> This is research code. Expect rough edges and breaking changes.

## Install and Testing

```
uv sync --dev
uv sync --group render   # optional: MuJoCo, for the locomotion figures
uv run pytest
uv run ruff check
```

## Layout

```
cfm/            library
  core/         solvers, constraints, utilities
  methods/      LDF, baselines, and the method registry
  models/       flow architectures
  datasets/     training datasets
  cli.py        command line entry point
  sweep.py      benchmark sweeps
problems/       example problems and the problem registry
experiments/    sweep configs and the recorded performance baseline
plots.py        paper figures
```

Methods are registered in `cfm/methods/__init__.py` and problems in
`problems/__init__.py`. `uv run -m cfm.cli list` shows both.

## Methods

| Name | Description | Constraints |
|------|-------------|-------------|
| `ldf` | Lagrangian Dual Flows (ours) | equality, inequality |
| `penalty` | Quadratic penalty on constraint violation | equality, inequality |
| `pcfm` | [Physics-constrained flow matching](https://arxiv.org/abs/2506.04171) | equality |
| `pigdm` | [Pseudoinverse guidance](https://arxiv.org/abs/2310.04432) | equality |
| `cbf` | [SafeFlow](https://arxiv.org/abs/2504.08661) control barrier function filter | inequality |

## Examples

### 2-D toy problems

```bash
uv run -m cfm.cli train --problem star
uv run -m cfm.cli generate --problem star                  # unconstrained
uv run -m cfm.cli generate --problem star --method ldf
uv run -m cfm.cli generate --problem star --method ldf --constraint right_half
```

`bimodal`, `spiral`, `star`, and `unit_circle` each train in about a minute on a
small GPU. Default gains are set per problem and can be overridden on the
command line (`--penalty-weight`, `--rescale-factor`, `--dt`, ...).

### Obstacle avoidance

A flow model is trained on obstacle-free start-to-goal paths, parameterized as
cubic Bezier splines. At inference time, obstacle avoidance is imposed as
inequality constraints on a randomly generated scene.

```bash
uv run -m cfm.cli train --problem obstacles
uv run -m cfm.cli generate --problem obstacles --method ldf --dt 0.002 \
    --num-obstacles 3 --scene-seed 7
uv run -m cfm.cli generate --problem obstacles --method cbf --qp elastic
```

### D4RL locomotion

Walker2D and Hopper examples from
[SafeFlowMatcher](https://arxiv.org/abs/2509.24243). A temporal U-Net is
trained on 32-step windows of D4RL medium-expert data, and a torso height limit
$z_t + \phi v_{z,t} \le h_r$ is imposed at inference time. `h_r` and `phi`
follow [SafeDiffuser](https://arxiv.org/abs/2306.00148).

```bash
uv run -m cfm.cli train --problem walker2d
uv run -m cfm.cli generate --problem walker2d --method ldf
uv run -m cfm.cli generate --problem hopper --method ldf --height-limit 1.5
```

The first run downloads the D4RL data (~770 MB) into `data/d4rl/` from the
`imone/D4RL` HuggingFace mirror.

### MNIST

Inpainting: the top half of each image is fixed and the model fills in the
bottom half. Training requires a GPU.

```bash
uv run -m cfm.cli train --problem mnist
uv run -m cfm.cli generate --problem mnist --method ldf
```

Trained models are saved to `data/<problem>_model.pkl` (override with
`--save-path`).

## Benchmarks

```bash
# time a single method
uv run -m cfm.cli benchmark --problem star --method ldf --num-samples 20

# Table 1
uv run -m cfm.cli sweep experiments/table1.toml
uv run -m cfm.cli table --format markdown   # or latex
```

Each sweep case writes a JSON file to `results/`. Completed cases are skipped
unless `--force` is passed. `--check-baseline` fails if any case regressed
relative to `experiments/baseline.json`. See `experiments/table1.toml` for the
config format.

## Paper reproduction

```bash
for p in star mnist obstacles walker2d hopper; do
    uv run -m cfm.cli train --problem $p
done
uv run python plots.py --plot all --regenerate   # figures -> plots/figures
uv run -m cfm.cli sweep experiments/table1.toml
uv run -m cfm.cli table
```

## Contributing

- **New method:** add a module under `cfm/methods/` exposing
  `generate(model, normalizer, constraint, **cfg) -> Samples`, and register it
  in `cfm/methods/__init__.py`.
- **New problem:** add a `Problem` under `problems/` and register it in
  `problems/__init__.py`.
- **Golden tests:** `tests/goldens/*.npy` pin each method's output on a tiny
  fixed model. If a change is intended, regenerate with
  `uv run python -m tests.make_goldens`.
