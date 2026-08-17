# Constrained Flow Matching via Lagrangian Dual Flows

This repository implements the constrained flow matching method described in the
paper [Constrained Flow Matching via Lagragian Dual
Flows](https://arxiv.org/abs/2607.04513) by Vince Kurtz and Alexander Davydov.

This method takes a pre-trained [flow matching](https://arxiv.org/abs/2210.02747)
model
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

## Examples

### 2D toy datasets

Five simple 2D examples live in `examples/`: `bimodal`, `spiral`, `star`,
`unit_circle`. Each supports `--train` and `--generate`; `star` and
`unit_circle` also support `--generate_constrained` (unit-norm constraint).

```bash
# train
uv run -m examples.bimodal --train
uv run -m examples.spiral --train
uv run -m examples.star --train
uv run -m examples.unit_circle --train

# generate (unconstrained)
uv run -m examples.bimodal --generate
uv run -m examples.spiral --generate
uv run -m examples.star --generate
uv run -m examples.unit_circle --generate

# generate (constrained to the unit circle)
uv run -m examples.star --generate_constrained
uv run -m examples.unit_circle --generate_constrained
```

Training takes about a minute on a laptop CPU.

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
uv run -m examples.obstacles --train

# unconditional generation
uv run -m examples.obstacles --generate

# plan around a new, randomly generated scene
uv run -m examples.obstacles --generate_constrained
uv run -m examples.obstacles --generate_constrained --num-obstacles 3 --scene-seed 7
```

`--num-obstacles` and `--scene-seed` control the test scene. With the default
closed-form slack, all 64 generated paths clear every obstacle across 1-3
obstacle scenes, while 25-90% of unconstrained samples collide.

`--slack ode` switches to carrying the slack variable as an extra ODE state
(see `generate_inequality_constrained`); `--penalty-weight` and
`--rescale-factor` override the per-mode default gains. The slack ODE needs
much gentler gains to stay stable and enforces the constraint less tightly,
especially as obstacles are added.

### MNIST

The MNIST example trains a UNet-based flow-matching model on handwritten digits
and supports inpainting: the top half of each image is fixed to a reference
sample and the model generates plausible completions.

```bash
# train (requires a GPU; takes ~30 minutes)
uv run -m examples.mnist --train

# unconditional generation
uv run -m examples.mnist --generate

# inpainting: fix top half, generate bottom half
uv run -m examples.mnist --generate_constrained
```

A pre-trained model is saved to `data/mnist_model.pkl` by default
(`--save-path` overrides this for all three commands).

### Further Details and Baselines

The main implementation of Lagragian Dual Flows is in `generate.py`.
[Physics-constrained flow matching](https://arxiv.org/abs/2506.04171) and
[pseudoinverse guidance](https://arxiv.org/abs/2310.04432) baselines are
implemented in `pcfm.py` and `pi_gdm.py` respectively.

## Paper Reproduction

To reproduce all examples in the paper, run
```bash
# train unconstrained flow matching models
uv run -m examples.star --train
uv run -m examples.mnist --train

# Create and save figures to plots/figures
uv run -m plots --regenerate
```

To reproduce Table 1 benchmarks, run
```bash
uv run ./make_benchmark_table.sh
```

