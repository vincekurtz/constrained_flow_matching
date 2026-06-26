# Constrained Flow Matching

This repository demonstrates several methods for modifying the flow field
```math
\dot{x} = v_\theta(x, t)
```
of a pre-trained flow-matching generative model to enforce equality constraints
```math
g(x) = 0.
```

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

### Avoiding (D3IL-style robotic manipulation)

A 2D surrogate of the D3IL "Avoiding" task (cf. HardFlow,
[arXiv:2511.08425](https://arxiv.org/abs/2511.08425)): an unconditional
flow-matching model is trained over whole demonstration *trajectories* that weave
from a fixed start to a goal line around six pillars. At inference we impose a
**new** constraint set the demonstrations never obeyed — an enlarged keep-out
circle around the first pillar plus two half-plane "funnel" constraints that cut
off the last two pillars — and steer toward it with smooth constraint guidance
added to the learned flow (soft enforcement, so satisfaction is not guaranteed).

```bash
# train the trajectory model (GPU recommended; ~1 minute)
uv run -m examples.avoiding --train

# unconstrained samples (many violate the inference-time constraints)
uv run -m examples.avoiding --generate

# constrained samples: guide toward the enlarged circle + funnel
uv run -m examples.avoiding --generate_constrained
```

Both generation commands print feasibility / violation / goal-reach metrics and
save a side-by-side figure to `plots/avoiding.png` (`--fig-path` overrides). The
demonstration data itself can be visualized with `uv run -m datasets.avoiding`.
