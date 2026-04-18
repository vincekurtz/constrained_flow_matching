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
