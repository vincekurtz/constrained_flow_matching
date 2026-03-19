Can contraction theory inform or improve flow-matching generative models?

## Install

Install dependencies and such with [uv](https://docs.astral.sh/uv/):
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

## Usage

There are several simple 2D examples in the `examples/` directory
(e.g., `bimodal | star | spiral | unit_circle`). For each example, you can:

- Train and save a flow-matching model:
```
uv run -m examples.bimodal --train
```
- Use the trained model to generate samples, then visualize the samples
alongside the training data:
```
uv run -m examples.bimodal --generate
```

For these simple examples, training takes about a minute on a laptop.
