Can contraction theory inform or improve flow-matching generative models?

## Usage

Install dependencies and such with [uv](https://docs.astral.sh/uv/):
```
uv sync --dev
```

Train a flow-matching model to generate data from a toy bimodal distribution:
```
uv run -m examples.bimodal --train
```

Use the trained model to generate samples, and visualize them alongside the
training data:
```
uv run -m examples.bimodal --generate
```
