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
| `time_varying_penalty` | Quadratic penalty scaled by `1/(1-t)^2` | equality, inequality |
| `pcfm` | [Physics-constrained flow matching](https://arxiv.org/abs/2506.04171) | equality |
| `pigdm` | [Pseudoinverse guidance](https://arxiv.org/abs/2310.04432) | equality |
| `opc` | ΠGDM with [estimated posterior covariance](https://arxiv.org/abs/2402.02149) | equality |
| `dps` | [Diffusion posterior sampling](https://arxiv.org/abs/2209.14687) | equality |
| `cbf` | [SafeFlow](https://arxiv.org/abs/2504.08661) control barrier function filter | inequality |

`--num-projection-iters` works with every method. LDF and `penalty` apply
it themselves; the other methods have the same final Gauss-Newton
projection applied to their samples, in normalized coordinates and
included in the timing. Table 1 reports each baseline with and without
it. PCFM and CBF already end with their own 20-step projections.

`opc` reads a per-checkpoint variance table. After training, build it with
`uv run -m experiments.posterior_variance --problem <name>`.

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

### Pretrained ImageNet inpainting

`imagenet` uses the pretrained [JiT-H/32](https://github.com/LTH14/JiT)
model to generate 512 × 512 RGB images, conditioned on the lakeside class.
No training or dataset is required. All samplers use the same velocity model
and image state through the same API as MNIST. LDF applies equality
constraints to the observed pixels while generating the whole image, with
no pixel pasting or final projection.

```bash
# Tested on a 32 GB RTX 5090; generate one image at a time.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90
uv run -m experiments.prepare_imagenet --download

uv run -m cfm.cli generate --problem imagenet --seed 2 \
    --no-plot --output-dir results/image_unconstrained
uv run -m cfm.cli generate --problem imagenet --method pigdm --seed 2 \
    --no-plot --output-dir results/image_pigdm
uv run -m cfm.cli generate --problem imagenet --method ldf --seed 2 \
    --no-plot --output-dir results/image_ldf
uv run -m cfm.cli generate --problem imagenet --method pcfm --seed 2 \
    --no-plot --output-dir results/image_pcfm
```

Preparation verifies the upstream checkpoint and writes
`data/imagenet_native_model.pkl`; allow about 15.3 GB for both files.
Re-run preparation if upgrading from the earlier wrapped-model example.
Inference runs in JAX; PyTorch is only needed to read the upstream weights.

The bundled reference uses a 192 × 192 missing region. Use
`--reference image.png --mask-path mask.png` for custom inputs. Masks must
be 512 × 512, white for observed pixels and black for missing pixels.
References are center-cropped and resized. Raw outputs are saved before
clipping for display.

The LDF defaults are penalty 40, dual factor 1, exponent 1,
`initial_dual_scale=-1`, and Midpoint `dt=0.0005`. This initialization
cancels the initial penalty force. PiGDM and PCFM use an equivalent
residual-norm calculation to reduce memory usage; LDF retains one equality
and dual per observed RGB value. PCFM uses 10 steps, one projection per
step, and disables its optional relaxed correction and final projection.
All three use the native model time and image coordinates.

See the [native-model comparison](plots/figures/imagenet_native_comparison.png)
and [five-case gallery](plots/figures/imagenet_native_inpainting_five.png).
These remain exploratory results: LDF closely matches the observed pixels,
but both methods can produce visible artifacts in the missing region.
The gallery retains all five random cases and uses one configuration per
method; case 1 was used to tune PiGDM's integration settings.

To reproduce the gallery with adaptive PiGDM integration, run each stage
and inspect its outputs before proceeding:

```bash
uv run -m experiments.imagenet_inpainting_gallery --stage references \
    --pigdm-settings experiments/imagenet_native/gallery_pigdm.json
uv run -m experiments.imagenet_inpainting_gallery --stage pigdm \
    --pigdm-settings experiments/imagenet_native/gallery_pigdm.json
uv run -m experiments.imagenet_inpainting_gallery --stage ldf \
    --pigdm-settings experiments/imagenet_native/gallery_pigdm.json
uv run -m experiments.imagenet_inpainting_gallery --stage figure
```

Raw samples and records are saved to `results/imagenet_native_gallery`.
Figures have adjacent JSON files with settings, input hashes, and pixel
errors. Further tuning plans are in [experiments/imagenet_native](experiments/imagenet_native),
for use with `uv run -m experiments.imagenet_native_tuning --help`.
See `NOTICE` and `licenses/JiT-MIT.txt` for model attribution.

The [three-method calibration gallery](plots/imagenet_three_method_gallery/representatives.png)
adds PCFM to the existing five cases. PCFM settings were selected from
[twelve candidates](plots/imagenet_pcfm_calibration.csv) by missing-region
FID on those cases; visible artifacts remain. Reproduce that search with
`uv run --group evaluation -m experiments.imagenet_pcfm_calibration --stage run`.

The larger study uses new paired reference images, masks, and sampler
seeds. It saves whole-image and missing-region
[Clean-FID](https://github.com/GaParmar/clean-fid), paired KID,
observed-pixel errors, missing-region RMSE, runtime, and a figure of the
first five cases. Tables are exported as CSV, Markdown, and JSON.

```bash
uv sync --group evaluation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS=false
uv run --group evaluation -m experiments.imagenet_inpainting_study \
    --stage init --count 256 --output-dir results/imagenet_study_256_selected
uv run --group evaluation -m experiments.imagenet_inpainting_study \
    --stage all --output-dir results/imagenet_study_256_selected \
    --batch-size 4 --report-dir plots/imagenet_study_256
```

The default references are new samples from the lakeside model, making this
a **synthetic-reference pilot**, not a real-image benchmark. For real images,
add `--reference-dir /path/to/images` at initialization and use a checkpoint
conditioned on the appropriate class. FID is sensitive to sample count;
256 cases are exploratory ([FID bias study](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chong_Effectively_Unbiased_FID_and_Inception_Score_and_Where_to_Find_CVPR_2020_paper.pdf)).
Missing-region FID helps expose artifacts that
the large observed area can hide in full-image FID. Paired KID excludes
matched reference/output pairs from its cross-kernel average.

Plans and raw outputs are immutable on resume; rerunning `--stage all`
skips validated samples. `run_status.json` tracks the current stage.
Batching applies to the fixed-step samplers; adaptive PiGDM runs singly.
Numerical failures are recorded and counted;
distribution metrics use the same successful cases for all methods.
An unfinished study cannot produce a completed table. Individual stages
are `references`, `pcfm`, `pigdm`, `ldf`, and `evaluate`. The methods use
fixed settings with different computational costs; this is not a matched
compute or exhaustive hyperparameter comparison.

#### DPS and OPC baselines

DPS and OPC were added to the completed study without changing its plan.
`--stage extend` records their settings in `extensions.json`; each new
sample stores the hash of those settings, so existing samples stay valid.

```bash
# OPC's r_t^2 table, from 64 fresh unconstrained samples of the checkpoint
uv run -m experiments.imagenet_posterior_variance
# DPS step size and OPC guidance scale on the five calibration cases
uv run --group evaluation -m experiments.imagenet_baseline_calibration --stage run
S="uv run --group evaluation -m experiments.imagenet_inpainting_study \
    --output-dir results/imagenet_study_256_selected"
$S --stage extend
$S --stage dps --batch-size 4
$S --stage opc --batch-size 4
$S --stage evaluate --report-dir plots/imagenet_study_256
# paper figure and LaTeX table
uv run -m experiments.imagenet_inpainting_paper
```

Both use 500 Midpoint steps (1000 network evaluations, as in DPS) and are
calibrated on the five gallery cases by missing-region FID
([summary](plots/imagenet_baseline_calibration.csv)). DPS uses
`zeta' = 0.1`; larger steps added artifacts without fitting the observed
pixels better. OPC's estimated `r_t^2` is 12-140x below PiGDM's Gaussian
prior for this model, which makes the guidance very stiff: adaptive Tsit5
exceeds PiGDM's 2000-step cap, and PiGDM's guidance scale 0.6 diverges with
fixed steps. OPC therefore uses guidance scale 0.002 (grid 0.6 to 0.0006).
Neither baseline closely matches the observed pixels at this budget.

`experiments.imagenet_inpainting_paper` writes
[`plots/figures/imagenet_inpainting_comparison.png`](plots/figures/imagenet_inpainting_comparison.png)
(and `.pdf`), the first five planned cases without selection, and
[`plots/imagenet_inpainting.tex`](plots/imagenet_inpainting.tex), a complete
`table` float with FID and paired KID for the whole image and the hole.

`experiments.imagenet_inpainting_projection` writes
[`plots/figures/imagenet_inpainting_projection.png`](plots/figures/imagenet_inpainting_projection.png)
(and `.pdf`): one selected case (107) with every method's raw output and the
same output after an exact final projection (copying the observed pixels,
which is what one Gauss-Newton step does for this linear constraint).
`--rank N` lists the cases where projected baselines show the largest seams
relative to LDF.

## Benchmarks

```bash
# time a single method
uv run -m cfm.cli benchmark --problem star --method ldf --num-samples 20

# time all methods and all examples
uv run -m cfm.cli sweep experiments/table1.toml
uv run -m cfm.cli table --format markdown   # or latex
```

Each sweep case writes a JSON file to `results/`. Completed cases are skipped
unless `--force` is passed. `--check-baseline` fails if any case regressed
relative to `experiments/baseline.json`. See `experiments/table1.toml` for the
config format.

The time-varying penalty uses
`xdot = v(x,t) - c^2 J_g(x)^T g(x)/(1-t)^2`, with positive-part residuals
for inequalities. As with the constant penalty, `--penalty-weight` sets
the residual scale `c`, so the quadratic coefficient is `c^2`. This method
has no dual state or final projection. It stops at `t1 = 1-dt` by default
(the usual final saved time of LDF); its Python API accepts an explicit
`t1 < 1`. The singular denominator is not clipped. Table 1 uses the same
fixed-step Midpoint resolution and residual weights as the constant
penalty; failed samples are counted, and all-failed cells display `failed`.

The adaptive star/unit-circle study produces two additional figures:

```bash
uv run python plots.py --plot time_varying_penalty --regenerate
```

It compares constant penalty, time-varying penalty, and unprojected LDF
with the same 64 initial samples, `t1=0.99`, and adaptive Dopri5
(`rtol=atol=1e-5`, initial step `0.001`, no minimum step). Denoising steps
count accepted **and rejected** steps for the jointly integrated batch.
All runs use float32 and a 100,000-step budget. The cache in
`plots/data/time_varying_penalty.json` records per-sample violations,
accepted/rejected steps, failures, settings, and the checkpoint hash.
Failures are marked at the top of the penalty-weight figure and excluded
from the cost-versus-violation curve because they have no terminal sample.
PNG and vector PDF outputs are in `plots/figures/time_varying_penalty_*`.

`uv run -m experiments.table1 --run` runs the configured cases and exports
`plots/table1.tex`, `plots/table1.csv`, and `plots/data/table1.json`.
`plots/table1.tex` is the paper's complete `table*` float (caption and
label included) and uses the paper's `\sci`, `\scib`, `\norm`, and `\relu`
macros, so it can replace the table in the paper source directly.
Use `--available-only` to explicitly allow missing checkpoints, and
`--force` to refresh timings. The checked-in export covers every case,
timed on one RTX 5090 GPU; the device and checkpoint hash of each case are
recorded in the JSON. Pending cases, if any, are listed there separately
from numerical failures.

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
