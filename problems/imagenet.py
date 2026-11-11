"""512px landscape inpainting with the pretrained JiT-H/32 flow."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageOps

from cfm.core import checkpoint
from cfm.core.constraints import equality
from problems import Problem

IMAGE_SHAPE = (512, 512, 3)
REFERENCE_PATH = Path(__file__).parent / "assets/imagenet_reference.png"
CHECKPOINT_PATH = "data/imagenet_native_model.pkl"


@jax.custom_jvp
def pixel_residual_norm(residual):
    """Euclidean norm with the zero subgradient at exact satisfaction."""
    return jnp.linalg.norm(residual)


@pixel_residual_norm.defjvp
def _pixel_residual_norm_jvp(primals, tangents):
    (r,), (dr,) = primals, tangents
    norm = pixel_residual_norm(r)
    return norm, jnp.vdot(r, dr) / jnp.where(norm > 0, norm, 1.)


def reference_and_mask(reference=None, mask="interior", mask_path=None, **_):
    """Load RGB in [0, 1]; True/white means observed, black means missing.

    References are center-cropped to a square and resized to 512px. Custom
    masks must already be 512px square to avoid silently misaligned pixels.
    """
    with Image.open(reference or REFERENCE_PATH) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (512, 512), Image.Resampling.LANCZOS)
        ref = np.asarray(image, dtype=np.float32) / 255
    if mask_path is not None:
        with Image.open(mask_path) as image:
            if image.size != (512, 512):
                raise ValueError("Custom masks must be 512 x 512 pixels")
            observed = np.asarray(image.convert("L")) >= 128
    elif mask == "interior":
        observed = np.ones((512, 512), dtype=bool)
        observed[288:480, 96:288] = False
    elif mask == "center":
        observed = np.ones((512, 512), dtype=bool)
        observed[160:352, 160:352] = False
    elif mask == "lower-half":
        observed = np.zeros((512, 512), dtype=bool)
        observed[:256] = True
    elif mask == "lower-left":
        observed = np.ones((512, 512), dtype=bool)
        observed[256:, :384] = False
    else:
        raise ValueError(f"Unknown inpainting mask: {mask}")
    if not observed.any() or observed.all():
        raise ValueError("Mask must contain both observed and missing pixels")
    return ref, observed


def make_constraint(reduction="pixels", **options):
    reference, observed = reference_and_mask(**options)
    return fixed_pixel_constraint(reference, observed, reduction=reduction)


def fixed_pixel_constraint(reference, observed, *, reduction="pixels"):
    """Linear fixed-pixel equalities, optionally reduced exactly for PiGDM/PCFM.

    LDF uses r=x[observed]-reference[observed], one multiplier per RGB value.
    PiGDM uses ||r||=0: the same feasible set and, under a scalar normalizer,
    the same correction as the full vector r, with only one Jacobian row.
    PCFM also preserves its Gauss-Newton step and squared penalty.
    The reduction is not equivalent LDF dynamics.
    """
    observed = np.broadcast_to(np.asarray(observed)[..., None], reference.shape)
    indices = jnp.asarray(np.flatnonzero(observed))
    return indexed_pixel_constraint(reference, indices, reduction=reduction)


def indexed_pixel_constraint(reference, indices, *, reduction="pixels"):
    """The same equalities with precomputed flat RGB indices.

    Passing indices and reference as JIT arguments lets repeated experiments
    reuse a compiled sampler for different masks with the same pixel count.
    Indices must select distinct RGB values, as in ``fixed_pixel_constraint``.
    """
    if reduction not in ("pixels", "norm"):
        raise ValueError(f"Unknown fixed-pixel reduction: {reduction}")
    target = jnp.asarray(reference).ravel()[indices]

    def residual(x):
        return x.ravel()[indices] - target

    return equality(
        (residual if reduction == "pixels" else
         lambda x: jnp.atleast_1d(pixel_residual_norm(residual(x)))),
        name=f"RGB pixels fixed ({reduction})",
        violation=lambda x: jnp.max(jnp.abs(residual(x))),
    )


def make_model():
    """Load the converted checkpoint; no training or download on import."""
    return checkpoint.load(CHECKPOINT_PATH)[0]


def plot(problem, samples, constraint=None, *, output_path=None, show=True,
         **options):
    """Raw model output next to the reference and the observed pixels."""
    del problem
    x = np.asarray(samples.x)
    panels = []
    if constraint is not None:
        ref, observed = reference_and_mask(**options)
        panels.extend([
            (ref, "Reference"),
            (np.where(observed[..., None], ref, 0.5), "Observed"),
        ])
    panels.extend((im, f"Sample {i + 1}") for i, im in enumerate(x))
    columns = min(4, len(panels))
    rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(3 * columns, 3 * rows), squeeze=False
    )
    for ax in axes.flat:
        ax.axis("off")
    for ax, (im, title) in zip(axes.flat, panels):
        ax.imshow(np.clip(im, 0, 1))
        ax.set_title(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


PROBLEM = Problem(
    name="imagenet",
    label="ImageNet (512px)",
    make_dataset=None,
    make_model=make_model,
    train=None,
    make_constraint=make_constraint,
    plot=plot,
    save_figure=plot,
    retain_trajectory=False,
    unconstrained_gains={"dt": 0.002},
    method_constraint_options={
        name: {"reduction": "norm"}
        for name in ("pigdm", "pcfm", "opc", "dps")
    },
    method_gains={
        "pcfm": {
            "num_steps": 10, "num_projection_iters": 1,
            "correction_weight": 0.,
            "num_correction_iters": 1,
            "num_final_projection_iters": 0,
        },
        "ldf": {
            "penalty_weight": 40., "dt": 0.0005,
            "rescale_factor": 1., "rescale_exponent": 1.,
            "initial_dual_scale": -1., "save_trajectory": False,
            "num_projection_iters": 0,
        },
        "pigdm": {
            "dt": 0.0005, "guidance_scale": 0.6, "eps_reg": 1e-8,
            "save_trajectory": False,
        },
        # PiGDM with the variance table estimated from samples of this
        # checkpoint (experiments/imagenet_posterior_variance.py). That r_t^2
        # is far below PiGDM's prior, so the guidance is stiff: adaptive
        # Tsit5 exceeds 2000 steps, and PiGDM's scale 0.6 diverges with
        # fixed steps. Midpoint steps match DPS's 1000 evaluations; the
        # scale is the best of {0.6, ..., 0.0006} by missing-region FID on
        # five held-aside calibration cases.
        "opc": {
            "dt": 0.002, "guidance_scale": 0.002, "eps_reg": 1e-8,
            "save_trajectory": False,
            "posterior_variance":
                "experiments/posterior_variance/imagenet.json",
        },
        # 500 Midpoint steps: 1000 network evaluations, as in DPS. The step
        # size is the best of {0.1, ..., 10} by missing-region FID on the
        # same five calibration cases.
        "dps": {"dt": 0.002, "step_size": 0.1, "save_trajectory": False},
    },
    supported_methods=("ldf", "pigdm", "pcfm", "opc", "dps"),
    options={
        "--reference": {"type": str, "default": None,
                        "help": "RGB reference image (center crop to 512px)."},
        "--mask": {"choices": ("interior", "center", "lower-half",
                               "lower-left"),
                   "default": "interior",
                   "help": "Region to inpaint (not the observed region)."},
        "--mask-path": {"type": str, "default": None,
                        "help": "512px mask: white observed, black missing."},
    },
    default_num_samples=1,
    checkpoint=CHECKPOINT_PATH,
)
