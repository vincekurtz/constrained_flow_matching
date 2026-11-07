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
from cfm.models.image_coordinates import (
    ImageCoordinateFlow, ImageCoordinateNormalizer,
)
from problems import Problem

IMAGE_SHAPE = (512, 512, 3)
REFERENCE_PATH = Path(__file__).parent / "assets/imagenet_reference.png"
CHECKPOINT_PATH = "data/imagenet_model.pkl"


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
    """Linear fixed-pixel equalities, optionally reduced exactly for PiGDM.

    LDF uses r=x[observed]-reference[observed], one multiplier per RGB value.
    PiGDM uses ||r||=0: the same feasible set and, under a scalar normalizer,
    the same correction as the full vector r, with only one Jacobian row.
    This reduction is specific to PiGDM; it is not equivalent LDF dynamics.
    """
    if reduction not in ("pixels", "norm"):
        raise ValueError(f"Unknown fixed-pixel reduction: {reduction}")
    observed = np.broadcast_to(np.asarray(observed)[..., None], reference.shape)
    indices = jnp.asarray(np.flatnonzero(observed))
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


def adapt_model(model, normalizer, method_name):
    """LDF constrains image coordinates; PiGDM keeps its native estimate."""
    if method_name == "ldf":
        return ImageCoordinateFlow(model), ImageCoordinateNormalizer(normalizer)
    return model, normalizer


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
    adapt_model=adapt_model,
    retain_trajectory=False,
    unconstrained_gains={"dt": 0.001},
    method_constraint_options={"pigdm": {"reduction": "norm"}},
    method_gains={
        "ldf": {
            "penalty_weight": 17., "dt": 0.00125,
            "rescale_factor": 1. / 289., "rescale_exponent": 2.,
            "num_projection_iters": 0,
        },
        "pigdm": {
            "dt": 0.0005, "guidance_scale": 0.6, "eps_reg": 1e-8,
        },
    },
    supported_methods=("ldf", "pigdm"),
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
