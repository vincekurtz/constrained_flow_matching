"""Single-case inpainting figure: every method with and without projection.

    uv run -m experiments.imagenet_inpainting_projection [--case 107]

Top row: each method's raw output, as in the study. Bottom row: the same
output after a final projection onto the constraint set. The observed-pixel
constraint is linear and selects coordinates, so one Gauss-Newton step (the
projection used in Table 1) is exact and equals copying the observed pixels
into the output; that is what is applied here. PCFM already projects at every
step, so its two panels coincide.

Writes ``plots/figures/imagenet_inpainting_projection.{png,pdf}``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments import imagenet_inpainting_gallery as gallery
from experiments.imagenet_inpainting_paper import (
    FIGURE_LABELS, ORDER, RC, TEXT_WIDTH,
)
from experiments.imagenet_inpainting_study import load_sample

DEFAULT_CASE = 107

def projected(image, reference, observed):
    """Exact projection onto x[observed] = reference[observed]."""
    return np.where(observed[..., None], reference, image)


def panels(root, plan, case):
    reference, _ = load_sample(root, plan, case, "references")
    observed = gallery.observed_mask(case)
    raw = {m: load_sample(root, plan, case, m)[0] for m in ORDER}
    proj = {m: projected(x, reference, observed) for m, x in raw.items()}
    violation = {m: float(np.abs(x - reference)[observed].mean())
                 for m, x in raw.items()}
    masked = np.where(observed[..., None], reference, 0.5)
    return reference, masked, raw, proj, violation


def figure(root, plan, case, output, dpi=600):
    reference, masked, raw, proj, violation = panels(root, plan, case)
    gap, split = 0.04, 0.30  # inches between panels; after the input column
    title = 0.16
    side = (TEXT_WIDTH - split - gap * (len(ORDER) - 1)) / (len(ORDER) + 1)
    height = title + 2 * side + gap
    # The input column titles both of its panels, so they are smaller.
    small = side - (title - gap) / 2
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(TEXT_WIDTH, height))

        def axes(x, bottom, size=side):
            ax = fig.add_axes((x / TEXT_WIDTH, bottom / height,
                               size / TEXT_WIDTH, size / height))
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return ax

        def show(ax, image):
            ax.imshow(np.clip(image, 0, 1), interpolation="lanczos")

        for bottom, image, name in ((height - title - small, reference,
                                     "Reference"),
                                    (0, masked, "Masked input")):
            ax = axes((side - small) / 2, bottom, small)
            show(ax, image)
            ax.set_title(name)
        rows = [height - title - row * (side + gap) - side for row in (0, 1)]
        start = side + split
        for col, method in enumerate(ORDER):
            x = start + col * (side + gap)
            top = axes(x, rows[0])
            show(top, raw[method])
            top.set_title(FIGURE_LABELS[method])
            top.text(0.03, 0.03, f"viol. {sci_text(violation[method])}",
                     transform=top.transAxes, ha="left", va="bottom",
                     fontsize=6, color="white",
                     bbox={"facecolor": "black", "alpha": 0.55, "pad": 1.2,
                           "linewidth": 0})
            show(axes(x, rows[1]), proj[method])
        for bottom, name in zip(rows, ("Raw output", "+ projection")):
            fig.text((start - 0.04) / TEXT_WIDTH, (bottom + side / 2) / height,
                     name, rotation=90, ha="right", va="center")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output.with_suffix(".png"), dpi=dpi)
        if dpi >= 300:
            fig.savefig(output.with_suffix(".pdf"), dpi=300)
        plt.close(fig)


def sci_text(value):
    """Compact scientific notation for an in-panel label."""
    if value < 1e-6:
        return r"$<10^{-6}$"
    mantissa, exponent = f"{value:.0e}".split("e")
    return rf"${mantissa}\!\times\!10^{{{int(exponent)}}}$"


def boundary_jump(image, case):
    """Mean absolute pixel jump across the hole's edge."""
    top, left, h, w = case["hole"]
    jumps = []
    if top > 0:
        jumps.append(np.abs(image[top] - image[top - 1])[left:left + w])
    if top + h < image.shape[0]:
        jumps.append(np.abs(image[top + h - 1] - image[top + h])[left:left + w])
    if left > 0:
        jumps.append(np.abs(image[:, left] - image[:, left - 1])[top:top + h])
    if left + w < image.shape[1]:
        jumps.append(np.abs(image[:, left + w - 1]
                            - image[:, left + w])[top:top + h])
    return float(np.mean(np.concatenate(jumps)))


def rank(root, plan, count):
    """Cases where projected baselines show seams and LDF does not.

    Score: mean excess boundary jump (over the reference's own) of the
    projected PiGDM/OPC/DPS outputs, minus LDF's.
    """
    scores = []
    for case in plan["cases"]:
        try:
            reference, _, raw, proj, violation = panels(root, plan, case)
        except (FileNotFoundError, ValueError):
            continue  # a method failed on this case
        base = boundary_jump(reference, case)
        excess = {m: boundary_jump(np.clip(proj[m], 0, 1), case) - base
                  for m in ORDER}
        score = (np.mean([excess[m] for m in ("pigdm", "opc", "dps")])
                 - abs(excess["ldf"]))
        scores.append((score, case["id"], violation, excess))
    scores.sort(key=lambda s: -s[0])
    for score, case_id, violation, excess in scores[:count]:
        print(f"case {case_id}: score {score:.4f}  " + "  ".join(
            f"{m} {violation[m]:.1e}/{excess[m]:+.3f}" for m in ORDER))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",
                        default="results/imagenet_study_256_selected")
    parser.add_argument("--case", type=int, default=DEFAULT_CASE)
    parser.add_argument("--rank", type=int, default=0,
                        help="Print the top N candidate cases and exit")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--figure", default="plots/figures/imagenet_inpainting_projection")
    args = parser.parse_args()
    root = Path(args.study)
    plan = json.loads((root / "plan.json").read_text())
    if args.rank:
        rank(root, plan, args.rank)
        return
    case = next(c for c in plan["cases"] if c["id"] == args.case)
    figure(root, plan, case, Path(args.figure), dpi=args.dpi)
    print(f"wrote {args.figure}.png" + ("/.pdf" if args.dpi >= 300 else ""))


if __name__ == "__main__":
    main()
