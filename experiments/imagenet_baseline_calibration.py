"""Calibrate DPS and OPC on the old five cases, as was done for PCFM.

    uv run --group evaluation -m experiments.imagenet_baseline_calibration \
        --stage run

Each candidate is a separate five-case study initialized from the original
gallery (same references, masks, and sampler seeds as the PCFM
calibration). DPS varies its step size zeta'; OPC varies its guidance
scale (PiGDM's 0.6 is unstable with OPC's much smaller r_t^2 at this step).
All candidates and cases are retained. Selection is by missing-region FID
on this small calibration set; these are not held-out performance estimates.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.imagenet_inpainting_gallery import write_json

GRIDS = {
    "dps": ("step_size", (0.1, 0.3, 1.0, 3.0, 10.0)),
    "opc": ("guidance_scale", (0.6, 0.2, 0.06, 0.02, 0.006, 0.002, 0.0006)),
}


def name(method, value):
    return f"{method}_{value:g}".replace(".", "p")


def run(output_root, source_gallery, scratch):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="false")
    candidates = [(name(method, v), method, {key: v})
                  for method, (key, values) in GRIDS.items() for v in values]
    for label, method, overrides in candidates:
        root = Path(output_root) / label
        command = [sys.executable, "-m",
                   "experiments.imagenet_inpainting_study",
                   "--output-dir", str(root)]
        subprocess.run([*command, "--stage", "init", "--source-gallery",
                        source_gallery], env=env, check=True)
        path = Path(scratch) / f"{label}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, overrides)
        subprocess.run([*command, "--stage", "extend", "--methods", method,
                        f"--{method}-settings", str(path)], env=env, check=True)
        for stage in ("references", "pcfm", "pigdm", "ldf", method,
                      "evaluate"):
            subprocess.run([*command, "--stage", stage, "--report-dir",
                            str(root / "report")], env=env, check=True)


def summarize(output_root, output):
    output_root, output = Path(output_root), Path(output)
    trials, selected = [], {}
    for method, (key, values) in GRIDS.items():
        for value in values:
            result = json.loads((output_root / name(method, value)
                                 / "report/results.json").read_text())
            row = next(r for r in result["rows"] if r["method"] == method)
            trials.append({"name": name(method, value), "method": method,
                           key: value, "metrics": row})
        complete = [t for t in trials if t["method"] == method
                    and t["metrics"]["successful"] == t["metrics"]["planned"]]
        selected[method] = min(
            complete, key=lambda t: t["metrics"]["fid_hole"])["name"]
    result = {
        "selection_rule": "Minimum missing-region FID across all five cases",
        "role": "Calibration only; the 256-case study is separate",
        "selected": selected, "trials": trials,
        "notes": [
            "Same five cases, masks, and seeds as the PCFM calibration.",
            "Both use 500 Midpoint steps (1000 network evaluations).",
            "No calibration cases or completed candidates were dropped.",
            "Runs overlapped other GPU jobs; runtimes are not comparable.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), result)
    with output.with_suffix(".csv").open("w") as stream:
        rows = [{k: v for k, v in t.items() if k != "metrics"}
                | t["metrics"] for t in trials]
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for t in trials:
        m = t["metrics"]
        print(f"{t['name']:>10}  ok {m['successful']}/{m['planned']}  "
              f"FID full {m['fid_full']:.1f}  hole {m['fid_hole']:.1f}  "
              f"KID hole {1000 * m['paired_kid_hole']:.1f}  "
              f"known MAE {m['observed_mae']:.4f}")
    print("selected", selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "summarize"), required=True)
    parser.add_argument("--output-root",
                        default="results/imagenet_baseline_calibration")
    parser.add_argument("--source-gallery",
                        default="results/imagenet_native_gallery")
    parser.add_argument("--summary",
                        default="plots/imagenet_baseline_calibration")
    args = parser.parse_args()
    if args.stage == "run":
        run(args.output_root, args.source_gallery,
            Path(args.output_root) / "settings")
    summarize(args.output_root, args.summary)


if __name__ == "__main__":
    main()
