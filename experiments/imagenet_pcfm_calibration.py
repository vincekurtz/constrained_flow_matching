"""Reproduce PCFM calibration on the old five cases, before held-out study.

All candidates and cases are retained. Select by missing-region FID on this
small calibration set; these scores are not held-out performance estimates.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.imagenet_inpainting_gallery import write_json

SETTINGS = Path(__file__).parent / "imagenet_native/pcfm"


def summarize(root, output):
    root, output = Path(root), Path(output)
    trials = []
    identity = None
    for settings_path in sorted(SETTINGS.glob("*.json")):
        name = settings_path.stem
        result = json.loads((root / name / "report/results.json").read_text())
        plan = result["plan"]
        current = {k: plan[k] for k in (
            "checkpoint_sha256", "source", "cases",
        )}
        if identity is not None and current != identity:
            raise ValueError("Calibration candidates use different inputs")
        identity = current
        row = next(r for r in result["rows"] if r["method"] == "pcfm")
        if row["successful"] != row["planned"]:
            raise ValueError(f"Incomplete calibration candidate: {name}")
        trials.append({"name": name, "settings": result["plan"]["settings"][
            "pcfm"], "metrics": row})
    selected = min(trials, key=lambda trial: trial["metrics"]["fid_hole"])
    result = {
        "selection_rule": "Minimum missing-region FID across all five cases",
        "role": "Calibration only; the new 256-case study is separate",
        "selected": selected["name"], "trials": trials, "inputs": identity,
        "notes": [
            "The grid was expanded after initial calibration failures.",
            "No calibration cases or completed candidates were dropped.",
            "Ten corrections needed JAX simplified constants to compile.",
            "PCFM equations are unchanged; norm equivalence is tested.",
            "Different sampler compute budgets are reported, not equated.",
            "Calibration runtimes include overlapping GPU jobs.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), result)
    with output.with_suffix(".csv").open("w") as stream:
        rows = [{"name": t["name"], **t["settings"], **t["metrics"]}
                for t in trials]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(selected, indent=2))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("run", "summarize"), required=True)
    parser.add_argument("--output-root",
                        default="results/imagenet_pcfm_calibration")
    parser.add_argument("--source-gallery",
                        default="results/imagenet_native_gallery")
    parser.add_argument("--summary", default="plots/imagenet_pcfm_calibration")
    args = parser.parse_args()
    if args.stage == "run":
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
                   JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="true")
        for path in sorted(SETTINGS.glob("*.json")):
            root = Path(args.output_root) / path.stem
            command = [
                sys.executable, "-m", "experiments.imagenet_inpainting_study",
                "--output-dir", str(root),
            ]
            subprocess.run([*command, "--stage", "init", "--source-gallery",
                            args.source_gallery, "--pcfm-settings", str(path)],
                           env=env, check=True)
            subprocess.run([*command, "--stage", "all", "--report-dir",
                            str(root / "report")], env=env, check=True)
    summarize(args.output_root, args.summary)


if __name__ == "__main__":
    main()
