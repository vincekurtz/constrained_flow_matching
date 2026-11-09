"""Run and export Table 1, explicitly identifying unavailable cases."""

import argparse
import csv
import json
import math
from pathlib import Path

import problems
from cfm import sweep


def export_table(*, run=False, available_only=False, force=False):
    cases = sweep.build_cases(sweep.load_config("experiments/table1.toml"))
    missing_models = {
        c.problem: problems.get(c.problem).checkpoint_path for c in cases
        if not Path(problems.get(c.problem).checkpoint_path).exists()
    }
    if run and missing_models and not available_only:
        raise FileNotFoundError(
            f"Missing checkpoints: {missing_models}. Supply them or use "
            "--available-only to explicitly export an incomplete table."
        )
    records, pending = [], []
    for case in cases:
        path = sweep.RESULTS_DIR / f"{case.key}.json"
        if (run and case.problem not in missing_models
                and (force or not path.exists())):
            print(f"Running {case.label}", flush=True)
            sweep.run_case(case)
        if path.exists():
            records.append(json.loads(path.read_text()))
        else:
            pending.append(case.label)
    if not records:
        raise ValueError("No cached Table 1 results; rerun with --run")
    metadata = {
        "complete": not pending, "pending_cases": pending,
        "missing_checkpoints": missing_models,
        "notes": [
            "Each new record includes the runtime device and checkpoint "
            "hash. Compare timings only on matching hardware.",
            "Nominal steps specify dt=1/steps. The time-varying penalty "
            "stops at 1-dt to exclude the singular endpoint.",
            "Null violations indicate numerical failure, not missing runs.",
        ],
    }

    def finite_json(value):
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    data = Path("plots/data/table1.json")
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps(
        {**metadata, "records": finite_json(records)}, indent=2,
        allow_nan=False,
    ) + "\n")
    note = (f"% INCOMPLETE: {len(pending)} cases pending; "
            "see plots/data/table1.json.\n"
            if pending else "")
    Path("plots/table1.tex").write_text(
        note + "% Local timings; see plots/data/table1.json for provenance.\n"
        + sweep.render_latex_table(records) + "\n"
    )
    with Path("plots/table1.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Problem", "Steps", "Method", "Time (ms)",
                         "Mean violation", "Failed samples"])
        for r in records:
            writer.writerow([r["problem_label"], r["steps"],
                             r["method_label"], r["mean_time_ms"],
                             finite_json(r["mean_violation"]), r["num_nan"]])
    print(f"Exported {len(records)}/{len(cases)} cases; "
          f"{len(pending)} pending. See plots/table1.tex and plots/table1.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--available-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    export_table(**vars(parser.parse_args()))
