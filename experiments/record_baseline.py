"""Record pre-refactor benchmark numbers into ``experiments/baseline.json``.

Phase 0 of the refactor: every later phase asserts that violation is no worse
and wall-clock is no more than 5% slower than what this captures. Run once,
on the machine the comparison will be made on, and commit the result.

    uv run python -m experiments.record_baseline

Timings are machine-specific, so a baseline recorded elsewhere is only good
for the violation comparison.
"""

import json
import platform
import subprocess
import sys
from pathlib import Path

RAW_DIR = Path("experiments/baseline_raw")
OUT = Path("experiments/baseline.json")

# Mirrors benchmark.SUPPORTED_METHODS. Kept explicit here so the baseline
# records exactly what was run, even as the registry replaces that table.
CASES = [
    ("star", "ours"), ("star", "pigdm"), ("star", "pcfm"),
    ("mnist", "ours"), ("mnist", "pigdm"), ("mnist", "pcfm"),
    ("obstacle", "ours"), ("obstacle", "cbf"),
]

NUM_SAMPLES = 20


def git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for example, method in CASES:
        key = f"{example}/{method}"
        raw = RAW_DIR / f"{example}_{method}.json"
        print(f">>> {key}", flush=True)
        cmd = [
            sys.executable, "benchmark.py",
            "--example", example, "--method", method,
            "--num-samples", str(NUM_SAMPLES),
            "--out", str(raw),
        ]
        # The obstacle scene defaults differ between benchmark.py and the
        # example; benchmark.py's defaults are what the table uses.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(proc.stdout[-2000:])
            print(proc.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"benchmark failed for {key}")

        record = json.loads(raw.read_text())
        times = record["times_s"]
        viol = record["violations"]
        results[key] = {
            "example": example,
            "method": method,
            "num_samples": record["num_samples"],
            "mean_time_ms": 1000 * sum(times) / len(times),
            "min_time_ms": 1000 * min(times),
            "mean_violation": sum(viol) / len(viol),
            "max_violation": max(viol),
            "params": record["params"],
        }
        r = results[key]
        print(f"    {r['mean_time_ms']:8.2f} ms   "
              f"viol mean={r['mean_violation']:.3e} "
              f"max={r['max_violation']:.3e}", flush=True)

    payload = {
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "num_samples": NUM_SAMPLES,
        "results": results,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
