"""Regenerate the golden arrays in ``tests/goldens/``.

Run only when a change to the algorithms is *intended*:

    uv run python -m tests.make_goldens

Re-running this after an unintended behavior change would paper over exactly
what the goldens exist to catch, so the diff on tests/goldens/*.npy should be
scrutinised in review like any other change.
"""

from pathlib import Path

import numpy as np

from tests.golden_cases import CASES, GOLDEN_DIR


def main():
    out_dir = Path(GOLDEN_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in CASES.items():
        x = np.asarray(fn())
        path = out_dir / f"{name}.npy"
        status = "updated" if path.exists() else "created"
        np.save(path, x)
        finite = "ok" if np.all(np.isfinite(x)) else "HAS NON-FINITE VALUES"
        print(f"  {status:8s} {path}  shape={x.shape}  {finite}")


if __name__ == "__main__":
    main()
