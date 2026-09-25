"""Pin the numerical output of every generator.

If an algorithm change is intended, regenerate with
``uv run python -m tests.make_goldens``.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.golden_cases import CASES, GOLDEN_DIR

RTOL = 1e-6
ATOL = 1e-6


@pytest.mark.parametrize("name", sorted(CASES), ids=str)
def test_matches_golden(name):
    path = Path(GOLDEN_DIR) / f"{name}.npy"
    assert path.exists(), (
        f"missing golden {path}; run `uv run python -m tests.make_goldens`"
    )
    expected = np.load(path)
    actual = np.asarray(CASES[name]())

    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual)), f"{name} produced non-finite values"
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_every_case_has_a_golden():
    on_disk = {p.stem for p in Path(GOLDEN_DIR).glob("*.npy")}
    assert on_disk == set(CASES), (
        f"cases without goldens: {set(CASES) - on_disk}; "
        f"goldens without cases: {on_disk - set(CASES)}"
    )
