"""Pin the numerical output of every generator.

These are the refactor's safety net: each case in ``tests/golden_cases`` is
re-run and compared against the array it produced on the pre-refactor code.
A failure here means an algorithm changed, which during a pure restructuring
is always a bug.

If a change to the algorithms is intended, regenerate with
``uv run python -m tests.make_goldens`` and review the array diff.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.golden_cases import CASES, GOLDEN_DIR

# Phase 1 is a pure code move and should reproduce bit-for-bit. Later phases
# reassociate float ops (e.g. merging the equality and inequality flows), so
# the check is tight-but-not-exact.
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
    # A golden full of NaN would silently pass an allclose with equal_nan, so
    # finiteness is asserted separately and up front.
    assert np.all(np.isfinite(actual)), f"{name} produced non-finite values"
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_every_case_has_a_golden():
    """No case may be added without recording its baseline."""
    on_disk = {p.stem for p in Path(GOLDEN_DIR).glob("*.npy")}
    assert on_disk == set(CASES), (
        f"cases without goldens: {set(CASES) - on_disk}; "
        f"goldens without cases: {on_disk - set(CASES)}"
    )
