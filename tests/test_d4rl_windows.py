"""Cutting D4RL demonstrations into training windows (synthetic data only)."""

import numpy as np
import pytest

from cfm.datasets.d4rl_locomotion import (
    D4RLWindowDataset,
    dataset_path,
    download_url,
    episode_bounds,
    make_transitions,
    subsample_starts,
    window_starts,
)
from problems.locomotion_spec import HOPPER, WALKER2D

HORIZON = 4


def flags(length, done_at):
    """Boolean done-flags of the given length, set at the given indices."""
    f = np.zeros(length, dtype=bool)
    f[list(done_at)] = True
    return f


def episode_dataset(lengths, dim=3, horizon=HORIZON, max_windows=None):
    """Transitions where every episode is filled with its own constant."""
    total = sum(lengths)
    transitions = np.zeros((total, dim), dtype=np.float32)
    terminals = np.zeros(total, dtype=bool)
    start = 0
    for value, length in enumerate(lengths, start=1):
        transitions[start:start + length] = value
        terminals[start + length - 1] = True
        start += length
    return D4RLWindowDataset.from_arrays(
        transitions, terminals, np.zeros(total, dtype=bool),
        horizon=horizon, max_windows=max_windows,
    )


# --------------------------------------------------------------------------
# episode segmentation
# --------------------------------------------------------------------------


def test_episode_bounds_splits_at_terminals():
    bounds = episode_bounds(flags(10, [4, 9]), np.zeros(10, dtype=bool))
    np.testing.assert_array_equal(bounds, [[0, 5], [5, 10]])


def test_episode_bounds_splits_at_timeouts():
    bounds = episode_bounds(np.zeros(10, dtype=bool), flags(10, [4, 9]))
    np.testing.assert_array_equal(bounds, [[0, 5], [5, 10]])


def test_episode_bounds_handles_a_trailing_partial_episode():
    """A run with no final flag is still an episode."""
    bounds = episode_bounds(flags(10, [4]), np.zeros(10, dtype=bool))
    np.testing.assert_array_equal(bounds, [[0, 5], [5, 10]])


def test_episode_bounds_partitions_every_index():
    rng = np.random.default_rng(0)
    n = 40
    terminals = rng.random(n) < 0.2
    bounds = episode_bounds(terminals, rng.random(n) < 0.1)
    assert bounds[0, 0] == 0 and bounds[-1, 1] == n
    np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])


# --------------------------------------------------------------------------
# windowing
# --------------------------------------------------------------------------


def test_window_starts_stay_inside_one_episode():
    bounds = np.array([[0, 10], [10, 25]])
    starts = window_starts(bounds, HORIZON)
    for start in starts:
        stop = bounds[np.searchsorted(bounds[:, 1], start, "right"), 1]
        assert start + HORIZON <= stop


def test_window_starts_drop_short_episodes():
    """An episode shorter than the horizon contributes nothing."""
    assert len(window_starts(np.array([[0, HORIZON - 1]]), HORIZON)) == 0
    assert len(window_starts(np.array([[0, HORIZON]]), HORIZON)) == 1


def test_window_count_matches_the_formula():
    bounds = np.array([[0, 10], [10, 12], [12, 30]])
    expected = sum(
        max(0, (stop - start) - HORIZON + 1) for start, stop in bounds
    )
    assert len(window_starts(bounds, HORIZON)) == expected


# --------------------------------------------------------------------------
# subsampling
# --------------------------------------------------------------------------


def test_subsample_is_deterministic_sorted_and_unique():
    starts = np.arange(100)
    a = subsample_starts(starts, 20, seed=0)
    b = subsample_starts(starts, 20, seed=0)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 20 == len(np.unique(a))
    np.testing.assert_array_equal(a, np.sort(a))


def test_subsample_returns_everything_when_it_fits():
    starts = np.arange(10)
    np.testing.assert_array_equal(subsample_starts(starts, 50, 0), starts)
    np.testing.assert_array_equal(subsample_starts(starts, None, 0), starts)


def test_subsample_differs_across_seeds():
    starts = np.arange(1000)
    a = subsample_starts(starts, 50, seed=0)
    b = subsample_starts(starts, 50, seed=1)
    assert not np.array_equal(a, b)


# --------------------------------------------------------------------------
# transition layout
# --------------------------------------------------------------------------


def test_make_transitions_puts_actions_first():
    actions = np.arange(6, dtype=np.float32).reshape(2, 3)
    observations = np.arange(10, dtype=np.float32).reshape(2, 5)
    transitions = make_transitions(actions, observations)
    assert transitions.shape == (2, 8)
    np.testing.assert_array_equal(transitions[:, :3], actions)
    np.testing.assert_array_equal(transitions[:, 3:], observations)
    assert transitions.dtype == np.float32


# --------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------


def test_window_shape_and_dtype():
    data = episode_dataset([12, 9])
    assert data[0].shape == (HORIZON, 3)
    assert data[0].dtype.is_floating_point
    assert data.transition_dim == 3


def test_windows_are_contiguous_slices():
    data = episode_dataset([12])
    start = int(data.starts[2])
    np.testing.assert_array_equal(
        data[2].numpy(), data.transitions[start:start + HORIZON].numpy()
    )


def test_no_window_crosses_an_episode_boundary():
    """Each episode holds one constant, so a mixed window is a bug."""
    data = episode_dataset([12, 5, 9, 4])
    for i in range(len(data)):
        assert len(np.unique(data[i].numpy())) == 1


def test_getitem_returns_a_view_not_a_copy():
    data = episode_dataset([12])
    assert data[0].data_ptr() >= data.transitions.data_ptr()
    assert data[0].untyped_storage().data_ptr() == \
        data.transitions.untyped_storage().data_ptr()


def test_length_respects_max_windows():
    assert len(episode_dataset([50], max_windows=7)) == 7
    assert len(episode_dataset([50], max_windows=None)) == 50 - HORIZON + 1


def test_windows_stacks_at_most_what_exists():
    data = episode_dataset([12])
    assert data.windows(3).shape == (3, HORIZON, 3)
    assert data.windows(10_000).shape == (len(data), HORIZON, 3)


# --------------------------------------------------------------------------
# the mirror
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [WALKER2D, HOPPER],
                         ids=["walker2d", "hopper"])
def test_download_url_names_the_mirror_file(spec):
    assert download_url(spec.filename) == (
        "https://huggingface.co/datasets/imone/D4RL/resolve/main/"
        f"{spec.filename}"
    )
    assert dataset_path(spec.filename).parts[-2:] == ("d4rl", spec.filename)
