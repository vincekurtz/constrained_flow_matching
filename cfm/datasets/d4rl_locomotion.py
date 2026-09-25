"""Fixed-length windows of D4RL locomotion episodes.

Each sample is ``(horizon, action_dim + obs_dim)``, actions first (Diffuser
layout). The v2 hdf5 files come from the ``imone/D4RL`` HuggingFace mirror and
are cached under ``data/d4rl/``.
"""

import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

HF_BASE = "https://huggingface.co/datasets/imone/D4RL/resolve/main/{filename}"
DEFAULT_ROOT = "data/d4rl"
DOWNLOAD_CHUNK = 1 << 20


def download_url(filename: str) -> str:
    """Where a D4RL file lives on the mirror."""
    return HF_BASE.format(filename=filename)


def dataset_path(filename: str, root: str = DEFAULT_ROOT) -> Path:
    """Local cache path for a D4RL file."""
    return Path(root) / filename


def ensure_downloaded(filename: str, root: str = DEFAULT_ROOT) -> Path:
    """Return the cached file, fetching it from the mirror if necessary."""
    path = dataset_path(filename, root)
    if path.exists():
        return path

    url = download_url(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    print(f"Downloading {filename} from {url} ...")
    try:
        with urllib.request.urlopen(url) as response, \
                open(partial, "wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            done = 0
            while chunk := response.read(DOWNLOAD_CHUNK):
                out.write(chunk)
                done += len(chunk)
                print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="\r")
    except (urllib.error.URLError, OSError) as err:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {filename}: {err}. Fetch it by hand with\n"
            f"  curl -L --create-dirs -o {path} {url}"
        ) from err
    os.replace(partial, path)
    print(f"\nSaved {path}")
    return path


def make_transitions(
    actions: np.ndarray, observations: np.ndarray
) -> np.ndarray:
    """Concatenate to (N, action_dim + obs_dim), actions first."""
    return np.concatenate(
        [np.asarray(actions), np.asarray(observations)], axis=-1
    ).astype(np.float32)


def episode_bounds(
    terminals: np.ndarray, timeouts: np.ndarray
) -> np.ndarray:
    """Return (num_episodes, 2) ``[start, stop)`` pairs partitioning [0, N)."""
    done = np.asarray(terminals).astype(bool) | np.asarray(timeouts).astype(
        bool
    )
    stops = np.flatnonzero(done) + 1
    if len(stops) == 0 or stops[-1] != len(done):
        stops = np.append(stops, len(done))
    starts = np.concatenate([[0], stops[:-1]])
    return np.stack([starts, stops], axis=-1)


def window_starts(bounds: np.ndarray, horizon: int) -> np.ndarray:
    """Every start index whose window fits inside a single episode."""
    starts = [
        np.arange(start, stop - horizon + 1)
        for start, stop in np.asarray(bounds)
        if stop - start >= horizon
    ]
    if not starts:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(starts).astype(np.int64)


def subsample_starts(
    starts: np.ndarray, max_windows: Optional[int], seed: int = 0
) -> np.ndarray:
    """Keep a sorted random subset of at most ``max_windows`` starts."""
    starts = np.asarray(starts)
    if max_windows is None or len(starts) <= max_windows:
        return starts
    rng = np.random.default_rng(seed)
    keep = rng.choice(len(starts), size=max_windows, replace=False)
    return np.sort(starts[keep])


def load_hdf5(
    path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read (actions, observations, terminals, timeouts) from a D4RL file."""
    import h5py  # lazy, so tests of the helpers above don't need it

    with h5py.File(path, "r") as f:
        actions = np.asarray(f["actions"], dtype=np.float32)
        observations = np.asarray(f["observations"], dtype=np.float32)
        terminals = np.asarray(f["terminals"], dtype=bool)
        timeouts = np.asarray(f["timeouts"], dtype=bool)
    return actions, observations, terminals, timeouts


class D4RLWindowDataset(Dataset):
    """Windows of D4RL demonstrations, sliced lazily from the transitions.

    Args:
        filename: file on the mirror, e.g. "hopper_medium_expert-v2.hdf5".
        max_windows: cap on the number of windows, or None for all.
        root: cache directory.
        action_dim, obs_dim: expected widths, checked when given.
    """

    def __init__(
        self,
        filename: str,
        horizon: int = 32,
        max_windows: Optional[int] = 32768,
        seed: int = 0,
        root: str = DEFAULT_ROOT,
        action_dim: Optional[int] = None,
        obs_dim: Optional[int] = None,
    ):
        super().__init__()
        path = ensure_downloaded(filename, root)
        actions, observations, terminals, timeouts = load_hdf5(path)

        if action_dim is not None and actions.shape[1] != action_dim:
            raise ValueError(
                f"{filename} has {actions.shape[1]} action dimensions, "
                f"expected {action_dim}"
            )
        if obs_dim is not None and observations.shape[1] != obs_dim:
            raise ValueError(
                f"{filename} has {observations.shape[1]} observation "
                f"dimensions, expected {obs_dim}"
            )

        self._build(
            make_transitions(actions, observations),
            terminals,
            timeouts,
            horizon,
            max_windows,
            seed,
        )

    @classmethod
    def from_arrays(
        cls,
        transitions: np.ndarray,
        terminals: np.ndarray,
        timeouts: np.ndarray,
        horizon: int = 32,
        max_windows: Optional[int] = 32768,
        seed: int = 0,
    ) -> "D4RLWindowDataset":
        """Build a dataset from in-memory arrays, skipping the download."""
        dataset = cls.__new__(cls)
        Dataset.__init__(dataset)
        dataset._build(
            transitions, terminals, timeouts, horizon, max_windows, seed
        )
        return dataset

    def _build(
        self, transitions, terminals, timeouts, horizon, max_windows, seed
    ):
        self.horizon = horizon
        self.transitions = torch.as_tensor(
            np.ascontiguousarray(transitions, dtype=np.float32)
        )
        bounds = episode_bounds(terminals, timeouts)
        starts = subsample_starts(
            window_starts(bounds, horizon), max_windows, seed
        )
        self.starts = torch.as_tensor(starts, dtype=torch.int64)

    @property
    def transition_dim(self) -> int:
        """Width of one timestep."""
        return self.transitions.shape[1]

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> torch.Tensor:
        start = int(self.starts[idx])
        return self.transitions[start:start + self.horizon]

    def windows(self, num: Optional[int] = None) -> torch.Tensor:
        """Stack the first ``num`` windows, for plotting against samples."""
        count = len(self) if num is None else min(num, len(self))
        return torch.stack([self[i] for i in range(count)])


if __name__ == "__main__":
    # Report how often the data violates the height constraint.
    import argparse

    from problems.locomotion_spec import HORIZON, SPECS, height_residual

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=sorted(SPECS), default="walker2d")
    parser.add_argument("--height-limit", type=float, default=None)
    parser.add_argument("--phi", type=float, default=None)
    parser.add_argument("--num-windows", type=int, default=8192)
    args = parser.parse_args()

    spec = SPECS[args.env]
    limit = spec.height_limit if args.height_limit is None \
        else args.height_limit
    phi = spec.phi if args.phi is None else args.phi

    data = D4RLWindowDataset(
        spec.filename,
        horizon=HORIZON,
        max_windows=args.num_windows,
        action_dim=spec.action_dim,
        obs_dim=spec.obs_dim,
    )
    x = data.windows().numpy()
    z = x[..., spec.z_index]
    vz = x[..., spec.vz_index]
    residual = height_residual(x, spec.z_index, spec.vz_index, limit, phi)
    worst = np.max(residual, axis=-1)

    print(f"{spec.label}: {len(data)} windows of {HORIZON} steps, "
          f"transition dim {data.transition_dim}")
    for name, values in (("z", z), ("vz", vz), ("z + phi vz", z + phi * vz)):
        qs = np.percentile(values, [1, 50, 99, 100])
        print(f"  {name:>12}  p1={qs[0]:+.3f}  p50={qs[1]:+.3f}  "
              f"p99={qs[2]:+.3f}  max={qs[3]:+.3f}")
    print(f"  roof h_r={limit:g}, phi={phi:g}")
    print(f"  windows violating: {np.mean(worst > 0):.1%}")
    print(f"  timesteps violating: {np.mean(residual > 0):.1%}")
